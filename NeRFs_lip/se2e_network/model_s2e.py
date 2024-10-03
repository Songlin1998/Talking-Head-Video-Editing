# The model, audio to expression network

import os
import torch
import torch.nn as nn
import math

class ExpressionEstimator_Attention(nn.Module):
  def __init__(self, device = 'cuda', nIdentities=None, seq_len=9, subspace_dim=32, n_acoustic=64, n_expression=53, use_dropout=False):
    super().__init__()
    self.seq_len = seq_len
    self.subspace_dim = subspace_dim  # number of audio expressions
    self.n_expression = n_expression  # expression dim
    self.device = device

    self.convNet = nn.Sequential(
      nn.Conv1d(n_acoustic, 32, kernel_size = 3, stride = 1, padding = 1, bias=True),  # b * c (n_acoustic) * seq_len -> b * 32 * seq_len
      nn.LeakyReLU(0.02, True),
      nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
      nn.LeakyReLU(0.02, True),
      nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
      nn.LeakyReLU(0.02, True),
      nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1, bias=True),
      nn.LeakyReLU(0.02, True)
    )

    self.fullNet = nn.Sequential(
      nn.Linear(in_features=64, out_features=128, bias=True),
      nn.LeakyReLU(0.02),

      nn.Linear(in_features=128, out_features=64, bias=True),
      nn.LeakyReLU(0.02),

      nn.Linear(in_features=64, out_features=self.subspace_dim, bias=True),
      nn.Tanh()
    )

    # mapping from subspace to full expression space
    if nIdentities is not None:
      self.register_parameter('mapping', torch.nn.Parameter(
        torch.randn(1, nIdentities, self.n_expression, self.subspace_dim, requires_grad=True)))

    # attention
    self.attentionConvNet = nn.Sequential(  # b x subspace_dim x seq_len
      nn.Conv1d(self.subspace_dim, 16, kernel_size=3, stride=1, padding=1, bias=True),
      nn.LeakyReLU(0.02, True),
      nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1, bias=True),
      nn.LeakyReLU(0.02, True),
      nn.Conv1d(8, 4, kernel_size=3, stride=1, padding=1, bias=True),
      nn.LeakyReLU(0.02, True),
      nn.Conv1d(4, 2, kernel_size=3, stride=1, padding=1, bias=True),
      nn.LeakyReLU(0.02, True),
      nn.Conv1d(2, 1, kernel_size=3, stride=1, padding=1, bias=True),
      nn.LeakyReLU(0.02, True)
    )
    self.attentionNet = nn.Sequential(
      nn.Linear(in_features=self.seq_len, out_features=self.seq_len, bias=True),
      nn.Softmax(dim=1)
    )
    # self.hidden2subspace = nn.Linear(self.subspace_dim,self.subspace_dim)

  def forward_internal(self, audio_features_sequence, identity_id):
    result_subspace, intermediate_expression = self.getAudioExpressions_internal(audio_features_sequence)
    mapping = torch.index_select(self.mapping[0], dim=0, index=identity_id)
    result = 10.0 * torch.bmm(mapping, result_subspace)[:, :, 0]
    result_intermediate = 10.0 * torch.bmm(mapping, intermediate_expression)[:, :, 0]

    # print(torch.sum(self.mapping))
    return result, result_intermediate

  def forward(self, audio_features_sequence, identity_id):
    result_subspace = self.getAudioExpressions(audio_features_sequence)
    mapping = torch.index_select(self.mapping[0], dim=0, index=identity_id)
    result = torch.bmm(mapping, result_subspace)[:, :, 0]
    return 10.0 * result

  def getAudioExpressions_internal(self, audio_features_sequence):
    # audio_features_sequence: b x seq_len x 26
    b = audio_features_sequence.shape[0]
    conv_res = self.convNet(audio_features_sequence.transpose(1, 2))  # b * seq_len  x 64
    conv_res = torch.reshape(conv_res, (b * self.seq_len, 1, -1))  # b * seq_len  x 1 x 64
    result_subspace = self.fullNet(conv_res)[:, 0, :]  # b * seq_len x subspace_dim
    result_subspace = result_subspace.view(b, self.seq_len, self.subspace_dim)  # b x seq_len x subspace_dim

    #################
    ### attention ###
    #################
    result_subspace_T = torch.transpose(result_subspace, 1, 2)  # b x subspace_dim x seq_len
    intermediate_expression = result_subspace_T[:, :, (self.seq_len // 2):(self.seq_len // 2) + 1]
    att_conv_res = self.attentionConvNet(result_subspace_T)
    attention = self.attentionNet(att_conv_res.view(b, self.seq_len)).view(b, self.seq_len, 1)  # b x seq_len x 1
    # pooling along the sequence dimension
    result_subspace = torch.bmm(result_subspace_T, attention)

    return result_subspace.view(b, self.subspace_dim, 1), intermediate_expression
    # return    b x subdim x 1,   b x subdim x 1

  def getAudioExpressions(self, audio_features_sequence):
    expr, _ = self.getAudioExpressions_internal(audio_features_sequence)
    return expr

  def regist_mapping(self, input_mapping):
    self.register_parameter('mapping', torch.nn.Parameter(input_mapping))


  def inference(self, audio_feats, batch_size = 1024):
    # Inference a sequence of audio feats
    self.eval()
    # Start inference
    predict_feats = []
    with torch.no_grad():
      for j in range(math.ceil(audio_feats.shape[0] / batch_size)):
        start = j * batch_size
        end = (j + 1) * batch_size
        batch_input = audio_feats[start:end, ...].to(self.device)
        subspace_index = torch.zeros(batch_input.shape[0]).long().to(self.device)
        predict_feat = self.forward(batch_input, subspace_index)
        predict_feats.append(predict_feat)

      predict_expressions = torch.cat(predict_feats, 0)
    return predict_expressions


class Loss(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, fake_expressions, fake_expressions_intermediate, expressions, fake_expressions_prv, expressions_prv,
              fake_expressions_nxt, expressions_nxt):

    diff_expression = fake_expressions - expressions
    diff_expression_intermediate = fake_expressions_intermediate - expressions
    diff_expression_prv = fake_expressions_prv - expressions_prv
    diff_expression_nxt = fake_expressions_nxt - expressions_nxt
    # relative (temporal 1 timestep) cur - nxt and prv - cur
    diff_expression_tmp_cur_nxt = (fake_expressions - fake_expressions_nxt) - (
        expressions - expressions_nxt)
    diff_expression_tmp_prv_cur = (fake_expressions_prv - fake_expressions) - (
        expressions_prv - expressions)
    # relative (temporal 2 timesteps)  nxt - prv
    diff_expression_tmp_nxt_prv = (fake_expressions_nxt - fake_expressions_prv) - (
        expressions_nxt - expressions_prv)

    loss_G_L1_ABSOLUTE = 0.0
    loss_G_L1_RELATIVE = 0.0

    loss_G_L1_ABSOLUTE += 1000.0 * torch.sqrt(torch.mean(diff_expression * diff_expression))

    loss_G_L1_ABSOLUTE += 1000.0 * torch.sqrt(torch.mean(diff_expression_prv * diff_expression_prv))
    loss_G_L1_ABSOLUTE += 1000.0 * torch.sqrt(torch.mean(diff_expression_nxt * diff_expression_nxt))
    loss_G_L1_ABSOLUTE += 3000.0 * torch.sqrt(
      torch.mean(diff_expression_intermediate * diff_expression_intermediate))

    loss_G_L1_RELATIVE += 1000.0 * torch.sqrt(
      torch.mean(diff_expression_tmp_cur_nxt * diff_expression_tmp_cur_nxt))
    loss_G_L1_RELATIVE += 1000.0 * torch.sqrt(
      torch.mean(diff_expression_tmp_prv_cur * diff_expression_tmp_prv_cur))
    loss_G_L1_RELATIVE += 1000.0 * torch.sqrt(
      torch.mean(diff_expression_tmp_nxt_prv * diff_expression_tmp_nxt_prv))

    loss_G_L1 = loss_G_L1_ABSOLUTE + loss_G_L1_RELATIVE

    return loss_G_L1, loss_G_L1_ABSOLUTE, loss_G_L1_RELATIVE

