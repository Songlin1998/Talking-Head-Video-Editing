import sys
import math
import torch
from torch import nn
from se2e_network.operations import *
from torch.nn import functional as F


DEFAULT_MAX_SOURCE_POSITIONS = 20000
DEFAULT_MAX_TARGET_POSITIONS = 20000

OPERATIONS_ENCODER = {  # c = hidden size
    1: lambda c, dropout: EncConvLayer(c, 1, dropout),  # h, num_heads, dropout
    2: lambda c, dropout: EncConvLayer(c, 5, dropout),
    3: lambda c, dropout: EncConvLayer(c, 9, dropout),
    4: lambda c, dropout: EncConvLayer(c, 13, dropout),
    5: lambda c, dropout: EncConvLayer(c, 17, dropout),
    6: lambda c, dropout: EncConvLayer(c, 21, dropout),
    7: lambda c, dropout: EncConvLayer(c, 25, dropout),
    8: lambda c, dropout: EncSALayer(c, 2, dropout=dropout,
                                     attention_dropout=0.0, relu_dropout=dropout,
                                     kernel_size=9,
                                     padding='SAME'),
    9: lambda c, dropout: EncSALayer(c, 4, dropout),
    10: lambda c, dropout: EncSALayer(c, 8, dropout),
    11: lambda c, dropout: EncLocalSALayer(c, 2, dropout),
    12: lambda c, dropout: EncLSTMLayer(c, dropout),
    13: lambda c, dropout, g_bias, tao: EncGausSALayer(c, 1, dropout, gaus_bias=g_bias, gaus_tao=tao),
    14: lambda c, dropout: EncSALayer(c, 2, dropout, kernel_size=1),
    15: lambda c, dropout: EncSALayer(c, 2, dropout, kernel_size=15),
}


class Conv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, use_dropout=False, type_act='leakyrelu'):
        super().__init__()
        if type_act == 'leakyrelu':
            actn = nn.LeakyReLU(0.02, True)
        elif type_act == 'tanh':
            actn = nn.Tanh()
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            actn
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x):
        x = self.conv(x)
        if self.use_dropout:
            x = self.dropout(x)
        
        return x

class AudioNet_ds(nn.Module):
    def __init__(self, batchsize=8, num_frame=51, n_acoustic=72): # input: [B, T, 16, 29] output: [B, T, 72]
        super(AudioNet_ds, self).__init__()
        self.batch_size = batchsize
        self.num_frame = num_frame
        
        self.encoder_conv = nn.Sequential(   
            # Conv1dBlock(16*29, 16*29, type_act='leakyrelu', stride=1),  # batchsize x (16x29) x 51
            Conv1dBlock(16*29, 8*29, type_act='leakyrelu', stride=1),  # batchsize x (8x29) x 51
            # Conv1dBlock(8*29, 8*29, type_act='leakyrelu', stride=1),  # batchsize x (8x29) x 51
            Conv1dBlock(8*29, 4*29, type_act='leakyrelu', stride=1),  # batchsize x (4x29) x 51
            # Conv1dBlock(4*29, 4*29, type_act='leakyrelu', stride=1),  # batchsize x (4x29) x 51
        )

        # self.fc = nn.Sequential( # batchsize x 51 x 72
        #     nn.Linear(4*29, 72),
        #     nn.LeakyReLU(0.02, True),
        #     nn.Linear(72, n_acoustic),
        # )


        self.fc = nn.Sequential( # batchsize x 51 x 72
            nn.Linear(4*29, 100),
            nn.LeakyReLU(0.02, True),
            nn.Linear(100, 72),
            nn.LeakyReLU(0.02, True),
            nn.Linear(72, n_acoustic),
        )

    def forward(self, x):
        # batchsize x 51 x 16 x 29
        x = x.view(self.batch_size,self.num_frame,-1) # batchsize x 51 x (16x29)
        x = x.permute(0,2,1) # batchsize x (16x29) x 51
        x = self.encoder_conv(x) # batchsize x 29 x 51
        x = x.permute(0,2,1) # batchsize x 51 x 29
        x = self.fc(x) # batchsize x 51 x 72
        return x

class LipNet(nn.Module):
    def __init__(self, input_num=51200, output_num=72): # input: [B, T, 16, 29] output: [B, T, 72]
        super(LipNet, self).__init__()

        self.fc = nn.Sequential( # batchsize x 51 x 72
            nn.Linear(input_num, 512),
            nn.LeakyReLU(0.02, True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.02, True),
            nn.Linear(256, output_num),
        )

    def forward(self, x):
        # batchsize x 51 x 16 x 29
        x = self.fc(x) # batchsize x 51 x 72
        return x

class TimeDownsample(nn.Module):
    def __init__(self, n_acoustic, downsample=True):
        super().__init__()
        self.convs = nn.Sequential(
            Conv1dBlock(n_acoustic, n_acoustic, type_act='leakyrelu', stride=2 if downsample else 1),
            Conv1dBlock(n_acoustic, n_acoustic, type_act='leakyrelu', stride=1)
        )
    
    def forward(self, x):
        '''
        Args:
            x: acoustic inputs [B, T, C]
        Returns:
            encoder outputs [B, T // 2]
        '''
        # 1D-Convolution
        x = x.transpose(1, 2)  # [B, T, C] -> [B, C, T]
        x = self.convs(x).transpose(1, 2)  # [B, T//2, C]
        
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, layer, hidden_size, dropout):
        super().__init__()
        self.layer = layer
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.op = OPERATIONS_ENCODER[layer](hidden_size, dropout)

    def forward(self, x, **kwargs):
        return self.op(x, **kwargs)


class TransformerEncoder(nn.Module):
    def __init__(self, arch, hidden_size, last_ln=True, dropout=0.0):
        super().__init__()
        self.arch = arch
        self.num_layers = len(self.arch)
        self.hidden_size = hidden_size
        self.padding_idx = 0
        embed_dim = hidden_size
        self.dropout = dropout
        self.embed_scale = math.sqrt(hidden_size)
        self.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        self.embed_positions = SinusoidalPositionalEmbedding(
            embed_dim, self.padding_idx,
            init_size=self.max_source_positions + self.padding_idx + 1,
        )
        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(self.arch[i], self.hidden_size, self.dropout)
            for i in range(self.num_layers)
        ])
        self.last_ln = last_ln
        if last_ln:
            self.layer_norm = LayerNorm(embed_dim)

    def forward_embedding(self, inputs, src_tokens):
        positions = self.embed_positions(src_tokens)
        # x = self.prenet(x)
        x = inputs + positions
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x, inputs

    def forward(self, inputs, src_tokens):   # src_tokens
        """
        Args:
            inputs: [B, T, C]
        return: 
            {
                'encoder_out': [T x B x C]
                'encoder_padding_mask': [B x T]
                'encoder_embedding': [B x T x C]
                'attn_w': []
            }
        """
        x, _ = self.forward_embedding(inputs, src_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx).data  # [B, T]

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask=encoder_padding_mask)

        if self.last_ln:
            x = self.layer_norm(x)
            x = x * (1 - encoder_padding_mask.float()).transpose(0, 1)[..., None]
        return {
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
            'encoder_embedding': inputs,  # B x T x C
            'attn_w': []
        }


class Decoder(nn.Module):
    def __init__(self, arch, hidden_size=None, dropout=None):
        super().__init__()
        # self.arch = arch[4: 8]  # arch  = encoder op code
        self.arch = arch  # arch  = encoder op code
        self.num_layers = len(self.arch)
        if hidden_size is not None:
            embed_dim = self.hidden_size = hidden_size
        else:
            embed_dim = self.hidden_size = 64
        if dropout is not None:
            self.dropout = dropout
        else:
            self.dropout = 0.0
        self.max_source_positions = DEFAULT_MAX_TARGET_POSITIONS
        self.padding_idx = 0
        self.pos_embed_alpha = nn.Parameter(torch.Tensor([1]))
        self.embed_positions = SinusoidalPositionalEmbedding(
            embed_dim, self.padding_idx,
            init_size=self.max_source_positions + self.padding_idx + 1,
        )
        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(self.arch[i], self.hidden_size, self.dropout)
            for i in range(self.num_layers)
        ])
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, require_w=False):
        """
        Args: 
            x: [B, T, C]
            require_w: True if this module needs to return weight matrix
        return: [B, T, C]
        """
        padding_mask = x.abs().sum(-1).eq(0).data
        positions = self.pos_embed_alpha * self.embed_positions(x[..., 0])
        x = x + positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # encoder layers
        attn_w = []
        if require_w:
            for layer in self.layers:
                x, attn_w_i = layer(x, encoder_padding_mask=padding_mask, require_w=require_w)
                attn_w.append(attn_w_i)
        else:
            # modules/operations.py:122, modules.operations.EncSALayer
            for layer in self.layers:
                x = layer(x, encoder_padding_mask=padding_mask)  # remember to assign back to x

        x = self.layer_norm(x)
        x = x.transpose(0, 1)

        return (x, attn_w) if require_w else x


class Speech2Expression_Transformer(nn.Module):
    def __init__(self, n_acoustic, n_expression, downsample=True, dropout=0.0):
        super().__init__()

        # self.time_downsample = TimeDownsample(n_acoustic, downsample=downsample)
        # 变更： hidden_size = n_acoustic+n_expression
        self.encoder = TransformerEncoder(arch=[8] * 16, hidden_size=n_acoustic, dropout=dropout)
        self.decoder = Decoder(arch=[8] * 16, hidden_size=n_acoustic, dropout=dropout)
        self.blendshape_out = Linear(n_acoustic, n_expression, bias=True)
    
    def forward(self, input, input_length=None):
        '''
        Args:
            input: acoustic inputs [B, T, C]
            input_length: valid length of inputs [B]
        Returns:
            ret: [B, T//2, C] # cancel downsample
        '''
        # Time downsample [B, T, C] -> [B, T//2, C]
        # 需要对语音的部分进行下采样
        # input = self.time_downsample(input) # cancel downsample
       

        
        # [B, T] generate token from inputs to calculate positions
        src_tokens = input.abs().sum(-1)
        

        encoder_out = self.encoder(input, src_tokens)['encoder_out']
        src_nonpadding = (src_tokens > 0).float().permute(1, 0)[:, :, None]
        encoder_out = encoder_out * src_nonpadding  # [T, B, C]

        decoder_out = self.decoder(encoder_out.transpose(0, 1))

        ret = self.blendshape_out(decoder_out)

        return ret

class mySpeech2Expression_Transformer(nn.Module):
    def __init__(self, n_acoustic, n_expression, downsample=True, dropout=0.0):
        super().__init__()

        # self.time_downsample = TimeDownsample(n_acoustic, downsample=downsample)
        # 变更： hidden_size = n_acoustic+n_expression
        self.encoder = TransformerEncoder(arch=[8] * 16, hidden_size=n_acoustic, dropout=dropout)
        self.decoder = Decoder(arch=[8] * 16, hidden_size=n_acoustic, dropout=dropout)
        self.blendshape_out = Linear(n_acoustic, n_expression, bias=True)
    
    def forward(self, input, input_length=None):
        '''
        Args:
            input: acoustic inputs [B, T, C]
            input_length: valid length of inputs [B]
        Returns:
            ret: [B, T//2, C] # cancel downsample
        '''
        # Time downsample [B, T, C] -> [B, T//2, C]
        # 需要对语音的部分进行下采样
        # input = self.time_downsample(input) # cancel downsample
       

        
        # [B, T] generate token from inputs to calculate positions
        src_tokens = input.abs().sum(-1)
        

        encoder_out = self.encoder(input, src_tokens)['encoder_out']
        src_nonpadding = (src_tokens > 0).float().permute(1, 0)[:, :, None]
        encoder_out = encoder_out * src_nonpadding  # [T, B, C]

        decoder_out = self.decoder(encoder_out.transpose(0, 1))

        ret = self.blendshape_out(decoder_out)

        return ret

class myLoss(nn.Module):
    def __init__(self, loss_type='mse', delta=None, smooth_weight=0.05):
        super().__init__()
        self.loss_type = loss_type
        self.delta = delta
        self.smooth_weight = smooth_weight
    
    def weights_nonzero(self, target):
        # target: [B, T, C]
        # Assign weight 1.0 to all labels except for padding (id=0).
        dim = target.size(-1)
        return target.abs().sum(-1, keepdim=True).ne(0).float().repeat(1, 1, dim)
        # return target
    
    def l1_loss(self, predict, target, start, length):
        # predict: [B, T ,C]
        # target: [B, T, C] with zero padding
        l1_loss_miss = F.l1_loss(predict[:,start:start+length,:], target[:,start:start+length,:], reduction='mean')
        l1_loss_before = F.l1_loss(predict[:,:start,:], target[:,:start,:], reduction='mean')
        l1_loss_after = F.l1_loss(predict[:,start+length:,:], target[:,start+length:,:], reduction='mean')
        l1_loss = 0.2 * l1_loss_before + 0.6 * l1_loss_miss + 0.2 * l1_loss_after
        # weights = self.weights_nonzero(target)
        # l1_loss = (l1_loss * weights).sum() / weights.sum()
        return l1_loss

    def mse_loss(self, predict, target, start, length):
        # predict: [B, T ,C]
        # target: [B, T, C] with zero padding
        # mse_loss = F.mse_loss(predict[:,start:start+length,:], target[:,start:start+length,:], reduction='mean')
        mse_loss_miss = F.mse_loss(predict[:,start:start+length,:], target[:,start:start+length,:], reduction='mean')
        mse_loss_before = F.mse_loss(predict[:,:start,:], target[:,:start,:], reduction='mean')
        mse_loss_after = F.mse_loss(predict[:,start+length:,:], target[:,start+length:,:], reduction='mean')
        mse_loss = 0.2 * mse_loss_before + 0.6 * mse_loss_miss + 0.2 * mse_loss_after
        # weights = self.weights_nonzero(target)
        # mse_loss = (mse_loss * weights).sum() / weights.sum()
        return mse_loss
    
    def huber_loss(self, predict, target, delta, start, length):
        # predict: [B, T ,C]
        # target: [B, T, C] with zero padding
        # loss = F.smooth_l1_loss(predict[:,start:start+length,:], target[:,start:start+length,:], reduction='mean', beta=delta)
        smooth_l1_loss_miss = F.smooth_l1_loss(predict[:,start:start+length,:], target[:,start:start+length,:], reduction='mean')
        smooth_l1_loss_before = F.smooth_l1_loss(predict[:,:start,:], target[:,:start,:], reduction='mean')
        smooth_l1_loss_after = F.smooth_l1_loss(predict[:,start+length:,:], target[:,start+length:,:], reduction='mean')
        smooth_l1_loss = 0.2 * smooth_l1_loss_before + 0.6 * smooth_l1_loss_miss + 0.2 * smooth_l1_loss_after
        # loss = F.huber_loss(predict, target, reduction='mean', beta=delta)
        # weights = self.weights_nonzero(target)
        # loss = (loss * weights).sum() / weights.sum()
        return smooth_l1_loss
    
    def smooth_loss(self, predict, target, start, length):
        # predict: [B, T ,C]
        # target: [B, T, C] with zero padding
        predict_left_offset1 = predict[:, 0:-1, :]
        predict_right_offset1 = predict[:, 1:, :]
        loss = F.cosine_similarity(predict_left_offset1, predict_right_offset1, dim=-1, eps=1e-6)
        # print(self.weights_nonzero(target[:, 0:-1, :]))
        weights = self.weights_nonzero(target[:, 0:-1, :])[:,:,0]
        loss = (loss * weights).sum() / weights.sum()
        
        return loss

    
    def forward(self, predict, target, start, length):
        # print('predict:',predict.shape)
        assert predict.shape == target.shape

        # Smooth loss
        smooth_loss = self.smooth_loss(predict, target, start, length)

        # Target loss
        if self.loss_type == 'l1':
            target_loss = self.l1_loss(predict, target, start, length)
        elif self.loss_type == 'mse':
            target_loss = self.mse_loss(predict, target, start, length)
        elif self.loss_type == 'huber':
            target_loss = self.huber_loss(predict, target, self.delta, start, length)
        
        total_loss = target_loss - self.smooth_weight * smooth_loss

        return total_loss, target_loss, smooth_loss


# class Loss(nn.Module):
#     def __init__(self, loss_type='mse', delta=None, smooth_weight=0.05):
#         super().__init__()
#         self.loss_type = loss_type
#         self.delta = delta
#         self.smooth_weight = smooth_weight

#     def weights_nonzero(self, target):
#         # target: [B, T, C]
#         # Assign weight 1.0 to all labels except for padding (id=0).
#         dim = target.size(-1)
#         return target.abs().sum(-1, keepdim=True).ne(0).float().repeat(1, 1, dim)
#         # return target
    
#     def l1_loss(self, predict, target):
#         # predict: [B, T ,C]
#         # target: [B, T, C] with zero padding
#         l1_loss = F.l1_loss(predict, target, reduction='mean')
#         # weights = self.weights_nonzero(target)
#         # l1_loss = (l1_loss * weights).sum() / weights.sum()
#         return l1_loss

#     def mse_loss(self, predict, target):
#         # predict: [B, T ,C]
#         # target: [B, T, C] with zero padding
#         mse_loss = F.mse_loss(predict, target, reduction='mean')
#         # weights = self.weights_nonzero(target)
#         # mse_loss = (mse_loss * weights).sum() / weights.sum()
#         return mse_loss
    
#     def huber_loss(self, predict, target, delta):
#         # predict: [B, T ,C]
#         # target: [B, T, C] with zero padding
#         loss = F.smooth_l1_loss(predict, target, reduction='mean', beta=delta)
#         # loss = F.huber_loss(predict, target, reduction='mean', beta=delta)
#         # weights = self.weights_nonzero(target)
#         # loss = (loss * weights).sum() / weights.sum()
#         return loss
    
#     def smooth_loss(self, predict, target):
#         # predict: [B, T ,C]
#         # target: [B, T, C] with zero padding
#         predict_left_offset1 = predict[:, 0:-1, :]
#         predict_right_offset1 = predict[:, 1:, :]
#         loss = F.cosine_similarity(predict_left_offset1, predict_right_offset1, dim=-1, eps=1e-6)
#         # print(self.weights_nonzero(target[:, 0:-1, :]))
#         weights = self.weights_nonzero(target[:, 0:-1, :])[:,:,0]
#         loss = (loss * weights).sum() / weights.sum()
        
#         return loss

    
#     def forward(self, predict, target):
#         # print('predict:',predict.shape)
#         assert predict.shape == target.shape

#         # Smooth loss
#         smooth_loss = self.smooth_loss(predict, target)

#         # Target loss
#         if self.loss_type == 'l1':
#             target_loss = self.l1_loss(predict, target)
#         elif self.loss_type == 'mse':
#             target_loss = self.mse_loss(predict, target)
#         elif self.loss_type == 'huber':
#             target_loss = self.huber_loss(predict, target, self.delta)
        
#         total_loss = target_loss - self.smooth_weight * smooth_loss

#         return total_loss, target_loss, smooth_loss

class Loss(nn.Module):
    def __init__(self, loss_type='mse', delta=None, smooth_weight=0.01):
        super().__init__()
        self.loss_type = loss_type
        self.delta = delta
        self.smooth_weight = smooth_weight

    def weights_nonzero(self, target):
        # target: [B, T, C]
        # Assign weight 1.0 to all labels except for padding (id=0).
        dim = target.size(-1)
        return target.abs().sum(-1, keepdim=True).ne(0).float().repeat(1, 1, dim)
        # return target
    
    def l1_loss(self, predict, target):
        # predict: [B, T ,C]
        # target: [B, T, C] with zero padding
        l1_loss = F.l1_loss(predict, target, reduction='none')
        weights = self.weights_nonzero(target)
        l1_loss = (l1_loss * weights).sum() / weights.sum()
        return l1_loss

    def mse_loss(self, predict, target):
        # predict: [B, T ,C]
        # target: [B, T, C] with zero padding
        mse_loss = F.mse_loss(predict, target, reduction='none')
        weights = self.weights_nonzero(target)
        mse_loss = (mse_loss * weights).sum() / weights.sum()
        return mse_loss
    
    def huber_loss(self, predict, target, delta):
        # predict: [B, T ,C]
        # target: [B, T, C] with zero padding
        loss = F.smooth_l1_loss(predict, target, reduction='none', beta=delta)
        weights = self.weights_nonzero(target)
        loss = (loss * weights).sum() / weights.sum()
        return loss
    
    def smooth_loss(self, predict, target):
        # predict: [B, T ,C]
        # target: [B, T, C] with zero padding
        predict_left_offset1 = predict[:, 0:-1, :]
        predict_right_offset1 = predict[:, 1:, :]

        loss = F.cosine_similarity(predict_left_offset1, predict_right_offset1, dim=-1, eps=1e-6)
        weights = self.weights_nonzero(target[:, 0:-1, :])[:,:,0]
        loss = (loss * weights).sum() / weights.sum()
        
        return loss

    
    def forward(self, predict, target):
        assert predict.shape == target.shape

        # Smooth loss
        smooth_loss = self.smooth_loss(predict, target)

        # Target loss
        if self.loss_type == 'l1':
            target_loss = self.l1_loss(predict, target)
        elif self.loss_type == 'mse':
            target_loss = self.mse_loss(predict, target)
        elif self.loss_type == 'huber':
            target_loss = self.huber_loss(predict, target, self.delta)
        
        total_loss = target_loss - self.smooth_weight * smooth_loss

        return total_loss, target_loss, smooth_loss