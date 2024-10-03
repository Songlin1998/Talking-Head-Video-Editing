import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline


def smooth_xy(lx, ly):

    x = np.array(lx)
    y = np.array(ly)
    x_smooth = np.linspace(x.min(), x.max(), 300)
    y_smooth = make_interp_spline(x, y)(x_smooth)
    return [x_smooth, y_smooth]


if __name__ == '__main__':


    x_raw_0 = [0, 1, 2, 3, 4, 5]
    y_raw_0 = [3538,3801,3407,2490,2621,2647]
    xy_s_0 = smooth_xy(x_raw_0, y_raw_0)
    
    x_raw_1 = [1, 2, 3, 4]
    y_raw_1 = [3866,3080,3932,4050]
    xy_s_1 = smooth_xy(x_raw_1, y_raw_1)

    x_raw_2 = [1, 2, 3, 4]
    y_raw_2 = [3591,3198,3486,3538]
    xy_s_2 = smooth_xy(x_raw_2, y_raw_2)

    x_raw_3 = [1, 2, 3, 4]
    y_raw_3 = [3538,3407,3473,3460]
    xy_s_3 = smooth_xy(x_raw_3, y_raw_3)

    x_raw_4 = [1, 2, 3, 4]
    y_raw_4 = [3473,3538,3554,3565]
    xy_s_4 = smooth_xy(x_raw_4, y_raw_4)

    x_raw_5 = [1, 2, 3, 4]
    y_raw_5 = [3709,3211,3591,3420]
    xy_s_5 = smooth_xy(x_raw_5, y_raw_5)

    plt.scatter(x_raw_0, y_raw_0, color='b')
    plt.plot(xy_s_0[0], xy_s_0[1], color='b', linewidth=2, label='Origin')

    plt.scatter(x_raw_1, y_raw_1, color='red')
    plt.plot(xy_s_1[0], xy_s_1[1], color='red', linewidth=2, label='Test')

    plt.scatter(x_raw_2, y_raw_2, color='sandybrown')
    plt.plot(xy_s_2[0], xy_s_2[1], color='sandybrown', linewidth=1, label='AD-NeRF')

    plt.scatter(x_raw_3, y_raw_3, color='steelblue')
    plt.plot(xy_s_3[0], xy_s_3[1], color='steelblue', linewidth=1, label='DFRF')

    plt.scatter(x_raw_4, y_raw_4, color='violet')
    plt.plot(xy_s_4[0], xy_s_4[1], color='violet', linewidth=1, label='Wav2Lip')

    plt.scatter(x_raw_5, y_raw_5, color='black')
    plt.plot(xy_s_5[0], xy_s_5[1], color='black', linewidth=2, label='Ours')
    
    plt.yticks(rotation=60)
    plt.legend(loc='best',fontsize='large',frameon=False)
    plt.savefig("filename.png")
    plt.show()
