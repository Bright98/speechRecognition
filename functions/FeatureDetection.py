import numpy as np
from python_speech_features import mfcc
import math


def FeatureDetection(signal, fs, win):
    energy = []
    zero_cross = []
    feature = []

    for j in range(len(win[0])):
        sum = 0
        z = []
        for row in win:
            sum += pow(row[j], 2)
            z.append(row[j])
        energy.append(sum)
        zero_crossing = np.where(np.diff(np.signbit(z)))[0]
        zero_cross.append(len(zero_crossing))

    # energy
    energy = np.array(energy)
    energy_min = min(energy)
    energy_max = max(energy)
    energy_mean = np.mean(energy)
    energy_var = np.var(energy)

    # zero crossing
    zero_cross = np.array(zero_cross)
    zero_cross_min = min(zero_cross)
    zero_cross_max = max(zero_cross)
    zero_cross_mean = np.mean(zero_cross)
    zero_cross_var = np.var(zero_cross)

    feature.extend(
        (
            energy_min,
            energy_max,
            energy_mean,
            energy_var,
            zero_cross_min,
            zero_cross_max,
            zero_cross_mean,
            zero_cross_var,
        )
    )

    # MFCC
    mfcc_feat = mfcc(
        signal,
        samplerate=fs,
        numcep=13,
        nfilt=math.floor(3 * math.log(fs)),
        winfunc=np.hamming,
    )
    for col in range(13):
        mfcc_min = min(mfcc_feat[:, col])
        mfcc_max = max(mfcc_feat[:, col])
        mfcc_mean = np.mean(mfcc_feat[:, col])
        mfcc_var = np.var(mfcc_feat[:, col])
        feature.extend((mfcc_min, mfcc_max, mfcc_mean, mfcc_var))

    mfcc_diff = np.diff(mfcc_feat, axis=1)
    for col in range(12):
        mfcc_diff_min = min(mfcc_diff[:, col])
        mfcc_diff_max = max(mfcc_diff[:, col])
        mfcc_diff_mean = np.mean(mfcc_diff[:, col])
        mfcc_diff_var = np.var(mfcc_diff[:, col])
        feature.extend((mfcc_diff_min, mfcc_diff_max, mfcc_diff_mean, mfcc_diff_var))

    mfcc_diff2 = np.diff(mfcc_diff, axis=1)
    for col in range(11):
        mfcc_diff_min2 = min(mfcc_diff2[:, col])
        mfcc_diff_max2 = max(mfcc_diff2[:, col])
        mfcc_diff_mean2 = np.mean(mfcc_diff2[:, col])
        mfcc_diff_var2 = np.var(mfcc_diff2[:, col])
        feature.extend(
            (mfcc_diff_min2, mfcc_diff_max2, mfcc_diff_mean2, mfcc_diff_var2)
        )

    return feature
