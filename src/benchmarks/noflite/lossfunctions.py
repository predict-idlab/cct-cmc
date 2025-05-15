# Copied from: https://github.com/toonvds/NOFLITE

import torch
import math


def calcMMD(X, t):
    p = 0.5
    maskControl = torch.as_tensor([treat == 0 for treat in t])
    maskTreat = torch.as_tensor([treat == 1 for treat in t])

    LatentControl = X[maskControl]
    LatentTreat = X[maskTreat]

    meanControl = torch.mean(LatentControl, 0)
    meanTreat = torch.mean(LatentTreat, 0)

    if math.isnan(meanTreat[0]) or math.isnan(meanControl[0]):
        return 0

    mmd = torch.sum(torch.square(2.0*p*meanTreat - 2.0*(1-p)*meanControl))
    return mmd
