import torch
import torch.nn.functional as F
import numpy as np
from scipy.signal import convolve2d

def fspecial_gaussian(shape, sigma):
    """
    2D gaussian mask - should give the same result as MATLAB's fspecial('gaussian',...)
    """
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def vifp_mscale(ref, dist):
    ref = ref.unsqueeze(0)
    dist = dist.unsqueeze(0)
    sigma_nsq = 2
    num = 0
    den = 0
    for scale in range(1, 5):
        N = 2**(4-scale+1)+1
        win = torch.tensor(fspecial_gaussian((N, N), N/5), dtype=ref.dtype).unsqueeze(0).unsqueeze(0).to(ref.device)

        if scale > 1:
            ref = F.conv2d(ref, win, padding=0)  # 添加批次和通道维度
            dist = F.conv2d(dist, win, padding=0)
            ref = ref[::2, ::2]
            dist = dist[::2, ::2]

        mu1 = F.conv2d(ref, win, padding=0)
        mu2 = F.conv2d(dist, win, padding=0)
        mu1_sq = mu1*mu1
        mu2_sq = mu2*mu2
        mu1_mu2 = mu1*mu2
        sigma1_sq = F.conv2d(ref * ref, win, padding=0) - mu1_sq
        sigma2_sq =  F.conv2d(dist * dist, win, padding=0) - mu2_sq
        sigma12 =  F.conv2d(dist * ref, win, padding=0) - mu1_mu2
        sigma1_sq[sigma1_sq<0] = 0
        sigma2_sq[sigma2_sq<0] = 0

        g = sigma12 / (sigma1_sq + 1e-10)
        sv_sq = sigma2_sq - g*sigma12

        g[sigma1_sq<1e-10] = 0
        sv_sq[sigma1_sq<1e-10] = sigma2_sq[sigma1_sq<1e-10]
        sigma1_sq[sigma1_sq<1e-10] = 0

        g[sigma2_sq<1e-10] = 0
        sv_sq[sigma2_sq<1e-10] = 0

        sv_sq[g<0] = sigma2_sq[g<0]
        g[g<0] = 0
        sv_sq[sv_sq<=1e-10] = 1e-10

        num += torch.sum(torch.log10(1+g**2 * sigma1_sq/(sv_sq+sigma_nsq)))
        den += torch.sum(torch.log10(1+sigma1_sq/sigma_nsq))
    vifp = num/den
    return vifp