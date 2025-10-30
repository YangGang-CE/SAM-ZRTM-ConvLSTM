import numpy as np
from skimage.metrics import structural_similarity as ssim_metric


def mse(ground_truth, predictions):
    mse_value = np.square(ground_truth - predictions).sum()
    return mse_value / 10000


def ssim(ground_truth, predictions):
    return ssim_metric(ground_truth, predictions)
