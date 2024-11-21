import numpy as np
from skimage.metrics import structural_similarity as ssim

def compute_psnr(target, prediction):
    target = target.cpu().numpy()
    prediction = prediction.cpu().numpy()
    mse = np.mean((target - prediction) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0  # Assuming normalized images
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def compute_ssim(target, prediction):
    target = target.cpu().numpy().transpose(0, 2, 3, 1)  # Convert to HWC
    prediction = prediction.cpu().numpy().transpose(0, 2, 3, 1)
    
    # Check image dimensions and set win_size accordingly
    win_size = 3 if target.shape[1] <= 7 else 7
    
    # Calculate SSIM for each image in the batch
    return np.mean([
        ssim(target[i], prediction[i], multichannel=True, win_size=win_size, data_range=1.0, channel_axis=-1)
        for i in range(target.shape[0])
    ])

def compute_mse(target, prediction):
    target = target.cpu().numpy()
    prediction = prediction.cpu().numpy()
    return np.mean((target - prediction) ** 2)