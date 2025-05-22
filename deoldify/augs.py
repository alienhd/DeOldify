import random

from fastai.vision.image import TfmPixel

# Contributed by Rani Horev. Thank you!
def _noisify(
    x, pct_pixels_min: float = 0.001, pct_pixels_max: float = 0.4, noise_range: int = 30
):
    if noise_range > 255 or noise_range < 0:
        raise Exception("noise_range must be between 0 and 255, inclusively.")

    h, w = x.shape[1:]
    img_size = h * w
    mult = 10000.0
    pct_pixels = (
        random.randrange(int(pct_pixels_min * mult), int(pct_pixels_max * mult)) / mult
    )
    noise_count = int(img_size * pct_pixels)

    for ii in range(noise_count):
        yy = random.randrange(h)
        xx = random.randrange(w)
        noise = random.randrange(-noise_range, noise_range) / 255.0
        x[:, yy, xx].add_(noise)

    return x


noisify = TfmPixel(_noisify)

# Placeholder for denorm_lab_mean_std_tensor
# Actual implementation details might vary based on model's output normalization.
# Assumes input tensor is (3, H, W) with L, A, B channels.
# L expected in [0,1] or [-1,1] from model, A,B in [-1,1].
# Output L in [0,100], A,B in approx [-128,127] (typical range for PIL conversion)
import torch # Ensure torch is imported in augs.py

def denorm_lab_mean_std_tensor(norm_lab_tensor: torch.Tensor) -> torch.Tensor:
    # Assuming norm_lab_tensor has L channel scaled 0-1 by ToTensor and model preserves/returns it similarly,
    # and AB channels are output by model in range approx -1 to 1.
    L = norm_lab_tensor[0:1, :, :] 
    AB = norm_lab_tensor[1:3, :, :]
    
    # Denormalize L: if input L is [0,1], scale to [0,100] for LAB standard
    # (This was already done in VideoColorizerFilter.merge_lab_tensors, so L here should be 0-100)
    # If L is already 0-100 (as merged by VideoColorizerFilter), no change needed for L here.
    # If L was, for instance, normalized to [-1, 1] by the model, it would need (L+1)/2 * 100.
    # Let's assume L is already scaled 0-100 as per merge_lab_tensors.
    
    # Denormalize AB: if model outputs AB in [-1,1], scale to approx [-128, 127]
    # A common scaling factor for AB channels (if normalized to [-1,1]) is 128.
    AB_denorm = AB * 128.0
    
    # Combine L (already assumed 0-100) and denormalized AB
    denormed_tensor = torch.cat((L, AB_denorm), dim=0)
    return denormed_tensor
