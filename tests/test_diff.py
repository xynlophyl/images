from diff import SSIM
from utils import show_image
import time
import numpy as np

def compare_images(
    source: np.ndarray, 
    target: np.ndarray,
    **kwargs
):
    
    ssim = SSIM(
        **kwargs
    )

   # generate ssim map
    print('generating ssim map')
    start = time.time()
    score, ssim_map = ssim.generate_ssim_map(source, target)
    end = time.time()
    print(f"mean ssim score: {score:.2f}")
    print(f"time elapsed: {str(end-start) + 's' if end-start>= 1 else str((end-start)*1000) + 'ms'}")

    show_image(ssim_map)
