from processing import ImageProcessor
from utils import show_image
import time
import numpy as np
import cv2

def process_image(
    source_image: np.ndarray,
    operation: str,
    **kwargs
):

    image_processor = ImageProcessor()

    operations = {
        "blur": image_processor.blur,
        "sharpen": image_processor.sharpen,
        "gaussian": image_processor.gaussian_blur,
        "ridge": image_processor.detect_edges
    }
    
    oper = operations[operation]

    start = time.time()

    processed_image = oper(source_image, **kwargs)
    print(f'time taken: {time.time() - start}')
    show_image(cv2.hconcat([source_image, processed_image]))