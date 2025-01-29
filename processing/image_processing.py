import numpy as np
from .convolution import convolve, fast_convolve

class ImageProcessor():
    def __init__(self) -> None:
        self.convolution_dict = {
            'convolve': convolve,
            'fast-convolve': fast_convolve,
        }

    def _apply_mask(self, image: np.ndarray, mask: np.ndarray, stride: int, mode: str) -> np.ndarray:
        """
        convolve image with processing mask
        """
        
        convolve_fn = self.convolution_dict[mode]
        return convolve_fn(image, mask, stride)
    
    def blur(self, image: np.ndarray, mask_size: int = 7, stride: int = 1, mode: str = "convolve") -> np.ndarray:
        blur_mask = np.full((mask_size, mask_size, image.shape[-1]), 1/mask_size**2)

        return self._apply_mask(image, blur_mask, stride = stride, mode = mode)

    def sharpen(self, image: np.ndarray, mask = [[0,-1,0], [-1,5,-1], [0,-1,0]], stride : int = 1, mode: str = "convolve"):
        # expand sharpen mask across each rgb channel 
        mask = np.array(mask)
        mask_rgb = np.repeat(np.expand_dims(mask, axis = -1), 3, axis = -1)

        return self._apply_mask(image, mask_rgb, stride = stride, mode = mode)

    def _gaussian(self, x: int, y: int, std_dev: float) -> np.ndarray:
        return (1/(2*np.pi*std_dev**2))*np.exp(-(x**2+y**2)/(2*std_dev**2))

    def gaussian_blur(self, image: np.ndarray, std_dev: float = 1.0, mask_size: int = 3, stride: int = 1, mode: str = "convolve") -> np.ndarray:
        center_x = center_y = mask_size//2
        gaussian_mask = np.array([[self._gaussian(x-center_x, y-center_y, std_dev) for x in range(mask_size)] for y in range(mask_size)])
        mask_rgb = np.repeat(np.expand_dims(gaussian_mask, axis = -1), 3, axis = -1)

        return self._apply_mask(image, mask_rgb, stride = stride, mode = mode)
    
    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        mask = np.array([
            [0.25, 0.00, -0.25],
            [0.50, 0.00, -0.50],
            [0.25, 0.00, -0.25],
        ])

        return self._apply_mask(image, mask) + self._apply_mask(image, mask.T)
