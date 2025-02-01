import numpy as np
from .convolution import convolve, naive_convolve, spatial_convolve
from typing import List

class ImageProcessor():
    def __init__(self) -> None:
        self.convolution_dict = {
            "convolve": convolve,
            "naive-convolve": naive_convolve,
            "spatial-convolve": spatial_convolve
        }

    def _expand_to_rgb(self, arr: np.ndarray):
        return np.repeat(np.expand_dims(arr, axis = -1), 3, axis = -1)

    def _apply_mask(self, image: np.ndarray, mask_s: List[np.ndarray], stride: int, method: str) -> np.ndarray:
        """
        convolve image with processing mask
        """
        
        convolve_fn = self.convolution_dict[method]
        return convolve_fn(image, mask_s, stride)
    
    def blur(self, image: np.ndarray, mask_size: int = 7, stride: int = 1, method: str = "convolve") -> np.ndarray:
        if method == "spatial-convolve":
            num_channels = image.shape[2]
            mask_rgb_y = np.ones((mask_size,num_channels))
            mask_rgb_x = mask_rgb_y/(mask_size**2)

            return self._apply_mask(image, [mask_rgb_y, mask_rgb_x], stride = stride, method = "spatial-convolve")
        else:
            blur_mask = np.full((mask_size, mask_size, image.shape[-1]), 1/mask_size**2)
            return self._apply_mask(image, blur_mask, stride = stride, method = method)

    def sharpen(self, image: np.ndarray, mask = [[0,-1,0], [-1,5,-1], [0,-1,0]], stride : int = 1, method: str = "convolve"):
        # expand sharpen mask across each rgb channel 
        mask = np.array(mask)
        mask_rgb = self._expand_to_rgb(mask)

        return self._apply_mask(image, mask_rgb, stride = stride, method = method)

    def _gaussian1d(self, x: int, std_dev: float) -> float:
        return np.float64((1/(2*np.pi*std_dev**2)**0.5)*np.exp(-x**2/(2*std_dev**2)))

    def _gaussian2d(self, x: int, y: int, std_dev: float) -> float:
        return np.float64((1/(2*np.pi*std_dev**2))*np.exp(-(x**2+y**2)/(2*std_dev**2)))
    
    def gaussian_blur(self, image: np.ndarray, std_dev: float = 1.0, mask_size: int = 3, stride: int = 1, method: str = "convolve") -> np.ndarray:
        if method == "spatial-convolve":
            center = mask_size//2
            mask = np.array([self._gaussian1d(y-center, std_dev) for y in range(mask_size)])
            
            # expand to rgb space
            mask_rgb = self._expand_to_rgb(mask)

            return self._apply_mask(image, [mask_rgb, mask_rgb], stride = stride, method = "spatial-convolve")
        else:
            center_x = center_y = mask_size//2
            gaussian_mask = np.array([[self._gaussian2d(x-center_x, y-center_y, std_dev) for x in range(mask_size)] for y in range(mask_size)])
            mask_rgb = self._expand_to_rgb(gaussian_mask)
            
            return self._apply_mask(image, mask_rgb, stride = stride, method = method)
    
    def detect_edges(self, image: np.ndarray, scale = 1, method = "convolve") -> np.ndarray:
        if method == "spatial-convolve":
            raise ValueError("Method: spatial-convolve is not available for edge detection")
        
        mask = np.array([
            [1, 0.00, -1],
            [2, 0.00, -2],
            [1, 0.00, -1],
        ])*scale

        vertical_edges = self.__apply_mask(image, mask, method = method)
        horizontal_edges = self._apply_mask(image, mask.T, method = method)

        return (vertical_edges**2 + horizontal_edges**2)**0.5