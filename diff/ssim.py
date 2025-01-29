import numpy as np
from typing import Tuple

class SSIM():

    def __init__(self, window_size: int = 5, alpha: float = 1, beta: float = 1, gamma: float = 1) -> None:
        self.N = window_size
        self.alpha = alpha
        self.beta = beta 
        self.gamma = gamma
    
    def get_covariance(
            self, 
            source_kernel: np.ndarray, 
            target_kernel: np.ndarray, 
            source_mean: float, 
            target_mean: float
    ) -> float:
        N = source_kernel.shape[0]*source_kernel.shape[1]*source_kernel.shape[2]
        return np.sum((source_kernel-source_mean)*(target_kernel-target_mean), axis = (0,1))/(N-1)
    
    def get_bits_per_pixel(self, arr: np.ndarray) -> int | None:
        dtype = str(arr.dtype)
        if not dtype[-1].isnumeric():
            raise Exception("dtype incorrect")

        return int(dtype[-1])
    
    def add_padding(self, image: np.ndarray):
        r = self.N // 2
        return np.pad(image, pad_width = ((r, r), (r, r), (0,0)))

    def get_image_ssim_constants(self, image: np.ndarray) -> Tuple[float, float]:
        # dynamic range: 2**(# bits per pixel) - 1
        L = 2**self.get_bits_per_pixel(image)-1
        
        # constants: k
        k1 = 0.01
        k2 = 0.03

        # constants: c
        c1 = (k1*L)**2
        c2 = (k2*L)**2

        return c1, c2

    def measure_ssim_index(self, source_kernel: np.ndarray, target_kernel: np.ndarray, c1: float, c2: float) -> float:
        """
        calculates ssim index of the current kernel
        """
        # sample mean
        mu_x = np.mean(source_kernel, axis = (0,1))
        mu_y = np.mean(target_kernel, axis = (0,1))   

        # variance
        var_x = np.var(source_kernel, axis = (0,1))
        var_y = np.var(target_kernel, axis = (0,1))

        # covariance
        cov_xy = self.get_covariance(source_kernel, target_kernel, mu_x, mu_y)

        # group formula parameters
        source_params = (mu_x, var_x)
        target_params = (mu_y, var_y)
        consts = (cov_xy, c1, c2)

        # calculate each individual cov_xyomponent (luminance, contrast, structure)
        luminance = self.calculate_luminance_similarity(source_params, target_params, consts)
        contrast = self.calculate_contrast_similarity(source_params, target_params, consts)
        structure = self.calculate_structure_similarity(source_params, target_params, consts)

        ssim_rgb = (luminance**self.alpha * contrast**self.beta * structure**self.gamma)
        rgb2gray = np.array([0.2989, 0.5810, 0.1140])

        return rgb2gray @ ssim_rgb
    
    def calculate_luminance_similarity(self, source_params: Tuple[float, float], target_params: Tuple[float, float], consts: Tuple[float, float, float]) -> float:
        mu_x, _ = source_params
        mu_y, _ = target_params
        _, c1, _ = consts

        return (2*mu_x*mu_y+c1) / (mu_x**2+mu_y**2+c1)

    def calculate_contrast_similarity(self, source_params: Tuple[float, float], target_params: Tuple[float, float], consts: Tuple[float, float, float]) -> float:
        _, var_x = source_params
        _, var_y = target_params
        _, _, c2 = consts

        return (2*(var_x**0.5)*(var_y**0.5)+c2) / (var_x+var_y+c2)
    
    def calculate_structure_similarity(self, source_params: Tuple[float, float], target_params: Tuple[float, float], consts: Tuple[float, float, float]) -> float:
        _, var_x = source_params
        _, var_y = target_params
        cov_xy, _, c2 = consts

        return (cov_xy + c2/2) / ((var_x**0.5)*(var_y**0.5) + c2/2)
    
    def calculate_global_ssim(self, source: np.ndarray, target: np.ndarray) -> float:
        """
        assumes that both images are of the same resolution
        """

        c1, c2 = self.get_image_ssim_constants(source)
        return self.measure_ssim_index(source, target, c1, c2)
    
    def generate_ssim_map(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        # get image dimensions
        height, width, channels = source.shape

        # add padding
        padded_source, padded_target = self.add_padding(source), self.add_padding(target)

        # get ssim constants
        c1, c2 = self.get_image_ssim_constants(source)
        
        ssim_map = np.zeros_like(source)
        for y in range(height):
            for x in range(width):
                cropped_source = padded_source[y:y+self.N, x:x+self.N, :]
                cropped_target = padded_target[y:y+self.N, x:x+self.N, :]
                ssim_val = self.measure_ssim_index(cropped_source, cropped_target, c1, c2)
                ssim_map[y,x,2] = ssim_val*255
        
        print('mean ssim', np.mean(ssim_map))
        return ssim_map
