import numpy as np
from typing import List, Tuple

def _calculate_weighted_average(x, y, axis: int | Tuple[int, int]):
    return np.sum(x*y, axis = axis)

def naive_convolve(mat: np.ndarray, kernel: np.ndarray, stride: int = 1):
    """
    naive implementation (4 nested loops) of computing convolutions between a (m,n) matrix and a (k,l) kernel
        first, add padding of lengths (k//2, l//2) to the horizontal and vertical sides of the matrix
        next, for each cell in the original image, calculate the respective convolution:
    """
    # get kernel dimensions
    kernel_h, kernel_w, _ = kernel.shape

    # get matrix dimensions
    height, width, _ = mat.shape

    # add padding to height and width of image
    radius_y, radius_x = kernel_h//2, kernel_w//2
    padded_mat = np.pad(mat, pad_width = ((radius_y, radius_y), (radius_x, radius_x), (0,0)))

    output = np.zeros_like(mat, dtype = np.float64)
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            for dy in range(kernel_h):
                for dx in range(kernel_w):
                    output[y,x,:] += padded_mat[y+dy, x+dx,:]*kernel[dy,dx,:]
    
    return output.astype(mat.dtype)

def convolve(mat: np.ndarray, kernel: np.ndarray, stride: int = 1) -> np.array:
    """
    convolution implementation utilizing numpy for faster matrix calculation
    """
    # get kernel dimensions
    kernel_h, kernel_w, _ = kernel.shape

    # get matrix dimensions
    height, width, _ = mat.shape

    # add padding to height and width of image
    radius_y, radius_x = kernel_h//2, kernel_w//2
    padded_mat = np.pad(mat, pad_width = ((radius_y, radius_y), (radius_x, radius_x), (0,0)))

    output = np.zeros_like(mat, dtype = np.float64)
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            crop = padded_mat[y:y+kernel_h, x:x+kernel_w,:]
            output[y,x,:] = _calculate_weighted_average(crop, kernel[::-1], axis = (0,1))

    return output.astype(mat.dtype)

def spatial_convolve(mat, kernels_xy: List[np.ndarray], stride: int):
    """
    convolution implemenation utlizing spatial separable kernel, allowing for a more efficient two pass solution 
    """
    kernel_y, kernel_x = kernels_xy

    # get matrix dimensions
    height, width, _ = mat.shape

    # add padding to matrix
    y_len, x_len =  len(kernel_y), len(kernel_x)
    r_y, r_x = y_len//2, x_len//2
    padded_mat = np.pad(mat, pad_width = ((r_y, r_y),(r_x, r_x), (0,0)))

    # first pass: convolve along y axis
    output = np.zeros_like(mat, dtype = np.float64)
    for x in range(width):
        for y in range(height):
            crop = padded_mat[y:y+y_len,x+r_x,:]
            output[y, x, :] = _calculate_weighted_average(crop, kernel_y[::-1], axis = 0)

    # second pass: convolve along x axis
    padded_out = np.pad(output, pad_width = ((r_y, r_y),(r_x, r_x), (0,0)))
    output = np.zeros_like(mat, dtype = np.float64)
    for y in range(height):
        for x in range(width):
            crop = padded_out[y+r_y, x:x+x_len,:]
            output[y,x,:] = _calculate_weighted_average(crop, kernel_x, axis = 0)

    return output.astype(mat.dtype)

def fourier_transorm(): # ???
    pass

def fast_fourier_transform():
    pass

def fft_convolve(x, y):
    pass
