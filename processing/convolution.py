import numpy as np
from .fourier_transform import fast_fourier_transform, inverse_fast_fourier_transform
from typing import List, Tuple

def _calculate_weighted_average(x, y, axis: int | Tuple[int, int]):
    return np.sum(x*y, axis = axis)

def naive_convolve(mat: np.ndarray, kernel: np.ndarray):
    """
    naive implementation (4 nested loops) of computing convolutions between a (m,n) matrix and a (k,l) kernel
        first, add padding of lengths (k//2, l//2) to the horizontal and vertical sides of the matrix
        next, for each cell in the original image, calculate the weighted average of the current window of values, 
    """
    # get kernel dimensions
    kernel_h, kernel_w, _ = kernel.shape

    # get matrix dimensions
    height, width, _ = mat.shape

    # add padding to height and width of matrix
    radius_y, radius_x = kernel_h//2, kernel_w//2
    padded_mat = np.pad(mat, pad_width = ((radius_y, radius_y), (radius_x, radius_x), (0,0)))

    output = np.zeros_like(mat, dtype = np.float64)
    for y in range(0, height):
        for x in range(0, width):
            for dy in range(kernel_h):
                for dx in range(kernel_w):
                    output[y,x,:] += padded_mat[y+dy, x+dx,:]*kernel[dy,dx,:]
    
    return output.astype(mat.dtype)

def convolve(mat: np.ndarray, kernel: np.ndarray) -> np.array:
    """
    conolve using numpy for faster matrix calculation
    """
    # get kernel dimensions
    kernel_h, kernel_w, _ = kernel.shape

    # get matrix dimensions
    height, width, _ = mat.shape

    # add padding to height and width of matrix
    radius_y, radius_x = kernel_h//2, kernel_w//2
    padded_mat = np.pad(mat, pad_width = ((radius_y, radius_y), (radius_x, radius_x), (0,0)))

    output = np.zeros_like(mat, dtype = np.float64)
    for y in range(0, height):
        for x in range(0, width):
            crop = padded_mat[y:y+kernel_h, x:x+kernel_w,:]
            output[y,x,:] = _calculate_weighted_average(crop, kernel[::-1], axis = (0,1))

    return output.astype(mat.dtype)

def spatial_convolve(mat:  np.ndarray, kernels_xy: List[np.ndarray]):
    """
    convolve utlizing spatial separable kernel, allowing for a more efficient two pass solution 
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
    for x in range(0, width, 1):
        for y in range(height):
            crop = padded_mat[y:y+y_len,x+r_x,:]
            output[y, x, :] = _calculate_weighted_average(crop, kernel_y[::-1], axis = 0)

    # second pass: convolve along x axis
    padded_out = np.pad(output, pad_width = ((r_y, r_y),(r_x, r_x), (0,0)))
    output = np.zeros_like(mat, dtype = np.float64)
    for y in range(0, height, 1):
        for x in range(0, width):
            crop = padded_out[y+r_y,x:x+x_len,:]
            output[y,x,:] = _calculate_weighted_average(crop, kernel_x, axis = 0)

    return output.astype(mat.dtype)

def fft_convolve(mat: np.ndarray, kernel: np.ndarray):

    def _fft_conv(mat: np.ndarray, kernel: np.ndarray):
        mat_h, mat_w = mat.shape[:2]
        kernel_h, kernel_w = kernel.shape[:2]

        # add padding to height and width of matrix
        padded_mat = np.pad(mat, pad_width = ((0, kernel_h-1), (0, kernel_w-1)))
        
        # add padding to height and width of kernel
        padded_kernel = np.pad(kernel, pad_width = ((0, mat_h-1), (0, mat_w-1)))

        # compute fft for both inputs
        mat_fft = fast_fourier_transform(padded_mat)
        kernel_fft = fast_fourier_transform(padded_kernel)

        # multiply element-wise
        output_fft = mat_fft*kernel_fft

        # apply inverse fft to obtain final output
        output = inverse_fast_fourier_transform(output_fft)

        m = padded_mat.shape[0]*padded_mat.shape[1]
        return output[:m].reshape(padded_mat.shape)
    
    dims = len(mat.shape)
    
    # borders for the output image
    kernel_h, kernel_w = kernel.shape[:2]
    b_h = kernel_h//2
    b_w = kernel_w//2

    if dims > 3:
        raise ValueError("np.ndarray dimensionality > 3 is not handled in current impl")
    elif dims == 3: # rgb color model
        mat_r, mat_g, mat_b = mat[:,:,0], mat[:,:,1], mat[:,:,2]
        kernel_r, kernel_g, kernel_b = kernel[:,:,0], kernel[:,:,1], kernel[:,:,2]
        
        output_r = _fft_conv(mat_r, kernel_r)
        output_g = _fft_conv(mat_g, kernel_g)
        output_b = _fft_conv(mat_b, kernel_b)

        return np.stack([output_r, output_g, output_b], axis =2)[b_h:-b_h, b_w:-b_w, :].astype(mat.dtype)
    else:
        return _fft_conv(mat)[b_h:-b_h, b_w:-b_w].astype(mat.dtype)
    