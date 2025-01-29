import numpy as np

def convolve(mat: np.ndarray, kernel: np.ndarray, stride: int = 1) -> np.array:
    # get kernel dimensions
    kernel_h, kernel_w, _ = kernel.shape

    # get matrix dimensions
    height, width, _ = mat.shape

    # add padding to height and width of image
    radius_y, radius_x = kernel_h//2, kernel_w//2
    padded_image = np.pad(mat, pad_width = ((radius_y, radius_y), (radius_x, radius_x), (0,0)))

    output = np.zeros_like(mat)
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            crop = padded_image[y:y+kernel_h, x:x+kernel_w,:]
            output[y,x,:] = np.sum(crop*kernel[::-1], axis = (0,1))
    
    return output.astype(mat.dtype)

def fast_convolve(mat, kernel, stride):
    # get kernel dimensions
    kernel_h, kernel_w, _ = kernel.shape

    # get matrix dimensions
    height, width, _ = mat.shape

    # add padding to height and width of image
    radius_y, radius_x = kernel_h//2, kernel_w//2
    padded_image = np.pad(mat, pad_width = ((radius_y, radius_y), (radius_x, radius_x), (0,0)))

    output = np.zeros_like(mat, dtype=float)
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            for dy in range(radius_y):
                for dx in range(radius_x):
                    output[y,x,:] += padded_image[y+dy, x+dx,:]*kernel[dy,dx,:] 

    return output.astype(mat.dtype)

def fourier_transorm(): # ???
    pass

def fast_fourier_transform():
    pass

def fft_convolve(x, y):
    pass


"""
0,0,0,0,0                    00,00,00,00,00
0,1,2,3,0      1,1,1         00,12,21,16,00
0,4,5,6,0  * [ 1,1,1 ]x1/9=> 00,27,45,33,00 x 1/9
0,7,8,9,0      1,1,1         00,24,39,28,00
0,0,0,0,0                    00,00,00,00,00

1,2,3,4,5,6,7,8,9
1,1,1,1,1,1,1,1,1
"""


