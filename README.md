# Images: implementations of image processing and analysis techniques

Convolutions:

Fast Fourier Transform:

Gaussian Blur: blurring image using convolution

Edge Detection: using matrix like [[0.25, 0, -0.25], [0.5, 0, -0.5], [0.25, 0, -0.25]] to detect variations between left and right edges of convolutional window (transpose matrix to detect horizontal edges )

Sharpening: 

Structural Similarity Index Measure (SSIM): measuring similarity between two images

### Benchmarks
#### Convolve (Box-blur, 2500px x 1967px image, mask_size = 25)

    Naive (w/ numpy): 82.99s  
    Two-Pass (Separable Masks): 61.2s
    FFT: 342.16s

### References:

- Convolutions
    - Convolution (Wikipedia): https://en.wikipedia.org/wiki/Convolution
    - But what is a convolution? (3Blue1Brown, Youtube): https://www.youtube.com/watch?v=KuXjwB4LzSA
    - Convolutions in Image Processing (The Julia Programming Language, Youtube): https://www.youtube.com/watch?v=8rrHTtUzyZA
    - 2D Convolution using Python and NumPy (Samrat Sahoo, Medium): https://medium.com/analytics-vidhya/2d-convolution-using-python-numpy-43442ff5f381
    - A basic introduction to separable convolutions (Chi-Feng Wang, Towards Data Science): https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728

- Image Processing
    - Kernel, Image Processing (Wikipedia): https://en.wikipedia.org/wiki/Kernel_(image_processing) 
    - Gaussian Blur (Wikipedia): https://en.wikipedia.org/wiki/Gaussian_blur
    - Image Kernels, explained visually (Victor Powell, Blog): https://setosa.io/ev/image-kernels/#:~:text=An%20image%20kernel%20is%20a,important%20portions%20of%20an%20image.

- Fast Fourier Transform
    - The Fast Fourier Transform (FFT): Most Ingenious Algorithm Ever? (Reducible, Youtube): https://www.youtube.com/watch?v=KuXjwB4LzSA
    - FFT-based 2D convolution (Research Paper, NVIDIA): https://developer.download.nvidia.com/compute/cuda/2_2/sdk/website/projects/convolutionFFT2D/doc/convolutionFFT2D.pdf

- SSIM
    - https://www.cns.nyu.edu/pub/lcv/wang03-reprint.pdf
    - https://en.wikipedia.org/wiki/Structural_similarity_index_measure
    - https://scikit-image.org/docs/0.24.x/auto_examples/transform/plot_ssim.html#structural-similarity-index