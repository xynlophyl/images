# Images: implementations of image processing and analysis techniques

Convolutions:

Fast Fourier Transform:

Gaussian Blur: blurring image using convolution

Edge Detection: using matrix like [[0.25, 0, -0.25], [0.5, 0, -0.5], [0.25, 0, -0.25]] to detect variations between left and right edges of convolutional window (transpose matrix to detect horizontal edges )

Sharpening: 

Structural Similarity Index Measure (SSIM): measuring similarity between two images

### Benchmarks
400 x 400 pixel image

    Initial Implementation: 28.41s (can be improved) 

References:

- Convolution
    - Convolution (Wikipedia): https://en.wikipedia.org/wiki/Convolution
    - But what is a convolution? (3Blue1Brown, Youtube): https://www.youtube.com/watch?v=KuXjwB4LzSA
    - Convolutions in Image Processing (The Julia Programming Language, Youtube): https://www.youtube.com/watch?v=8rrHTtUzyZA
    - 2D Convolution using Python and NumPy (Samrat Sahoo, Medium): https://medium.com/analytics-vidhya/2d-convolution-using-python-numpy-43442ff5f381
    - All about convolutions, kernels, features in CNN (Abhishek Jain, Medium)
    - A basic introduction to separable convolutions (Chi-Feng Wang, Towards Data Science): https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728

- Image Processing
    - Image kernels, explained visually (Victor Powell, Blog)
    - Kernel, Image Processing (Wikipedia): https://en.wikipedia.org/wiki/Kernel_(image_processing) 
    - Gaussian Blur (Wikipedia): https://en.wikipedia.org/wiki/Gaussian_blur
    - 

- Fast Fourier Transform
    - https://www.youtube.com/watch?v=spUNpyF58BY
    - https://www.youtube.com/watch?v=h7apO7q16V0

- SSIM
    - https://www.cns.nyu.edu/pub/lcv/wang03-reprint.pdf
    - https://en.wikipedia.org/wiki/Structural_similarity_index_measure
    - https://scikit-image.org/docs/0.24.x/auto_examples/transform/plot_ssim.html#structural-similarity-index