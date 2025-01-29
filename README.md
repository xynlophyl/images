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
    - https://www.youtube.com/watch?v=KuXjwB4LzSA&t=1090s
    - https://www.youtube.com/watch?v=IaSGqQa5O-M

- Fast Fourier Transform
    - https://www.youtube.com/watch?v=spUNpyF58BY
    - https://www.youtube.com/watch?v=h7apO7q16V0

- SSIM
    - https://www.cns.nyu.edu/pub/lcv/wang03-reprint.pdf
    - https://en.wikipedia.org/wiki/Structural_similarity_index_measure
    - https://scikit-image.org/docs/0.24.x/auto_examples/transform/plot_ssim.html#structural-similarity-index