from processing import ImageProcessor
from diff import SSIM
from utils import show_image
from skimage.metrics import structural_similarity
from skimage.color import rgb2gray, gray2rgb
import cv2
import time

def evaluate_ssim(source, target):
    show_image(cv2.hconcat([source, target]))

    start = time.time()
    # load SSIM
    ssim = SSIM(
        window_size = 7
    )
    
    # get ssim from skimage implementation
    source_gray = rgb2gray(source)
    target_gray = rgb2gray(target) 

    score_truth, ssim_map_truth = structural_similarity(
        source_gray, target_gray, 
        multichannel=True, 
        gaussian_weights=True, 
        sigma=1.35, 
        use_sample_covariance=False, 
        data_range=1.0,
        full = True
    )
    print("truth:", score_truth)

    # generate ssim map
    print('generating ssim map')
    start = time.time()
    score2, ssim_map = ssim.generate_ssim_map(source, target)
    end = time.time()
    print(f"mean ssim score: {score2:.2f}")
    print(f"time elapsed: {str(end-start) + 's' if end-start>= 1 else str((end-start)*1000) + 'ms'}")

    show_image(ssim_map)

def main():
    print("read image")
    # source_path = "assets/jamie_small.jpg"
    source_path = "assets/mona-lisa.png"
    source_image = cv2.imread(source_path)

    # target_path = "../van-gogh-box-blur.jpeg"
    # blurred_image = cv2.imread(target_path)
    
    # load image processor
    image_processor = ImageProcessor()

    # blur image
    print('blur')
    start = time.time()
    blurred_image = image_processor.blur(source_image, mask_size=11)
    print(f't: {time.time() - start}')
    
    # print('blur naive')
    # start = time.time()
    # x = image_processor.blur(source_image, mask_size=11, method="naive-convolve")
    # print(f't: {time.time() - start}')

    print('blur spatial')
    start = time.time()
    y = image_processor.blur(source_image, mask_size=11, method="spatial-convolve")
    print(f't: {time.time() - start}')
    # show_image(cv2.hconcat([source_image, y]))

    print((blurred_image == y).all())
    show_image(cv2.hconcat([blurred_image, y]))

    # # sharpen image
    # print('sharpen')
    # blurred_image = image_processor.sharpen(source_image)
    # show_image(cv2.hconcat([source_image, blurred_image]))

    # # gaussian blur
    # print('gaussian blur')
    # gaussian_image = image_processor.gaussian_blur(source_image, std_dev=1, mask_size = 11)
    # x = image_processor.gaussian_blur(source_image, std_dev=1, mask_size = 11, method = "naive-convolve")
    # y = image_processor.gaussian_blur(source_image, std_dev=1, mask_size = 11, method = "spatial-convolve")
    # print((gaussian_image==x).all(), (gaussian_image == y).all())
    # show_image(cv2.hconcat([source_image, gaussian_image, x, y]))
    

    # # ssim
    # img2 = cv2.imread("assets/mona-lisa-gauss-1.png")
    # evaluate_ssim(source_image, img2)


if __name__ == "__main__":
    main()
