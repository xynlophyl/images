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

    # generate global ssim score
    score1 = ssim.calculate_global_ssim(source, target)
    end = time.time()
    print(f"ssim score: {score1:.2f}")
    print(f"time elapsed: {str(end-start) + ' s' if end-start >= 1 else str((end-start)*1000) + ' ms'}")
    
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

    # # blur image
    # print('blur')
    # start = time.time()
    # blurred_image= image_processor.blur(source_image, mask_size=3)
    # print(f't: {time.time() - start}')
    

    # print('blur fast')
    # start = time.time()
    # x = image_processor.blur(source_image, mask_size=7, mode="fast-convolve")
    # print(f't: {time.time() - start}')

    # show_image(cv2.hconcat([source_image, blurred_image, x]))

    # # sharpen image
    # print('sharpen')
    # blurred_image = image_processor.sharpen(source_image)
    # show_image(cv2.hconcat([source_image, blurred_image]))

    # # gaussian blur
    # print('gaussian blur')
    # gaussian_image = image_processor.gaussian_blur(source_image, std_dev=1.5, mask_size = 11)
    # show_image(cv2.hconcat([source_image, gaussian_image]))
    
    # ssim
    img2 = cv2.imread("assets/mona-lisa-gauss-1.png")

    evaluate_ssim(source_image, img2)


if __name__ == "__main__":
    main()
