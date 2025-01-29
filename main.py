from processing import ImageProcessor
from diff import SSIM
from utils import show_image
from skimage.metrics import structural_similarity
from skimage.color import rgb2gray
import cv2
import time

def evaluate_ssim(source, target):
    start = time.time()
    # load SSIM
    ssim = SSIM(
        window_size = 40
    )

    # generate global ssim score
    score1 = ssim.calculate_global_ssim(source, target)
    end = time.time()
    print(f"ssim score: {score1:.2f}")
    print(f"time elapsed: {str(end-start) + ' s' if end-start >= 1 else str((end-start)*1000) + ' ms'}")
    
    # get ssim from skimage implementation
    source_gray = rgb2gray(source)
    target_gray = rgb2gray(target) 

    score2 = structural_similarity(
        source_gray, target_gray, 
        multichannel=True, 
        gaussian_weights=True, 
        sigma=1.35, 
        use_sample_covariance=False, 
        data_range=1.0
    )
    print("truth:", score2)

    # generate ssim map
    start = time.time()
    ssim_map = ssim.generate_ssim_map(source, target)
    end = time.time()
    print(f"time elapsed: {str(end-start) + 's' if end-start>= 1 else str((end-start)*1000) + 'ms'}")
    show_image(cv2.hconcat([source, target, ssim_map]))

def main():
    print("read image")
    source_path = "assets/jamie_small.jpg"
    source_path = "assets/mona-lisa.png"
    source_image = cv2.imread(source_path)

    # target_path = "../van-gogh-box-blur.jpeg"
    # blurred_image = cv2.imread(target_path)
    
    # load image processor
    image_processor = ImageProcessor()

    # blur image
    print('blur')
    start = time.time()
    blurred_image= image_processor.blur(source_image, mask_size=7)
    print(f't: {time.time() - start}')
    print('blur fast')
    start = time.time()
    x = image_processor.blur(source_image, mask_size=7, mode="fast-convolve")
    print(f't: {time.time() - start}')

    show_image(cv2.hconcat([source_image, blurred_image, x]))

    # # sharpen image
    # print('sharpen')
    # blurred_image = image_processor.sharpen(source_image)
    # show_image(cv2.hconcat([source_image, blurred_image]))

    # # gaussian blur
    # print('gaussian blur')
    # gaussian_image = image_processor.gaussian_blur(source_image)
    # show_image(cv2.hconcat([source_image, gaussian_image]))
    
    # # ssim
    # evaluate_ssim(source_image, img2)


if __name__ == "__main__":
    main()
