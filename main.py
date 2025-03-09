import argparse
import cv2
from tests import process_image, compare_images

CONVOLUTIONS = ["blur", "sharpen", "gaussian", "ridge"]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("sourcepath")
    parser.add_argument(
        "action",
        choices = ["blur", "sharpen", "gaussian", "ridge", "ssim"]
    )
    parser.add_argument(
        "--method",
        choices = ["naive", "spatial", "fft"]
    )

    parser.add_argument("--targetpath")
    parser.add_argument("--mask-size", type=int)
    parser.add_argument("--mask")
    parser.add_argument("--std", type = float)
    parser.add_argument("--scale", type = int)
    parser.add_argument("--window-size", type= int)

    args = parser.parse_args()

    # add argument conditions (i.e. window size for ssim)

    source_image = cv2.imread(args.sourcepath)
    action = args.action

    kwargs = {k: v for k, v in vars(args).items() if v}
    del kwargs["sourcepath"]
    del kwargs["action"]

    if action in CONVOLUTIONS:
        print('action', action)
        print(kwargs)
        process_image(source_image, action, **kwargs)
    
    elif action == "ssim":
        target_image = cv2.imread(kwargs["targetpath"])
        del kwargs["targetpath"]
        
        compare_images(source_image, target_image, **kwargs)
        
    elif action == "magicwand":
        ...
