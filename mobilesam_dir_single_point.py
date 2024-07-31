import os
import argparse
import glob
import functools, operator

import numpy as np
import cv2
import torch
from mobile_sam import sam_model_registry, SamPredictor

def get_torch_device():
    if torch.cuda.is_available(): return "cuda"
    elif torch.backends.mps.is_available(): return "mps"
    return "cpu"

def mobilesam_dir_single_point(image_dir_path, 
                                mobilesam_weights_path, 
                                include_subdirs = True, 
                                click_point = True, 
                                output_folder = "", 
                                resize_width = -1):
    # load the MobileSAM model
    model_type = "vit_t"
    sam_checkpoint = os.path.join(mobilesam_weights_path, "mobile_sam.pt")
    mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    mobile_sam.to(device= get_torch_device())
    mobile_sam.eval()

    predictor = SamPredictor(mobile_sam)

    # extract image file paths inside the dir using glob (possibly including subdirectories) 
    accepted_exts = [".jpg", ".jpeg", ".png"]
    image_files =  [glob.glob(os.path.join(image_dir_path,  "*"+ex)) for ex in accepted_exts]
    if include_subdirs:
        image_files.extend([glob.glob(os.path.join(image_dir_path + "/**/",  "*"+ex)) for ex in accepted_exts])
    image_files = functools.reduce(operator.iconcat, image_files, [])
    
    for file_path in image_files:
        print(f"\nProcessing {file_path}")
        image = cv2.imread(file_path)

        # compute scale factors, to be use if necessary
        s_width = resize_width/image.shape[1]
        s_height = (s_width*image.shape[0])/image.shape[0]
    
        # a single class is defined.
        input_label = np.array([1])

        # save image for visualization
        image_viz = image.copy()
        if resize_width > 0: image_viz = cv2.resize(image_viz, (0,0), fx=s_width, fy=s_height)
        
        global input_point
        input_point = None

        if not click_point:
            input_point = [image_viz.shape[1]//2, image_viz.shape[0]//2]
        else:
            def click_event(event, x, y, flags, param):
                global input_point
                if event == cv2.EVENT_LBUTTONUP:
                    input_point = [x, y]
                    # print(f"Click {x},{y}")

            cv2.namedWindow("mobilesam_dir_single_point")
            cv2.setMouseCallback("mobilesam_dir_single_point", click_event)
            while input_point is None:
                cv2.imshow("mobilesam_dir_single_point", image_viz)
                cv2.waitKey(1)

        image_viz_circle = image_viz.copy()
        cv2.circle(image_viz_circle, (input_point[0], input_point[1]), 9, (0, 0, 255), -1)
        cv2.imshow("mobilesam_dir_single_point", image_viz_circle)
        cv2.waitKey(1)

        if resize_width > 0:
            # import pdb; pdb.set_trace()
            msam_input_point = np.array([[int(input_point[0]/s_width), int(input_point[1]/s_height)]]) 
        else:
            msam_input_point = np.array([input_point])
        
            
        start = cv2.getTickCount()
        predictor.set_image(image)
        masks, scores, _ = predictor.predict(
            point_coords=msam_input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        end = cv2.getTickCount()
        time = (end - start) / cv2.getTickFrequency()
        print(f"Processing time: {time} s")

        
        # multmask_output == True makes MobileSAM to return 3 masks, with scores estimating 
        # the quality of each mask. We'll get the mask with the highest score
        best_mask = masks[np.argmax(scores)]
        best_mask = best_mask.astype(np.uint8)*255

        # plot mask over image
        green_mask = np.zeros_like(image_viz)
        mask_viz = cv2.resize(best_mask, (0,0), fx=s_width, fy=s_height) if resize_width > 0 else best_mask.copy()
        green_mask[mask_viz == 255] = (0, 255, 0)

        image_viz = cv2.addWeighted(image_viz, 0.5, green_mask, 0.5, 0)
        cv2.circle(image_viz, (input_point[0], input_point[1]), 9, (0, 0, 255), -1)

        cv2.imshow("mobilesam_dir_single_point", image_viz)
        cv2.waitKey(500) # wait for half a second for the next image

        if output_folder != "":
            os.makedirs(output_folder, exist_ok=True)
            file_name = os.path.basename(file_path)
            cv2.imwrite(os.path.join(output_folder, file_name), image_viz)

            cv2.imshow("mobilesam_dir_single_point", mask_viz)
            cv2.waitKey(200)

            # also saves an image with the only the mask
            mask_filename = os.path.join(output_folder, os.path.splitext(file_name)[0]+"_mask.png")
            print("Saving output mask to", mask_filename)
            cv2.imwrite(mask_filename, best_mask)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--image_dir_path", type=str, required=True, help="Path to the image directory")
    args.add_argument("--mobilesam_weights_path", type=str, default="./mobilesam_weights/", 
                                                help="Path to the MobileSAM weights")
    args.add_argument("--exclude_subdirs", action=argparse.BooleanOptionalAction, default=False,
                                                help="Exclude subdirectories in the image directory")
    args.add_argument("--output_folder", type=str, default="", 
                                                help="(Optional) Output folder to save the results (segmentations and masks)")
    args.add_argument("--use_image_center_point", action=argparse.BooleanOptionalAction, default=False,
                                                help="Instead of clicking a point, use the center of the image to guide the segmentation")
    args.add_argument("--resize_width", type=int, default=-1,
                                                help="Rezie images, for visualization purposes, to the desired width and original aspect ratio.")
    args = args.parse_args()

    mobilesam_dir_single_point(args.image_dir_path, 
                               args.mobilesam_weights_path, 
                               not args.exclude_subdirs,
                               not args.use_image_center_point, 
                               args.output_folder,
                               args.resize_width)

    