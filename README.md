# MobileSAM Annotator
A blazing fast (and simple) annotator tool using SAM (Segment Anything Models) using the awesome [MobileSAM](https://github.com/ChaoningZhang/MobileSAM).

## `mobilesam_dir_single_point.py`

This script take all the images inside a folder and request a single point from the user to segment the desired object (single one per image).

### What it looks like?

![How easy is this?](mobilesam_annotator_sample.gif)


### Requirements:
```
pip install matplotlib \ 
            git+https://github.com/ChaoningZhang/MobileSAM.git \
            torch \
            opencv-python
            timm
```

### Quickstart:

### 1. Download the model from [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)

`wget https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt -P mobilesam_weights/`

### 2. Run!

Sample exec:
```
python3 mobilesam_dir_single_point.py --image_dir_path sample_images/
                                        --mobilesam_weights_path mobilesam_weights
                                        --output_folder out_sample_images_segmentation/ 
                                        --resize_width 640
```

All arguments:
```
usage: mobilesam_dir_single_point.py [-h] --image_dir_path IMAGE_DIR_PATH
                                     [--mobilesam_weights_path MOBILESAM_WEIGHTS_PATH]
                                     [--exclude_subdirs | --no-exclude_subdirs] [--output_folder OUTPUT_FOLDER]
                                     [--use_image_center_point | --no-use_image_center_point]
                                     [--resize_width RESIZE_WIDTH]

options:
  --image_dir_path IMAGE_DIR_PATH
                Path to the image directory
  --mobilesam_weights_path MOBILESAM_WEIGHTS_PATH
                Path to the MobileSAM weights
  --exclude_subdirs
                Exclude subdirectories in the image directory
  --output_folder OUTPUT_FOLDER
                (Optional) Output folder to save the results (segmentations and masks)
  --use_image_center_point
                Instead of clicking a point, use the center of the image to guide the segmentation
  --resize_width RESIZE_WIDTH
                Resize images, for visualization purposes, to the desired width and original aspect ratio.
```

If your object is centralized in all images, you don't need to click anything, just use ```--use_image_center_point```



### TODO:
- Convert masks to bounding boxes annotations, specially in the [COCO format](https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html);
- Decouple interface code;
- Make possible to click several times and for new classes;

### Acknowledges

[MobileSAM](https://github.com/ChaoningZhang/MobileSAM) made it possible to really use SAM for annotation, since it's incredibly accurate for its speed.
