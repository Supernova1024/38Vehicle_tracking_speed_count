# Object Tracking using YOLOv3, Deep Sort and Tensorflow

## Getting started

#### Conda (Recommended)

```bash
# Tensorflow CPU
conda env create -f conda-cpu.yml
conda activate tracker-cpu

# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate tracker-gpu
```

#### Pip
```bash
# TensorFlow CPU
pip install -r requirements.txt

# TensorFlow GPU
pip install -r requirements-gpu.txt
```

### Nvidia Driver (For GPU, if you haven't set it up already)
```bash
# Ubuntu 18.04
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt install nvidia-driver-430
# Windows/Other
https://www.nvidia.com/Download/index.aspx
```

## Running the Object Tracker
Now you can run the object tracker for whichever model you have created, pretrained, tiny, or custom.
```
# yolov3 on video
python object_tracker.py --video ./data/video/test.mp4 --output ./data/video/results.avi

#yolov3 on webcam 
python object_tracker.py --video 0 --output ./data/video/results.avi

```
The output flag saves your object tracker results as an avi file for you to watch back. It is not necessary to have the flag if you don't want to save the resulting video.

There is a test video uploaded in the data/video folder called test.mp4.

## Command Line Args Reference
```
Run the script generally:
  python object_tracker.py --video ./data/video/test.mp4 --output ./data/video/results.avi

Run the script with parameters:
  object_tracker.py:
    --classes: path to classes file
      (default: './data/labels/coco.names')
    --video: path to input video (use 0 for webcam)
      (default: './data/video/test.mp4')
    --output: path to output video (remember to set right codec for given format. e.g. XVID for .avi)
      (default: None)
    --output_format: codec used in VideoWriter when saving video to file
      (default: 'XVID)
    --[no]tiny: yolov3 or yolov3-tiny
      (default: 'false')
    --weights: path to weights file
      (default: './weights/yolov3.tf')
    --num_classes: number of classes in the model
      (default: '80')
      (an integer)
    --yolo_max_boxes: maximum number of detections at one time
      (default: '100')
      (an integer)
    --yolo_iou_threshold: iou threshold for how close two boxes can be before they are detected as one box
      (default: 0.5)
      (a float)
    --yolo_score_threshold: score threshold for confidence level in detection for detection to count
      (default: 0.5)
      (a float)
```

