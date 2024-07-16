# YOLO Curb Detection without Classes

This repository provides code that detects the boundary of obstacles (curbs) using the color and depth information provided by an OAK-D Pro. It uses intermediate data from a YOLO model to find object boundaries given some example points in the inside of the object.

The concept document is [here](https://docs.google.com/document/d/1lS5mrRXDqX3NMxOKWP08jXFPXZpRKm6UVFRw1kWfFeQ), which describes the overall principle, processing pipeline, failure cases, and abandoned approaches.

The model blob is borrowed from [here](https://github.com/tirandazi/depthai-yolov8-segment), although it is possible to generate one from scratch. We don't use the bounding box branch of the network, so future development may remove it from the original pytorch model and regenerate a smaller blob.

## Quickstart

These steps get the code running on an oakd-pro connected via ethernet to `10.25.76.102`

1. Install pip packages by running `pip install -r requirements.txt` (there's also a conda `environment.yml` file which may be better at reproducing the setup exactly)
2. Run `python main.py`
3. Should see serveral windows popup: the `Ground/Obs/Curbs` masks generated from the depth data
   The `Depth` output from the camera
   The final `Output`, which is YOLO masks and curbs overlaid on the input RGB
4. To show the prototype masks, pass `--show_protos` to the cmdline

