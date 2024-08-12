import json
from math import radians

import cv2
import depthai as dai
import numpy as np

from pc import PointCloudVisualizer, create_projection_matrix, depth_to_3d
from ransac import ransac_indices
from scipy.ndimage import sobel
import argparse


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

with open("config.json", "r") as config:
    model_data = json.load(config)

# Set up some constants for convenience
model_img_width = model_data["input_width"]
model_img_height = model_data["input_height"]
model_input_shape = [1, 3, model_img_height, model_img_width]

model_coefficient_shape = model_data["shapes"]["output0"]
model_proto_mask_shape = model_data["shapes"]["output1"]

path_to_yolo_blob = "models/yolov8n-seg.blob"

def main(show_protos):
    pipeline = dai.Pipeline()

    # Create node to sync RGB, NN, and Depth streams
    sync = pipeline.create(dai.node.Sync)
    xoutGrp = pipeline.create(dai.node.XLinkOut)
    xoutGrp.setStreamName("xout")

    # Neural network pipeline properties
    nn = pipeline.createNeuralNetwork()
    nn.setBlobPath(path_to_yolo_blob)
    nn.out.link(sync.inputs["xout_yolo_nn"])

    # Color cam properties (Cam_A/RGB)
    cam_rgb = pipeline.createColorCamera()
    cam_rgb.setPreviewSize(model_img_width, model_img_height)
    cam_rgb.setInterleaved(False)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
    cam_rgb.preview.link(nn.input)
    cam_rgb.preview.link(sync.inputs["xout_rgb"])
    print("Color cam resolution: ", cam_rgb.getResolutionSize())

    # Stereo/Depth node properties
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A) # Align to RGB
    stereo.setOutputSize(model_img_width, model_img_height)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)
    stereo.setExtendedDisparity(True)
    stereo.depth.link(sync.inputs["xout_depth"])

    # Link
    sync.out.link(xoutGrp.input)

    # Left cam properties (Cam_B/Mono)
    left = pipeline.create(dai.node.MonoCamera)
    left.setCamera("left")
    left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    left.out.link(stereo.left)
    print("Left cam resolution: ", left.getResolutionSize())

    # Right cam properties (Cam_C/Mono)
    right = pipeline.create(dai.node.MonoCamera)
    right.setCamera("right")
    right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    right.out.link(stereo.right)
    print("Right cam resolution: ", right.getResolutionSize())
    
    # Connect to device
    device_info = dai.DeviceInfo("10.25.76.102")
    device = dai.Device(pipeline, device_info)

    # Create camera projection matrix (depth map -> pointcloud)
    projection_matrix = create_projection_matrix(device, (model_img_height, model_img_width))

    device.setIrLaserDotProjectorBrightness(500) # 0 to 1200

    # Get output queue
    sync_queue = device.getOutputQueue("xout", 5, False)

    pcl_converter = PointCloudVisualizer()
    
    while True:
        # Quit if q is pressed
        if cv2.waitKey(1) == ord("q"):
            break
        
        # Read from output queue, and skip if anything is missing
        message_group = sync_queue.get()
        for name, msg in message_group:
            match name:
                case "xout_rgb":
                    frame = msg.getCvFrame()
                case "xout_depth":
                    depth_frame = msg.getCvFrame()
                case "xout_yolo_nn":
                    yolo_nn_queue_msg = msg

        if frame is None or depth_frame is None or yolo_nn_queue_msg is None:
            continue
        
        # This outputs the bounding boxes and classes, which we don't care about
        # output0 = np.reshape(
        #     yolo_nn_queue_msg.getLayerFp16("output0"),
        #     newshape=(model_coefficient_shape),
        # )

        # Exract proto masks from NN 
        output1 = np.reshape(
            yolo_nn_queue_msg.getLayerFp16("output1"),
            newshape=(model_proto_mask_shape),
        )

        if len(output1) == 0:
            continue

        if show_protos:
            rows = [
                np.concatenate((output1[0, 0:8]), axis=1),
                np.concatenate((output1[0, 8:16]), axis=1),
                np.concatenate((output1[0, 16:24]), axis=1),
                np.concatenate((output1[0, 24:32]), axis=1),
            ]

            bigone = np.concatenate(rows, axis=0)

            cv2.imshow("All Prototypes", bigone)

        # cv2.imshow("Color Input", frame)

        # Get pointcloud
        points = depth_to_3d(depth_frame, projection_matrix)
        print('3D points shape', points.shape)
        
        # Get image-space masks for ground, obstacles, and curbs
        ground_points, floor_mask, obs_mask, curb_mask = ransac_indices(points)

        # Visualize pointcloud
        # pcl_converter.visualize_pcl(points, downsample=False)

        # Dilate the curb mask with a semi-circle kernel
        # These are all the areas that are near a ground-truth curb as detected by the depth map
        R = 35
        kernel = np.zeros((2 * R + 1, 2 * R + 1), np.uint8)
        cv2.circle(kernel, (R, R), R, 1, -1)
        kernel[R:(2*R+1), :] = 0
        curb_mask_dilate = cv2.dilate(curb_mask, kernel, iterations=1)

        cv2.imshow('Grounds/Obs/Curbs', np.concatenate((floor_mask * 255,
                                                   obs_mask * 255,
                                                   curb_mask * 255), axis=1))
        
        # Resize these masks from prototype size to RGB size, so we can compare them
        floor_mask = cv2.resize(floor_mask, (output1.shape[3], output1.shape[2]), interpolation=cv2.INTER_AREA)
        obs_mask = cv2.resize(obs_mask, (output1.shape[3], output1.shape[2]), interpolation=cv2.INTER_AREA)
        curb_mask_dilate = cv2.resize(curb_mask_dilate, (output1.shape[3], output1.shape[2]), interpolation=cv2.INTER_AREA)

        print("Prototype shape ", output1.shape)

        # Optimize protos

        proto_sum = np.zeros(output1.shape[2:4], float)

        # Find the mask coefficients with highest adherence to dot
        for i, proto in enumerate(output1[0]):
            
            # Manually excluded some protos that only activate for partial regions
            # (left side, bottom edge, etc). Hopefully this encourages the
            # final masks to be more global. But you can experiment with this.
            if i in [2, 8, 12, 14, 19, 20, 24, 26, 27, 29, 31]:
                continue
            
            # At some point I penalized masks for areas that the depth map
            # didn't exist to make the masks mroe conservative. But it didn't
            # seem to make a difference visually
            unknown = (obs_mask == 0) & (floor_mask == 0) 
            
            # This part works by finding the best values for the formula 
            #              c * (proto - bias)
            # such that the result maximizes the dot product proto * obs_mask
            # and minimizes the dot product proto * floor_mask.
            # This is based on some vauge memory of Maximum Likelihod Estimation,
            # so I'm not sure if it's correct, but it works well enough somehow
            floor_similarity = (proto * floor_mask).sum() / floor_mask.sum()
            obs_similarity = (proto * obs_mask).sum() / obs_mask.sum()
            mean = (floor_similarity + obs_similarity) / 2
            coeff = obs_similarity - mean 

            proto_sum += coeff * (proto - mean)
        
        # Manual protos (Good for detecting only obvious objects)
        # proto_sum = output1[0, 1] + output1[0, 2] + output1[0, 4] + output1[0, 6] + -1 * output1[0, 16] + -1 * output1[0, 21]

        conf_thresh = 0.3

        # Create proto_sum_dilate, which is a mask of all regions _close_ to an obstacle mask.
        # The point of this is to allow curbs on the bottom edge to be counted
        # We also threshold by a confidence value, which binarizes what we consider a mask and
        # what is just noise
        kernel = np.ones((5,5),np.uint8)
        proto_sum_dilate = proto_sum.copy()
        proto_sum_dilate[proto_sum < conf_thresh] = 0
        proto_sum_dilate = cv2.dilate(proto_sum, kernel, iterations=1)

        # Edge Y
        sobelY = sobel(proto_sum, axis=0)
        # Edge X
        sobelX = sobel(proto_sum, axis=1)
        # Edge magnitude
        sobelS = np.sqrt(sobelX ** 2 + sobelY ** 2)
        # Edge angle
        sobelA = np.arctan2(sobelY, sobelX)
        # How much is the edge angle above the X axis
        sobelA = np.minimum(sobelA, np.pi - sobelA) 

        print("Mask Sobel", sobelS.shape, round(sobelS.min(), 3), round(sobelS.max(), 3))

        # Find curbs
        sobelS = np.clip(sobelS / 11, 0, 1) 
        curb_mask = np.where(
            (sobelS > 0.4) & # Needs to be a steep edge to count as curb (filter noise, basically)
            (sobelA < radians(-10)) & # Edge gradient needs to point down
            curb_mask_dilate &  # Needs to be near a curb, as detected by the depth map
            (proto_sum_dilate > 0), # Needs to be on the underside of a YOLO obstacle mask. TUNING: this number is can be much higher (like 10) for thinner curbs
            sobelS, 
            0)

        # Following is display code, not related to algorithm, tweak to look nice
        mask_alpha = 0.6

        # This is for displaying the green YOLO masks
        protoDisp = sigmoid(proto_sum)
        # Filter out low confidence, to make the visualization less "fuzzy" 
        protoDisp[protoDisp < conf_thresh] = 0
        # Make it green
        protoDispGreen = np.dstack((0 * protoDisp, 0.5 * protoDisp, 0 * protoDisp))
        print("maskdisp shape", protoDispGreen.shape)
        print("protoDisp shape", protoDisp.shape)

        # Combine the green YOLO masks with the curb mask
        protoDispGreen[..., 2] = np.maximum(protoDispGreen[..., 2], curb_mask)
        protoDispGreen[..., 0] = np.maximum(protoDispGreen[..., 0], 0.2 * curb_mask)
        overlayDisp = cv2.resize(protoDispGreen, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Combine the masks with the RGB image
        frame = cv2.addWeighted(
            (overlayDisp * 255).astype(np.uint8), 
            mask_alpha, frame, 1 - mask_alpha, 0)
        
        cv2.imshow("Output", frame)
        # cv2.imshow("proto_sum", proto_sum_dilate.astype(np.uint8) * 255)

        # Scale the depth frame from 0 - 2000mm to a range of 0 - 255
        # Then run it through a colormap to make it look nice
        depthFrameScaled = (np.clip(depth_frame / 2000, 0, 1) * 255).astype(np.uint8)
        depthFrameDisp = cv2.applyColorMap(depthFrameScaled, cv2.COLORMAP_JET)
        cv2.imshow('Depth', depthFrameDisp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show_protos", action="store_true", help="Show prototypes")
    args = parser.parse_args()

    main(args.show_protos)
