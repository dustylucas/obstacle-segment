import json
from math import radians

import cv2
import depthai as dai
import numpy as np

from pc import PointCloudVisualizer, create_projection_matrix, depth_to_3d, depth_to_3d_with_value, dist_to_ground_plane, points_to_image_torch
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
    # nn = pipeline.createNeuralNetwork()
    # nn.setBlobPath(path_to_yolo_blob)
    # nn.out.link(sync.inputs["xout_yolo_nn"])

    # Color cam properties (Cam_A/RGB)
    cam_rgb = pipeline.createColorCamera()
    cam_rgb.setPreviewSize(model_img_width, model_img_height)
    cam_rgb.setInterleaved(False)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
    # cam_rgb.preview.link(nn.input)
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
    print("Projection matrix: ", projection_matrix.shape)

    device.setIrLaserDotProjectorBrightness(500) # 0 to 1200

    # Get output queue
    sync_queue = device.getOutputQueue("xout", 1, True)
    
    frames = []
    depth_frames = []
    yolo_frames = []
    
    # Uncomment to use
    # pcl_converter = PointCloudVisualizer()
    
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

        if frame is None or depth_frame is None:
            continue
        
        # output0 = np.reshape(
        #     yolo_nn_queue_msg.getLayerFp16("output0"),
        #     newshape=(model_coefficient_shape),
        # )

        # Exract proto masks from NN 
        # output1 = np.reshape(
        #     yolo_nn_queue_msg.getLayerFp16("output1"),
        #     newshape=(model_proto_mask_shape),
        # )

        # if len(output1) == 0:
        #     continue

        frames.append(frame)
        depth_frames.append(depth_frame)
        # yolo_frames.append(output1)

        # if show_protos:
        #     rows = [
        #         np.concatenate((output1[0, 0:8]), axis=1),
        #         np.concatenate((output1[0, 8:16]), axis=1),
        #         np.concatenate((output1[0, 16:24]), axis=1),
        #         np.concatenate((output1[0, 24:32]), axis=1),
        #     ]

        #     bigone = np.concatenate(rows, axis=0)

        #     cv2.imshow("All Prototypes", bigone)

        # # cv2.imshow("Color Input", frame)

        # print()

        # # Get pointcloud (Also convert mm -> m)
        # points = depth_to_3d(depth_frame / 1000.0, projection_matrix)
        # print('3D points shape', points.shape)
        
        # # Get image-space masks for ground, obstacles, and curbs
        # ground_plane, ground_points, floor_mask, obs_mask, curb_mask = ransac_indices(points)

        # # Visualize pointcloud
        # # pcl_converter.visualize_pcl(points, downsample=False)

        # # Dilate curb mask
        # R = 35
        # kernel = np.zeros((2 * R + 1, 2 * R + 1), np.uint8)
        # cv2.circle(kernel, (R, R), R, 1, -1)
        # kernel[R:(2*R+1), :] = 0
        # curb_mask_dilate = cv2.dilate(curb_mask, kernel, iterations=1)

        # # cv2.imshow('Grounds/Obs/Curbs', np.concatenate((floor_mask * 255,
        # #                                            obs_mask * 255,
        # #                                            curb_mask * 255), axis=1))
        
        # floor_mask = cv2.resize(floor_mask, (output1.shape[3], output1.shape[2]), interpolation=cv2.INTER_AREA)
        # obs_mask = cv2.resize(obs_mask, (output1.shape[3], output1.shape[2]), interpolation=cv2.INTER_AREA)
        # curb_mask_dilate = cv2.resize(curb_mask_dilate, (output1.shape[3], output1.shape[2]), interpolation=cv2.INTER_AREA)

        # print("Prototype shape ", output1.shape)

        # # Optimized protos

        # proto_sum = np.zeros(output1.shape[2:4], float)

        # # Find the mask coefficients with highest adherence to dot
        # for i, proto in enumerate(output1[0]):
            
        #     # Some manually picked protos that only activate for partial regions
        #     # (left side, bottom edge, etc.) which have been excluded to encourage
        #     # final masks to be more global
        #     # if i in [2, 8, 12, 14, 19, 20, 24, 26, 27, 29, 31]:
        #         # continue
            
        #     unknown = (obs_mask == 0) & (floor_mask == 0) 

        #     floor_similarity = (proto * floor_mask).sum() / floor_mask.sum()
        #     obs_similarity = 2 * (proto * obs_mask).sum() / obs_mask.sum()
        #     mean = (floor_similarity + obs_similarity) / 2
        #     coeff = obs_similarity - mean 

        #     proto_sum += coeff * (proto - mean)
        
        # # Manual protos (Good for objects)
        # # proto_sum = output1[0, 1] + output1[0, 2] + output1[0, 4] + output1[0, 6] + -1 * output1[0, 16] + -1 * output1[0, 21]

        # conf_thresh = 0.3

        # # kernel = np.ones((5,5),np.uint8)
        # # proto_sum_dilate = proto_sum.copy()
        # # proto_sum_dilate[proto_sum < conf_thresh] = 0
        # # proto_sum_dilate = cv2.dilate(proto_sum, kernel, iterations=1)

        # # # Edge
        # # sobelY = sobel(proto_sum, axis=0)
        # # sobelX = sobel(proto_sum, axis=1)
        # # sobelS = np.sqrt(sobelX ** 2 + sobelY ** 2)
        # # sobelA = np.arctan2(sobelY, sobelX)
        # # sobelA = np.minimum(sobelA, np.pi - sobelA) # Angle distance above X axis

        # # print("Mask Sobel", sobelS.shape, round(sobelS.min(), 3), round(sobelS.max(), 3))

        # # # Get bottoms of mask
        # # sobelS = np.clip(sobelS / 11, 0, 1) 
        # # mask = np.where((sobelS > 0.4) & (sobelA < radians(-10)) & curb_mask_dilate & (proto_sum_dilate > 0), sobelS, 0)

        # mask_alpha = 0.6

        # proto_sum = sigmoid(proto_sum)
        # proto_sum[proto_sum < conf_thresh] = 0

        # # mask_upscaled = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # # # Display curb topdown
        # # depth_to_ground_plane = -1.0 * dist_to_ground_plane(projection_matrix, ground_plane)
        # # print("Ground plane depth min max:", depth_to_ground_plane.min(), depth_to_ground_plane.max(), depth_to_ground_plane.dtype)

        # # curb_pointcloud = depth_to_3d_with_value(depth_to_ground_plane, projection_matrix, mask_upscaled)
        # # print("Curb pointcloud min max:", curb_pointcloud.min(), curb_pointcloud.max())
        # # # Gather curb points into a top down view of z-coordinate
        # # bev = points_to_image_torch(curb_pointcloud[:, 0], curb_pointcloud[:, 1], curb_pointcloud[:, 3], -0.5, 0.5, -1, 0.2, 600)
        # # # Light gaussian blue to smooth out upscaling artifacts
        # # bev = cv2.GaussianBlur(bev, (11, 11), 0)
        # # # Make bev colored,
        # # bev = np.dstack((0 * bev, 0.2 * bev, 1.0 * bev))
        # # cv2.imshow("CurbTopDown", bev * 1.5)

        # maskDisp = np.dstack((0 * proto_sum, 0.5 * proto_sum, 0 * proto_sum))
        # print("maskdisp shape", maskDisp.shape)

        # maskDispUpscaled = cv2.resize(maskDisp, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)

        # frame = cv2.addWeighted(
        #     (maskDispUpscaled * 255).astype(np.uint8), 
        #     mask_alpha, frame, 1 - mask_alpha, 0)
        
        # cv2.imshow("Output", frame)
        # cv2.imshow("proto_sum", proto_sum_dilate.astype(np.uint8) * 255)

        depthFrameScaled = (np.clip(depth_frame / 2000, 0, 1) * 255).astype(np.uint8)
        depthFrameDisp = cv2.applyColorMap(depthFrameScaled, cv2.COLORMAP_JET)
        cv2.imshow('Depth', depthFrameDisp)
        cv2.imshow('Color', frame)

    # Save frames
    np.savez_compressed("frames.npz", frames=frames, depth_frames=depth_frames, yolo_frames=yolo_frames)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show_protos", action="store_true", help="Show prototypes")
    args = parser.parse_args()

    main(args.show_protos)
