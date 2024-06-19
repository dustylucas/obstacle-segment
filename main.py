import json
from math import radians
import time

import cv2
import depthai as dai
import numpy as np

from pc import PointCloudVisualizer, create_matrix, depth_to_3d, ransac_indices
from yolo_api import Segment
from scipy.ndimage import sobel


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

with open("helpers/config.json", "r") as config:
    model_data = json.load(config)

preview_img_width = model_data["input_width"]
preview_img_height = model_data["input_height"]
input_shape = [1, 3, preview_img_height, preview_img_width]

output0_shape = model_data["shapes"]["output0"]
output1_shape = model_data["shapes"]["output1"]


dot = np.zeros((output1_shape[2], output1_shape[3]), dtype=float)
center = (output1_shape[3] // 2, int(output1_shape[2] * 0.75)) # output1_shape[2] // 2
dot = cv2.circle(dot, center, output1_shape[3] // 16, 1, -1)
dot = dot > 0

path_to_yolo_blob = "models/yolov8n-seg.blob"


def main():
    pipeline = dai.Pipeline()

    # Init pipeline's output queue
    # xout_rgb = pipeline.createXLinkOut()
    # xout_yolo_nn = pipeline.createXLinkOut()
    # xout_depth = pipeline.createXLinkOut()
    sync = pipeline.create(dai.node.Sync)
    xoutGrp = pipeline.create(dai.node.XLinkOut)
    xoutGrp.setStreamName("xout")
    # xout_rgb.setStreamName("rgb")
    # xout_yolo_nn.setStreamName("yolo_nn")
    # xout_depth.setStreamName("depth")


    # Neural network pipeline properties
    nn = pipeline.createNeuralNetwork()
    nn.setBlobPath(path_to_yolo_blob)
    nn.out.link(sync.inputs["xout_yolo_nn"])

    # Color cam properties (Cam_A/RGB)
    cam_rgb = pipeline.createColorCamera()
    cam_rgb.setPreviewSize(preview_img_width, preview_img_height)
    cam_rgb.setInterleaved(False)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
    cam_rgb.preview.link(nn.input)
    cam_rgb.preview.link(sync.inputs["xout_rgb"])
    print("Color cam resolution: ", cam_rgb.getResolutionSize())

    # Stereo/Depth node properties
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A) # Align to RGB
    stereo.setOutputSize(preview_img_width, preview_img_height)
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
    
    device_info = dai.DeviceInfo("10.25.76.102")
    device = dai.Device(pipeline, device_info)

    calib_data = device.readCalibration()
    camera_intrinsics_matrix = np.array(calib_data.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, ))
    print(camera_intrinsics_matrix)

    device.setIrLaserDotProjectorBrightness(200) # 0 to 1200

    create_matrix(device, (preview_img_height, preview_img_width))

    # rgb_queue = device.getOutputQueue("rgb", maxSize=2, blocking=False)
    # yolo_nn_queue = device.getOutputQueue("yolo_nn", maxSize=2, blocking=False)
    # depth_queue = device.getOutputQueue("depth", maxSize=2, blocking=False)
    sync_queue = device.getOutputQueue("xout", 5, False)

    pcl_converter = PointCloudVisualizer()
    
    while True:
        if cv2.waitKey(1) == ord("q"):
            break

        # rgb_queue_msg = rgb_queue.get()
        # yolo_nn_queue_msg = yolo_nn_queue.get()
        # depth_queue_msg = depth_queue.get()

        # if rgb_queue_msg is None:
        #     print("rgb_queue message is empty")
        #     continue
        
        # depth_frame = (
        #     depth_queue_msg.getFrame() if depth_queue_msg is not None else None
        # )  # depth frame returns values in mm (millimeter)

        # if yolo_nn_queue_msg is None or depth_frame is None:
        #     continue

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

        output0 = np.reshape(
            yolo_nn_queue_msg.getLayerFp16("output0"),
            newshape=(output0_shape),
        )

        output1 = np.reshape(
            yolo_nn_queue_msg.getLayerFp16("output1"),
            newshape=(output1_shape),
        )

        rows = [
            np.concatenate((output1[0, 0:8]), axis=1),
            np.concatenate((output1[0, 8:16]), axis=1),
            np.concatenate((output1[0, 16:24]), axis=1),
            np.concatenate((output1[0, 24:32]), axis=1),
        ]

        bigone = np.concatenate(rows, axis=0)

        cv2.imshow("All Prototypes", bigone)

        if len(output0) == 0 or len(output1) == 0:
            continue

        # Begin!

        print('Depth max', depth_frame.max(), depth_frame.dtype)

        # Depth rendering and RANSAC
        points = depth_to_3d(depth_frame)
        print('3D points shape', points.shape)
        
        ground_points, floor_mask, obs_mask, curb_mask = ransac_indices(points)

        # pcl_converter.visualize_pcl(ground_points, downsample=False)

        floor_mask = cv2.resize(floor_mask, (output1.shape[3], output1.shape[2]), interpolation=cv2.INTER_NEAREST)
        obs_mask = cv2.resize(obs_mask, (output1.shape[3], output1.shape[2]), interpolation=cv2.INTER_NEAREST)

        R = 15
        kernel = np.zeros((2 * R + 1, 2 * R + 1), np.uint8)
        cv2.circle(kernel, (R, R), R, 1, -1)
        kernel[R:(2*R+1), :] = 0
        curb_mask_dilate = cv2.dilate(curb_mask, kernel, iterations=1)
        curb_mask_dilate = cv2.resize(curb_mask_dilate, (output1.shape[3], output1.shape[2]), interpolation=cv2.INTER_AREA)


        cv2.imshow('Grounds/Obs', np.concatenate((floor_mask * 255,
                                                   obs_mask * 255,
                                                   curb_mask_dilate * 255), axis=1))

        depthFrameScaled = (np.clip(depth_frame / 10000, 0, 1) * 255).astype(np.uint8)
        depthFrameDisp = cv2.applyColorMap(depthFrameScaled, cv2.COLORMAP_JET)
        # cv2.imshow('Grounds', depthFrameDisp)
          
        # Segment
        # yoloseg = Segment(
        #     input_shape=input_shape,
        #     input_height=preview_img_height,
        #     input_width=preview_img_width,
        #     conf_thres=0.2,
        #     iou_thres=0.5,
        # )

        # yoloseg.prepare_input_for_oakd(frame.shape[:2])

        # boxes, scores, class_ids, mask_maps = (
        #     yoloseg.segment_objects_from_oakd(output0, output1)
        # )

        # frame = yoloseg.draw_masks(frame.copy())
        mask_alpha = 0.6

        print("Prototype shape ", output1.shape)
        # print("Coefficient shape ", yoloseg.mask_pred.shape)

        # Optimized protos
        dot = floor_mask > 0
        not_dot = obs_mask > 0
        proto_sum = np.zeros(output1.shape[2:4], float)
        # Find the mask coefficients with highest adherence to dot
        for i, proto in enumerate(output1[0]):

            # if i in [2, 8, 12, 14, 19, 20, 24, 26, 27, 29, 31]:
            #     continue

            similarity = proto[dot].sum() / dot.sum()
            inv_similarity = proto[not_dot].sum() / (not_dot).sum()
            mean = (similarity + inv_similarity) / 2
            var = inv_similarity - mean # var = similarity - mean
            scale = var

            proto_sum += (proto - mean) * scale
        
        # Manual protos (Good for objects)
        proto_sum = output1[0, 1] + output1[0, 2] + output1[0, 4] + output1[0, 6] + -1 * output1[0, 16] + -1 * output1[0, 21]

        kernel = np.ones((5,5),np.uint8)
        proto_sum_dilate = cv2.dilate(proto_sum, kernel, iterations=1)

        freespace_mask = sigmoid(proto_sum * 3)

        # Edge
        sobelY = sobel(proto_sum, axis=0)
        sobelX = sobel(proto_sum, axis=1)
        sobelS = np.sqrt(sobelX ** 2 + sobelY ** 2)
        sobelA = np.arctan2(sobelY, sobelX)
        sobelA = np.minimum(sobelA, np.pi - sobelA) # Angle distance above X axis

        print("Mask Sobel", sobelS.shape, round(sobelS.min(), 3), round(sobelS.max(), 3))

        # Get bottoms of mask
        sobelS = np.clip(sobelS / 11, 0, 1) # N
        mask = np.where((sobelA < radians(-10)) & (proto_sum_dilate > 0.5) & curb_mask_dilate, sobelS, 0)

        # Make sure edges do not exceed lowest contour
        # y_coords = np.arange(mask.shape[0])[:, np.newaxis]  # Hx1
        # grid = np.tile(y_coords, (1, mask.shape[1]))
        # print("Meshgrid", grid.shape)
        # grid[mask == 0] = 0

        # lowest_indices = np.argmax(grid, axis=0)
        # lowest_acceptable = lowest_indices - output1.shape[2] // 8
        # # Set values of mask with y > lowest_acceptable to 0
        # row_indices = np.arange(mask.shape[0])[:, np.newaxis] # Hx1
        # mask[row_indices < lowest_acceptable] = 0

        rgb = [0.3, 0.0, 1.0]
        maskDisp = np.dstack((rgb[0] * mask + 0.2 * freespace_mask, rgb[1] * mask + 0.5 * freespace_mask, rgb[2] * mask))
        # maskDisp[..., 2][dot] = 0.99
        maskDispUpscaled = cv2.resize(maskDisp, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        cv2.imshow("Mask", (maskDispUpscaled * 255).astype(np.uint8))

        print("MaskDisp", maskDisp.shape, round(maskDisp.min(), 3), round(maskDisp.max(), 3))
        print("Frame", frame.shape, frame.min(), frame.max(), frame.dtype)

        frame = cv2.addWeighted(
            (maskDispUpscaled * 255).astype(np.uint8), 
            mask_alpha, frame, 1 - mask_alpha, 0)
        
        cv2.imshow("Output", frame)

        

        # for i in range(len(boxes)):
        #     mask = mask_maps[i].astype(np.uint8)
        #     contours, _ = cv2.findContours(
        #         mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        #     )
        #     for contour in contours:
        #         center, size, angle = cv2.minAreaRect(contour)
        #         box_points = cv2.boxPoints((center, size, angle))
        #         box_points = np.int32(box_points)
        #         cv2.drawContours(frame, [box_points], 0, (0, 255, 0), 2)

        #         # print("RotatedRect: ", center, size, angle)
        #         # print("box points: ", box_points)

        #         mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        #         cv2.drawContours(mask, [box_points], -1, 255, -1)

        #         masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        #         cv2.imshow("Masked Output", masked_frame)

        #         masked_depth_frame = cv2.bitwise_and(
        #             depth_frame, depth_frame, mask=mask
        #         )
        #         vals_inside_rotated_rect = masked_depth_frame[mask != 0]
        #         depth = np.median(vals_inside_rotated_rect)

        #         # print(vals_inside_rotated_rect)
        #         # print(len(vals_inside_rotated_rect))

        #         # depth_frame_roi = depth_frame[
        #         #     int(boxes[i][1]) : int(boxes[i][3]),
        #         #     int(boxes[i][0]) : int(boxes[i][2]),
        #         # ]
        #         # depth = np.median(depth_frame_roi.flatten())
        #         # print(len(depth_frame_roi.flatten()))
        #         # print(len(depth_frame.flatten()))

        #         cv2.putText(
        #             frame,
        #             f"Z: {int(depth)} mm",
        #             (int(center[0]) - 50, int(center[1]) - 25),
        #             cv2.FONT_HERSHEY_TRIPLEX,
        #             0.5,
        #             (255, 255, 255),
        #         )
        #         cv2.putText(
        #             frame,
        #             f"Angle: {round(angle, 1)}",
        #             (int(center[0]) - 50, int(center[1]) + 25),
        #             cv2.FONT_HERSHEY_TRIPLEX,
        #             0.5,
        #             (255, 255, 255),
        #         )

if __name__ == "__main__":
    main()
