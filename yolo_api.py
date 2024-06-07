import math

import cv2
import numpy as np

from helpers.utils import draw_detections, nms, sigmoid, xywh2xyxy


class Segment:
    def __init__(
        self,
        input_shape=[1, 3, 480, 640],
        input_height=480,
        input_width=640,
        conf_thres=0.7,
        iou_thres=0.5,
        num_masks=32,
    ):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.num_masks = num_masks

        self.input_names = "images"
        self.input_shape = input_shape
        self.input_height = input_height
        self.input_width = input_width
        self.output_names = ["output0", "output1"]

    def segment_objects_from_oakd(self, output0, output1):

        self.boxes, self.scores, self.class_ids, self.mask_pred = self.process_box_output(
            output0
        )

        self.mask_maps = self.process_mask_output(self.mask_pred, output1)

        return self.boxes, self.scores, self.class_ids, self.mask_maps

    def prepare_input_for_oakd(self, shape):

        self.img_height = shape[0]
        self.img_width = shape[1]

    def process_box_output(self, box_output):

        predictions = np.squeeze(box_output).T
        num_classes = box_output.shape[1] - self.num_masks - 4

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4 : 4 + num_classes], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], [], np.array([])

        box_predictions = predictions[..., : num_classes + 4]
        mask_predictions = predictions[..., num_classes + 4 :]

        # Get the class with the highest confidence
        class_ids = np.argmax(box_predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(box_predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = nms(boxes, scores, self.iou_threshold)

        return (
            boxes[indices],
            scores[indices],
            class_ids[indices],
            mask_predictions[indices],
        )

    def process_mask_output(self, mask_predictions, mask_output):
        print('In process_mask_output (prototypes, coeffs --> binary mask)')
        if mask_predictions.shape[0] == 0:
            return []

        mask_output = np.squeeze(mask_output)

        # Calculate the mask maps for each box
        num_mask, mask_height, mask_width = mask_output.shape  # CHW
        masks = sigmoid(mask_predictions @ mask_output.reshape((num_mask, -1)))
        masks = masks.reshape((-1, mask_height, mask_width))

        # Downscale the boxes to match the mask size
        scale_boxes = self.rescale_boxes(
            self.boxes, (self.img_height, self.img_width), (mask_height, mask_width)
        )

        # For every box/mask pair, get the mask map
        mask_maps = np.zeros((len(scale_boxes), self.img_height, self.img_width))
        blur_size = (
            int(self.img_width / mask_width),
            int(self.img_height / mask_height),
        )
        for i in range(len(scale_boxes)):

            scale_x1 = int(math.floor(scale_boxes[i][0]))
            scale_y1 = int(math.floor(scale_boxes[i][1]))
            scale_x2 = int(math.ceil(scale_boxes[i][2]))
            scale_y2 = int(math.ceil(scale_boxes[i][3]))

            x1 = int(math.floor(self.boxes[i][0]))
            y1 = int(math.floor(self.boxes[i][1]))
            x2 = int(math.ceil(self.boxes[i][2]))
            y2 = int(math.ceil(self.boxes[i][3]))

            scale_crop_mask = masks[i][scale_y1:scale_y2, scale_x1:scale_x2]
            crop_mask = cv2.resize(
                scale_crop_mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC
            )

            crop_mask = cv2.blur(crop_mask, blur_size)

            crop_mask = (crop_mask > 0.5).astype(np.uint8)
            mask_maps[i, y1:y2, x1:x2] = crop_mask

        return mask_maps

    def extract_boxes(self, box_predictions):
        # Extract boxes from predictions
        boxes = box_predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(
            boxes,
            (self.input_height, self.input_width),
            (self.img_height, self.img_width),
        )

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)
        # print(boxes)

        # Check the boxes are within the image
        boxes[:, 0] = np.clip(boxes[:, 0], 0, self.img_width)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, self.img_height)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, self.img_width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, self.img_height)

        return boxes

    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):
        return draw_detections(
            image, self.boxes, self.scores, self.class_ids, mask_alpha
        )

    def draw_masks(self, image, draw_scores=True, mask_alpha=0.5):
        return draw_detections(
            image,
            self.boxes,
            self.scores,
            self.class_ids,
            mask_alpha,
            mask_maps=self.mask_maps,
        )

    @staticmethod
    def rescale_boxes(boxes, input_shape, image_shape):
        # Rescale boxes to original image dimensions
        input_shape = np.array(
            [input_shape[1], input_shape[0], input_shape[1], input_shape[0]]
        )
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array(
            [image_shape[1], image_shape[0], image_shape[1], image_shape[0]]
        )

        return boxes
