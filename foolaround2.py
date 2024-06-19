import cv2
import numpy as np


#Read input image, and create output image
image = cv2.imread('image2.png')

# Reshape image to 1280x800
image = cv2.resize(image, (1280, 800))

# Camera intrinsic matrix
fx, fy = 882.5, 882.5  # example focal lengths
cx, cy = image.shape[1] / 2, image.shape[0] / 2  # center of the image
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]])

# Extrinsic parameters (assuming you know these)
R = np.array([[1, 0, 0],  # example rotation matrix
              [0, 1, 0],
              [0, 0, 1]])
t = np.array([0, 0, 500])  # example translation vector


# Constructing a basic homography matrix if camera is above the ground looking straight down
H = K @ np.hstack((R[:, :2], t.reshape(-1, 1)))

height, width = image.shape[:2]
warped_image = cv2.warpPerspective(image, H, (width, height))

cv2.imshow('Original Image', image)
cv2.imshow('Warped Image', warped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()