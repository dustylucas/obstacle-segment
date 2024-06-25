
import numpy as np
import depthai as dai
from sklearn import linear_model

from ransac import Plane

def create_xyz(width, height, camera_matrix):
    xs = np.linspace(0, width - 1, width, dtype=np.float32)
    ys = np.linspace(0, height - 1, height, dtype=np.float32)

    # generate grid by stacking coordinates
    base_grid = np.stack(np.meshgrid(xs, ys)) # WxHx2
    points_2d = base_grid.transpose(1, 2, 0) # 1xHxWx2

    # unpack coordinates
    u_coord: np.array = points_2d[..., 0]
    v_coord: np.array = points_2d[..., 1]

    # unpack intrinsics
    fx: np.array = camera_matrix[0, 0]
    fy: np.array = camera_matrix[1, 1]
    cx: np.array = camera_matrix[0, 2]
    cy: np.array = camera_matrix[1, 2]

    # projective
    x_coord: np.array = (u_coord - cx) / fx
    y_coord: np.array = (v_coord - cy) / fy

    xyz = np.stack([x_coord, y_coord], axis=-1)
    return np.pad(xyz, ((0,0),(0,0),(0,1)), "constant", constant_values=1.0)

def create_matrix(device, resolution):
    calibData = device.readCalibration()
    global matrix
    xyz = calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A,
        dai.Size2f(resolution[0], resolution[1]),
    )
    xyz = np.array(xyz)
    print("Camera matrix: ", xyz)
    matrix = create_xyz(resolution[1], resolution[0], xyz)
    
def depth_to_3d(depth):
    depth[depth > 2000] = 0

    depth = np.expand_dims(depth, axis=0)  # 1x1xHxW
    depth = np.expand_dims(depth, axis=0)  # 1x1x1xHxW

    # depth should come in Bx1xHxW
    points_depth = np.transpose(depth, (0, 2, 3, 1))  # 1xHxWx1
    points_3d = matrix * points_depth
    points_3d = np.transpose(points_3d, (0, 3, 1, 2))  # Bx3xHxW
    points_3d = points_3d.reshape(3, -1).T.astype(np.float64) / 1000.0

    # Rotate it almost there
    # radians = np.deg2rad(-30)
    # # Rotation matrix around the X-axis
    # rotation_matrix = np.array([
    #     [1, 0, 0],
    #     [0, np.cos(radians), -np.sin(radians)],
    #     [0, np.sin(radians), np.cos(radians)]
    # ])
    # rotated_point_cloud = np.dot(points_3d, rotation_matrix)
    # return rotated_point_cloud
    return points_3d

import open3d as o3d
from functools import partial

class PointCloudVisualizer():
    def __init__(self):
        self.pcl = None
        # transform from camera to world orientation (Note the absolute position won't be correct)
        self.R_camera_to_world = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).astype(np.float64)
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("[DepthAI] Open3D integration demo", 960, 540)
        self.isstarted = False

        # view_control = self.vis.get_view_control()
        # view_control.set_front([0, 0, -100])  # Adjust as necessary
        # view_control.set_lookat([0, 0, 0])  # Center of the point cloud
        # view_control.set_up([0, -1, 0])  # Up direction
        # view_control.set_zoom(0.1)  # Smaller values zoom out

    def visualize_pcl(self, pcl_data, downsample=False):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcl_data)
        pcd.remove_non_finite_points()
        if downsample:
            pcd = pcd.voxel_down_sample(voxel_size=0.001)
        # Remove noise
        # pcd = pcd.remove_statistical_outlier(30, 0.1)[0]
        if self.pcl is None:
            self.pcl = pcd
        else:
            self.pcl.points = pcd.points
        # Rotate the pointcloud such that it is in the world coordinate frame (easier to visualize)
        self.pcl.rotate(self.R_camera_to_world, center=np.array([0,0,0],dtype=np.float64))

        if not self.isstarted:
            self.vis.add_geometry(self.pcl)
            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            self.vis.add_geometry(origin)
            ctr = self.vis.get_view_control()
            # ctr.set_zoom(0.5)
            # ctr.camera_local_rotate()
            # ctr.camera_local_translate()
            self.isstarted = True
        else:
            self.vis.update_geometry(self.pcl)
            self.vis.poll_events()
            self.vis.update_renderer()

    def close_window(self):
        self.vis.destroy_window()


def ransac_indices(xyz, threshold=0.01):
    assert xyz.shape[1] == 3
    shape = (480, 640)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.remove_non_finite_points()
    pcd, trace_indices, _ = pcd.voxel_down_sample_and_trace(0.005, pcd.get_min_bound(), pcd.get_max_bound())
    downsampled_xyz = np.asarray(pcd.points)

    plane1 = Plane()
    
    if len(downsampled_xyz) < 3:
        return np.array([]), np.zeros(shape, dtype=np.uint8), np.zeros(shape, dtype=np.uint8), np.zeros(shape, dtype=np.uint8)
    
    best_eq, best_inliers, bottom_liers, top_liers, distance_map = plane1.fit(downsampled_xyz, thresh=0.01, obs_thresh=0.015) # 0.2 cm
    best_inliers = np.asarray(best_inliers, dtype=int)

    def indices_to_indices(idx):
        original_indices = []
        for idx in idx:
            original_indices.extend(trace_indices[idx])
        return np.asarray(original_indices)

    best_inliers = indices_to_indices(best_inliers)
    bottom_liers = indices_to_indices(bottom_liers)
    distance_map = indices_to_indices(distance_map)
    # print("downsampled_xyz shape", downsampled_xyz.shape)
    # print("trace_indices shape", trace_indices.shape)

    mask = np.zeros(shape, dtype=bool)
    rows = np.array(best_inliers) // shape[1]
    cols = np.array(best_inliers) % shape[1]
    if len(rows) > 0:
        mask[rows, cols] = True

    obs_mask = np.zeros(shape, dtype=bool)
    rows = np.array(bottom_liers) // shape[1]
    cols = np.array(bottom_liers) % shape[1]
    if len(rows) > 0:
        obs_mask[rows, cols] = True

    dist_mask = np.zeros(shape, dtype=bool)
    rows = np.array(distance_map) // shape[1]
    cols = np.array(distance_map) % shape[1]
    if len(rows) > 0:
        dist_mask[rows, cols] = True

    return xyz[best_inliers], mask.astype(np.uint8), obs_mask.astype(np.uint8), dist_mask.astype(np.uint8)
