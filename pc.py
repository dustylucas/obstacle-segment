
import numpy as np
import depthai as dai
import torch

def create_xyz(width, height, camera_matrix):
    '''
    Math for creating projection matrix
    '''
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

def create_projection_matrix(device, resolution):
    '''
    Create 4x4 projection matrix for converting depth map to 3D coordinates
    '''
    calibData = device.readCalibration()

    xyz = calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A,
        dai.Size2f(resolution[0], resolution[1]),
    )

    xyz = np.array(xyz)
    matrix = create_xyz(resolution[1], resolution[0], xyz)
    return matrix

def depth_to_3d(depth, matrix):
    '''
    Converts depth map (HxW) to 3D coordinates (Nx3) using pointwise projection matrix
    Clips depth values above 2, 
    uses meters
    '''
    depth[depth > 2] = 0

    depth = np.expand_dims(depth, axis=0)  # 1x1xHxW
    depth = np.expand_dims(depth, axis=0)  # 1x1x1xHxW

    # depth should come in Bx1xHxW
    points_depth = np.transpose(depth, (0, 2, 3, 1))  # 1xHxWx1
    points_3d = matrix * points_depth
    points_3d = np.transpose(points_3d, (0, 3, 1, 2))  # Bx3xHxW
    points_3d = points_3d.reshape(3, -1).T.astype(np.float64)
    return points_3d

def depth_to_3d_with_value(depth, matrix, optional_value=None):
    '''
    Converts depth map (HxW) to 3D coordinates (Nx4 or Nx3) using pointwise projection matrix.
    Clips depth values above 2 meters.
    Optionally projects additional values along with depth coordinates.
    
    Args:
    - depth (np.array): Depth map of shape (H, W).
    - matrix (np.array): Projection matrix of shape (H, W, 3).
    - optional_value (np.array, optional): An optional matrix of values to project with depth. Shape (H, W).
    
    Returns:
    - np.array: Array of 3D points (with optional values) of shape (N, 4) or (N, 3) if no optional_value is provided.
    '''
    depth[depth > 2] = 0  # Clipping depth values above 2 meters.

    # Preparing depth matrix for broadcasting.
    depth = np.expand_dims(depth, axis=-1)  # HxWx1

    # Calculate the initial 3D coordinates.
    points_3d = matrix * depth  # Element-wise multiplication for broadcasting.
    points_3d = points_3d.reshape(-1, 3)  # Reshaping to a list of 3D points.

    if optional_value is not None:
        # Ensuring optional_value is correctly shaped and appending it to the 3D points.
        optional_value = np.expand_dims(optional_value, axis=-1)
        optional_value = optional_value.reshape(-1, 1)  # Flatten to a single column.
        points_3d = np.hstack((points_3d, optional_value))  # Concatenating the optional values to the 3D coordinates.

    return points_3d.astype(np.float64)  # Ensuring the datatype is consistent.


def dist_to_ground_plane(matrix, plane):
    '''
    Given projection matrix, and
    plane = (a, b, c, d),
    solve the distance from the camera to the plane at each pixel
    Output: ndarray of HxW
    '''

    A, B, C, D = plane

    return D / (matrix * np.array([A, B, C])).sum(axis=2)

def points_to_image_torch(xs, ys, ps, minx, maxx, miny, maxy, px_per_meter):
    xt, yt, pt = torch.from_numpy(xs), torch.from_numpy(ys), torch.from_numpy(ps)

    width = int((maxx - minx) * px_per_meter)
    height = int((maxy - miny) * px_per_meter)
    img = torch.zeros((height, width), dtype=torch.float64)

    yt = ((yt - miny) * px_per_meter).int()
    xt = ((xt - minx) * px_per_meter).int()

    # if xt or yt is out of bounds, ignore
    mask = (yt >= 0) & (yt < height) & (xt >= 0) & (xt < width)
    yt, xt, pt = yt[mask], xt[mask], pt[mask]

    img.index_put_((yt, xt), pt, accumulate=False)
    return img.numpy()

import open3d as o3d

class PointCloudVisualizer():
    '''
    Quick and dirty visualizer for point clouds. Call visualize_pcl() to update
    '''
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


