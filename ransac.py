import random

import numpy as np
import open3d as o3d


class Plane:
    """
    Implementation of planar RANSAC.

    Class for Plane object, which finds the equation of a infinite plane using RANSAC algorithim.

    Call `fit(.)` to randomly take 3 points of pointcloud to verify inliers based on a threshold.

    ![Plane](https://raw.githubusercontent.com/leomariga/pyRANSAC-3D/master/doc/plano.gif "Plane")

    ---
    """

    def __init__(self):
        self.inliers = []
        self.equation = []

    def fit(self, pts, thresh=0.05, maxIteration=1000, obs_thresh=0.1):
        """
        Find the best equation for a plane.

        :param pts: 3D point cloud as a `np.array (N,3)`.
        :param thresh: Threshold distance from the plane which is considered inlier.
        :param maxIteration: Number of maximum iteration which RANSAC will loop over.
        :returns:
        - `self.equation`:  Parameters of the plane using Ax+By+Cy+D `np.array (1, 4)`
        - `self.inliers`: points from the dataset considered inliers

        ---
        """
        n_points = pts.shape[0]
        best_eq = []
        best_inliers = []

        bottom_liers = []
        top_liers = []

        for it in range(maxIteration):

            # Samples 3 random points
            id_samples = random.sample(range(0, n_points), 3)
            pt_samples = pts[id_samples]

            # We have to find the plane equation described by those 3 points
            # We find first 2 vectors that are part of this plane
            # A = pt2 - pt1
            # B = pt3 - pt1

            vecA = pt_samples[1, :] - pt_samples[0, :]
            vecB = pt_samples[2, :] - pt_samples[0, :]

            # Now we compute the cross product of vecA and vecB to get vecC which is normal to the plane
            vecC = np.cross(vecA, vecB)

            # The plane equation will be vecC[0]*x + vecC[1]*y + vecC[0]*z = -k
            # We have to use a point to find k
            vecC = vecC / np.linalg.norm(vecC)
            k = -np.sum(np.multiply(vecC, pt_samples[1, :]))
            plane_eq = [vecC[0], vecC[1], vecC[2], k]

            # Keep plane upward
            if (vecC[2] < 0):
                plane_eq = [-vecC[0], -vecC[1], -vecC[2], -k]

            # Distance from a point to a plane
            # https://mathworld.wolfram.com/Point-PlaneDistance.html
            pt_id_inliers = []  # list of inliers ids
            dist_pt = (
                plane_eq[0] * pts[:, 0] + plane_eq[1] * pts[:, 1] + plane_eq[2] * pts[:, 2] + plane_eq[3]
            ) / np.sqrt(plane_eq[0] ** 2 + plane_eq[1] ** 2 + plane_eq[2] ** 2)

            # Select indexes where distance is bigger than the threshold
            pt_id_inliers = np.where(np.abs(dist_pt) <= thresh)[0]
            if len(pt_id_inliers) > len(best_inliers):
                best_eq = plane_eq
                best_inliers = pt_id_inliers
                bottom_liers = np.where(dist_pt < -obs_thresh)[0]
                top_liers = np.where(dist_pt > obs_thresh)[0]
                curbs = np.where((dist_pt < -0.01) & (dist_pt > -obs_thresh * 4))[0]
            self.inliers = best_inliers
            self.equation = best_eq

        print("Plane equation: ", self.equation)

        return self.equation, self.inliers, bottom_liers, top_liers, curbs


def ransac_indices(xyz):
    '''
    Given a Nx3 pointcloud, run RANSAC on it to find the ground plane. Then, based on
    the distance of each from the ground plane, determine if it is floor (near to plane), 
    curb (slightly above plane), or obstacle (above plane). These categories are 
    reprojected back to image-space and returned as binary masks.
    
    Returns:
    - inliers_xyz: The array of points that are considered inliers to the best-fit plane.
    - floor_mask: A 480x640 binary image where floor points are set to 1.
    - obs_mask: A 480x640 binary image where obstacle points are set to 1.
    - curb_mask: A 480x640 binary image where curb points are set to 1.

    '''

    assert xyz.shape[1] == 3
    shape = (480, 640)

    # Downsample the pointcloud and remove non-finite points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.remove_non_finite_points()
    pcd, trace_indices, _ = pcd.voxel_down_sample_and_trace(0.005, pcd.get_min_bound(), pcd.get_max_bound())
    downsampled_xyz = np.asarray(pcd.points)

    plane1 = Plane()
    
    # If we don't have enough points for RANSAC, return empty arrays
    if len(downsampled_xyz) < 3:
        return np.array([]), np.zeros(shape, dtype=np.uint8), np.zeros(shape, dtype=np.uint8), np.zeros(shape, dtype=np.uint8)
    
    # Actually run RANSAC
    best_eq, best_inliers, bottom_liers, top_liers, distance_map = plane1.fit(downsampled_xyz, thresh=0.005, obs_thresh=0.015) # 0.2 cm
    best_inliers = np.asarray(best_inliers, dtype=int)

    def downsampled_indices_to_indices(idx):
        '''
        Convert indicies produced from voxel_down_sample_and_trace() to the original indices
        '''
        original_indices = []
        for idx in idx:
            original_indices.extend(trace_indices[idx])
        return np.asarray(original_indices)

    # Convert our downsampled indices to the original indices
    best_inliers = downsampled_indices_to_indices(best_inliers)
    bottom_liers = downsampled_indices_to_indices(bottom_liers)
    distance_map = downsampled_indices_to_indices(distance_map)

    # Convert floor indices to 2D indices, and fill floor_mask
    floor_mask = np.zeros(shape, dtype=bool)
    rows = np.array(best_inliers) // shape[1]
    cols = np.array(best_inliers) % shape[1]
    if len(rows) > 0:
        floor_mask[rows, cols] = True

    # Convert obstacle indices to 2D indices, and fill obs_mask
    obs_mask = np.zeros(shape, dtype=bool)
    rows = np.array(bottom_liers) // shape[1]
    cols = np.array(bottom_liers) % shape[1]
    if len(rows) > 0:
        obs_mask[rows, cols] = True

    # Convert curb indices to 2D indices, and fill curb_mask
    curb_mask = np.zeros(shape, dtype=bool)
    rows = np.array(distance_map) // shape[1]
    cols = np.array(distance_map) % shape[1]
    if len(rows) > 0:
        curb_mask[rows, cols] = True

    # Return the inlier points, the floor mask, obstacle mask, and curb mask (last three are HxW and correspond to camera image)
    return xyz[best_inliers], floor_mask.astype(np.uint8), obs_mask.astype(np.uint8), curb_mask.astype(np.uint8)
