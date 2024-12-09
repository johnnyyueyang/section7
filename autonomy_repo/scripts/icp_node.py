#!/usr/bin/env python3

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from sensor_msgs_py import point_cloud2
from tf2_ros import Buffer, TransformListener

from icp_utils import icp, open3d_icp # type: ignore

def get_pcd_array_from_point_cloud(pcd_msg: PointCloud2):
    pcd = point_cloud2.read_points_list(pcd_msg, field_names=["x", "y", "z"], skip_nans=True)
    return np.array(pcd)

def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    rotation_matrix = np.array([
        [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)]
    ])
    return rotation_matrix

class ICPNode(Node):
    def __init__(self):
        super().__init__('icp_node')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(0.2, self.get_tfs)

        self.prev_pcd = None
        self.pose = None
        self.lidar_pose = None
        self.transformation = np.eye(4)
        self.use_open3d = True
        
        # Task 2.3 — Subscribe to the point cloud.
        self.pcd_sub = self.create_subscription(
            PointCloud2,
            '/velodyne_points',
            self.pcd_callback,
            10
        )
        
        # Task 2.4 — Subscribe to /odom.
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        
        self.icp_poses = []
        self.odom_poses = []

    def get_tfs(self):
        try:
            init_transform = self.tf_buffer.lookup_transform(
                target_frame='odom',
                source_frame='base_footprint',
                time=Time()
            )
            
            translation = init_transform.transform.translation
            tx, ty, tz = translation.x, translation.y, translation.z
            
            rotation = init_transform.transform.rotation
            qx, qy, qz, qw = rotation.x, rotation.y, rotation.z, rotation.w
            rotation_matrix = quaternion_to_rotation_matrix(qx, qy, qz, qw)

            self.pose = np.eye(4)
            self.pose[:3, :3] = rotation_matrix
            self.pose[:3, 3] = [tx, ty, tz]
            self.get_logger().info('Set initial pose.')
            
            lidar_transform = self.tf_buffer.lookup_transform(
                target_frame='base_footprint',
                source_frame='velodyne',
                time=Time()
            )
            
            lidar_translation = lidar_transform.transform.translation
            lx, ly, lz = lidar_translation.x, lidar_translation.y, lidar_translation.z

            lidar_rotation = lidar_transform.transform.rotation
            l_qx, l_qy, l_qz, l_qw = lidar_rotation.x, lidar_rotation.y, lidar_rotation.z, lidar_rotation.w
            lidar_rotation_matrix = quaternion_to_rotation_matrix(l_qx, l_qy, l_qz, l_qw)

            self.lidar_pose = np.eye(4)
            self.lidar_pose[:3, :3] = lidar_rotation_matrix
            self.lidar_pose[:3, 3] = [lx, ly, lz]
            self.get_logger().info('Set lidar pose.')
            
            self.timer.cancel()
            
        except Exception as e:
            self.get_logger().warn(f'Could not get transforms: {e}')
            self.pose = None
            self.lidar_pose = None

    def pcd_callback(self, msg: PointCloud2):
        if self.pose is None or self.lidar_pose is None:
            return
        
        if self.prev_pcd is None:
            pts = get_pcd_array_from_point_cloud(msg)
            # Task 2.5 — Process point cloud
            # 1. Extract (N, 3) array of points
            # 2. Transform points from LiDAR frame to robot frame using self.lidar_pose
            # Vectorized transformation
            pts_homogeneous = np.hstack((pts, np.ones((pts.shape[0], 1))))
            transformed_pts = (self.lidar_pose @ pts_homogeneous.T).T[:, :3]
            
            # 3. Initialize Open3D point cloud
            self.prev_pcd = o3d.geometry.PointCloud()
            self.prev_pcd.points = o3d.utility.Vector3dVector(transformed_pts)
            
            # 4. Downsample uniformly by a factor of 10
            self.prev_pcd = self.prev_pcd.uniform_down_sample(10)
            self.get_logger().info('Initial point cloud received and processed.')
            return

        # Task 2.5 — Process point cloud
        pts = get_pcd_array_from_point_cloud(msg)
        pts_homogeneous = np.hstack((pts, np.ones((pts.shape[0], 1))))
        transformed_pts = (self.lidar_pose @ pts_homogeneous.T).T[:, :3]

        # Initialize new point cloud
        current_pcd = o3d.geometry.PointCloud()
        current_pcd.points = o3d.utility.Vector3dVector(transformed_pts)
        current_pcd = current_pcd.uniform_down_sample(10)
        
        # Task 2.6 — Run ICP
        if self.use_open3d:
            # Using Open3D ICP
            T_init = self.transformation
            T = open3d_icp(current_pcd, self.prev_pcd, T_init)
        else:
            # Using homework ICP
            A = np.asarray(current_pcd.points)
            B = np.asarray(self.prev_pcd.points)
            T = icp(A, B, init_pose=self.transformation, max_iterations=20, tolerance=0.001, knn_radius=0.01)
        
        self.transformation = T
        
        # Task 2.7 — Update pose
        # Compose the new transformation with the existing pose
        self.pose = self.pose @ self.transformation
        
        # Task 2.8 — Extract position and update point cloud
        x, y, z = self.pose[:3, 3]
        self.icp_poses.append(np.array([x, y]))
        
        # Update previous point cloud
        self.prev_pcd = current_pcd
        
        self.get_logger().info(f'Updated pose: x={x:.3f}, y={y:.3f}')

    def odom_callback(self, msg: Odometry):
        # Task 2.9 — Extract position from /odom
        position = msg.pose.pose.position
        x = position.x
        y = position.y
        self.odom_poses.append(np.array([x, y]))
        self.get_logger().debug(f'ODOM position: x={x:.3f}, y={y:.3f}')
        
    def plot_poses(self):
        if not self.icp_poses or not self.odom_poses:
            self.get_logger().warn('No poses to plot.')
            return

        icp_poses = np.array(self.icp_poses)
        odom_poses = np.array(self.odom_poses)
        icp_info = 'Open3D' if self.use_open3d else 'HW3'

        # Create plots directory if it doesn't exist
        plots_dir = Path("src/autonomy_repo/plots/")
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Plot x positions
        plt.figure(figsize=(8, 6))
        plt.plot(np.arange(len(icp_poses)), icp_poses[:, 0], 'b-', label='ICP x')
        plt.plot(np.arange(len(odom_poses)), odom_poses[:, 0], 'r-', label='ODOM x')
        plt.xlabel("N")
        plt.ylabel("X position")
        plt.legend()
        plt.title(f"X positions from {icp_info}")
        plt.grid()
        plt.savefig(plots_dir / f"{icp_info}_x.png")
        plt.close()

        # Plot y positions
        plt.figure(figsize=(8, 6))
        plt.plot(np.arange(len(icp_poses)), icp_poses[:, 1], 'b-', label='ICP y')
        plt.plot(np.arange(len(odom_poses)), odom_poses[:, 1], 'r-', label='ODOM y')
        plt.xlabel("N")
        plt.ylabel("Y position")
        plt.legend()
        plt.title(f"Y positions from {icp_info}")
        plt.grid()
        plt.savefig(plots_dir / f"{icp_info}_y.png")
        plt.close()

def main(args=None):
    rclpy.init(args=args)
    node = ICPNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down ICP node.')
    finally:
        node.plot_poses()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
