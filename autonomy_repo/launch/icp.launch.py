#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")

    return LaunchDescription(
        [
            # Declare the use_sim_time launch argument
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="false",
                description="Use simulation (Gazebo) clock if true"
            ),
            
            # Launch RViz with the specified configuration
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                output='screen',
                arguments=['-d', PathJoinSubstitution([
                    FindPackageShare("autonomy_repo"),
                    "rviz",
                    "default.rviz"
                ])],
                parameters=[{"use_sim_time": use_sim_time}]
            ),
            
            # Launch ICP Node
            Node(
                package='autonomy_repo',
                executable='icp_node.py',
                name='icp_node',
                output='screen',
                parameters=[{"use_sim_time": use_sim_time}]
            ),
        ]
    )
