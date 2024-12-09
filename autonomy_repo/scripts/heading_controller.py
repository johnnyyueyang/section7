#!/usr/bin/env python3

# Import necessary libraries
import numpy as np
import rclpy

from asl_tb3_lib.control import BaseHeadingController
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState
from asl_tb3_lib.math_utils import wrap_angle


class HeadingController(BaseHeadingController):
    def __init__(self) -> None:
        super().__init__()

        # Declare the 'kp' parameter with a default value of 2.0
        self.declare_parameter("kp", 2.0)

    @property
    def kp(self) -> float:
        """
        Get the real-time value of the proportional gain 'kp'.

        Returns:
            float: Current value of 'kp' parameter.
        """
        return self.get_parameter("kp").value

    def compute_control_with_goal(
        self,
        current_state: TurtleBotState,
        goal_state: TurtleBotState,
    ) -> TurtleBotControl:
        """
        This method computes the control for the TurtleBot to reach the desired heading.

        Args:
            current_state (TurtleBotState): The current state of the TurtleBot.
            goal_state (TurtleBotState): The desired state of the TurtleBot.

        Returns:
            TurtleBotControl: The control message with angular velocity.
        """
        # Calculate the heading error and wrap it within [-pi, pi]
        heading_error = wrap_angle(goal_state.theta - current_state.theta)
        
        # Apply proportional control to compute angular velocity using the 'kp' parameter
        omega = self.kp * heading_error
        
        # Create a control message
        control_msg = TurtleBotControl()
        control_msg.omega = omega
        
        return control_msg


def main(args=None):
    rclpy.init(args=args)  # Initialize the ROS2 system
    
    # Create an instance of HeadingController
    heading_controller = HeadingController()
    
    try:
        # Keep the node running and listening for messages
        rclpy.spin(heading_controller)
    except KeyboardInterrupt:
        pass
    finally:
        # Shutdown ROS2 system when done
        heading_controller.destroy_node()
        rclpy.shutdown()


# Check if this script is the main entry point
if __name__ == "__main__":
    main()
