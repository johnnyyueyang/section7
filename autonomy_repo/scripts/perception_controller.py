#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from asl_tb3_lib.control import BaseController
from asl_tb3_msgs.msg import TurtleBotControl
import threading

class PerceptionController(BaseController):
    def __init__(self):
        super().__init__('perception_controller')
        self.declare_parameter('active', True)
        self.declare_parameter('debounce_time', 2.0)  # Debounce time in seconds
        self._active = True  # Internal flag to manage stopping
        self.stop_timer = None  # Timer to handle the 5-second stop
        self.debounce_time = self.get_parameter('debounce_time').get_parameter_value().double_value
        self.debounce_timer = None  # Timer for debounce

        # Lock to manage thread-safe operations
        self.lock = threading.Lock()

        self.get_logger().info('PerceptionController initialized and active.')

        # Task 4.1 - Initialize Subscriber to /detector_bool
        self.bool_subscriber = self.create_subscription(
            Bool,
            '/detector_bool',
            self.detector_bool_callback,
            10  # QoS History Depth
        )
        self.bool_subscriber  # Prevent unused variable warning

    @property
    def active(self) -> bool:
        """
        Property to access the 'active' parameter.
        """
        return self.get_parameter('active').get_parameter_value().bool_value

    def detector_bool_callback(self, msg: Bool):
        """
        Callback function for /detector_bool topic.
        """
        if msg.data:
            self.get_logger().info('Stop sign detected!')
            with self.lock:
                if self._active and not self.in_debounce:
                    self._active = False  # Update internal state
                    # Stop the robot
                    self.get_logger().info('Stopping TurtleBot for 5 seconds.')
                    # Start the stop timer
                    if self.stop_timer is None:
                        self.stop_timer = self.create_timer(5.0, self.resume_movement)
                    # Start the debounce timer
                    self.start_debounce()
        else:
            self.get_logger().info('No stop sign detected.')

    def start_debounce(self):
        """
        Starts the debounce timer to ignore detections for a short period.
        """
        if self.debounce_timer is None:
            self.debounce_timer = self.create_timer(self.debounce_time, self.end_debounce)
            self.get_logger().info(f'Debounce started for {self.debounce_time} seconds.')

    def end_debounce(self):
        """
        Callback to end the debounce period.
        """
        with self.lock:
            self.get_logger().info('Debounce period ended. Ready to detect stop signs again.')
            if self.debounce_timer:
                self.debounce_timer.cancel()
                self.debounce_timer = None

    def resume_movement(self):
        """
        Callback function to resume movement after stopping.
        """
        with self.lock:
            self._active = True
            self.get_logger().info('Resuming movement after stop.')
            # Reset the active parameter to True
            self.set_parameters([rclpy.parameter.Parameter('active', rclpy.Parameter.Type.BOOL, True)])
            # Destroy the stop timer as it's no longer needed
            if self.stop_timer:
                self.stop_timer.cancel()
                self.stop_timer = None

    def compute_control(self) -> TurtleBotControl:
        control_msg = TurtleBotControl()

        with self.lock:
            if self.active and self._active:
                # Robot should spin
                control_msg.v = 0.0          # No linear movement
                control_msg.omega = 0.5      # Angular velocity for spinning
                self.get_logger().info('Spinning with omega = 0.5')
            elif not self.active and self._active:
                # Stop the robot and start the timer for resuming
                control_msg.v = 0.0
                control_msg.omega = 0.0
                self.get_logger().info('Stopping TurtleBot for 5 seconds.')
                self._active = False  # Update internal state

                # Start a timer to resume movement after 5 seconds
                if self.stop_timer is None:
                    self.stop_timer = self.create_timer(5.0, self.resume_movement)
            elif not self.active and not self._active:
                # Currently in stopped state; maintain zero velocities
                control_msg.v = 0.0
                control_msg.omega = 0.0
                self.get_logger().info('TurtleBot is stopped.')
            else:
                # Default behavior: ensure the robot is stopped
                control_msg.v = 0.0
                control_msg.omega = 0.0
                self.get_logger().info('Default stop behavior.')

        return control_msg

    @property
    def in_debounce(self):
        """
        Returns True if the node is currently in the debounce period.
        """
        return self.debounce_timer is not None

def main(args=None):
    rclpy.init(args=args)
    controller = PerceptionController()
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down PerceptionController node.')
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
