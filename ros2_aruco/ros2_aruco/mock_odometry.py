import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Point, Quaternion
import numpy as np


class OdometryMocker(Node):
    def __init__(self):
        super().__init__('odometry_mock_publisher')
        self.publisher = self.create_publisher(Odometry, '/scout/odometry', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10 Hz
        self.seq = 0

    def timer_callback(self):
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'base_link'

        # Random position around (1.0, 2.0, 0.5) with small noise
        position = Point()
        position.x = 1.0 + np.random.normal(0, 0.1)
        position.y = 2.0 + np.random.normal(0, 0.1)
        position.z = 0.5 + np.random.normal(0, 0.05)

        # Fixed orientation
        orientation = Quaternion()
        orientation.w = 1.0

        msg.pose.pose = Pose(position, orientation)
        self.publisher.publish(msg)
        self.get_logger().info(f'Published Odometry #{self.seq}: x={position.x:.2f}, y={position.y:.2f}, z={position.z:.2f}')
        self.seq += 1


def main(args=None):
    rclpy.init(args=args)
    node = OdometryMocker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
