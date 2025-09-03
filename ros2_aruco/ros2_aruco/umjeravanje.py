import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray
import numpy as np


class DataCollector(Node):
    def __init__(self):
        super().__init__('vector_sum_stats_node')
        self.odom_sub = self.create_subscription(Odometry, '/scout/main/odometry', self.odom_callback, 10)
        self.poses_sub = self.create_subscription(PoseArray, '/front/aruco/enu_transformed_poses', self.poses_callback, 10)

        self.odom_msgs = []
        self.pose_msgs = []
        self.max_count = 200

    def odom_callback(self, msg):
        if len(self.odom_msgs) < self.max_count:
            self.odom_msgs.append(msg)
            self.try_compute()

    def poses_callback(self, msg):
        if len(self.pose_msgs) < self.max_count:
            # assume using the first pose in PoseArray
            if len(msg.poses) > 0:
                self.pose_msgs.append(msg.poses[0])
            self.try_compute()

    def try_compute(self):
        if len(self.odom_msgs) == self.max_count and len(self.pose_msgs) == self.max_count:
            self.get_logger().info('Collected 200 messages from each topic. Computing...')
            self.compute_stats()
            rclpy.shutdown()

    def compute_stats(self):
        sums = []
        for odom, pose in zip(self.odom_msgs, self.pose_msgs):
            sum_x = odom.pose.pose.position.x + pose.position.x
            sum_y = odom.pose.pose.position.y + pose.position.y
            sum_z = odom.pose.pose.position.z + pose.position.z
            sums.append([sum_x, sum_y, sum_z])

        data = np.array(sums)
        mean = np.mean(data, axis=0)
        stddev = np.std(data, axis=0)

        print(f"Mean:    x={mean[0]:.3f}, y={mean[1]:.3f}, z={mean[2]:.3f}")
        print(f"Stddev:  x={stddev[0]:.3f}, y={stddev[1]:.3f}, z={stddev[2]:.3f}")


def main(args=None):
    rclpy.init(args=args)
    node = DataCollector()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
