"""
This node locates Aruco AR markers in images and publishes their ids and poses.

Subscriptions:
   /camera/image_raw (sensor_msgs.msg.Image)
   /camera/camera_info (sensor_msgs.msg.CameraInfo)
   /camera/camera_info (sensor_msgs.msg.CameraInfo)

Published Topics:
    /aruco_poses (geometry_msgs.msg.PoseArray)
       Pose of all detected markers (suitable for rviz visualization)

    /aruco_markers (ros2_aruco_interfaces.msg.ArucoMarkers)
       Provides an array of all poses along with the corresponding
       marker ids.

Parameters:
    marker_size - size of the markers in meters (default .0625)
    aruco_dictionary_id - dictionary that was used to generate markers
                          (default DICT_5X5_250)
    image_topic - image topic to subscribe to (default /camera/image_raw)
    camera_info_topic - camera info topic to subscribe to
                         (default /camera/camera_info)

Author: Nathan Sprague
Version: 10/26/2020

"""

import rclpy
import rclpy.node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
from packaging import version
import numpy as np
import cv2
import tf_transformations
import tf2_ros
import tf2_geometry_msgs
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from nav_msgs.msg import Odometry
from ros2_aruco_interfaces.msg import ArucoMarkers
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from scipy.spatial.transform import Rotation as R


class ArucoNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("aruco_node")

        # Declare and read parameters
        self.declare_parameter(
            name="marker_size",
            value=0.0625,
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description="Size of the markers in meters.",
            ),
        )

        self.declare_parameter(
            name="aruco_dictionary_id",
            value="DICT_5X5_250",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Dictionary that was used to generate markers.",
            ),
        )

        self.declare_parameter(
            name="image_topic",
            value="/camera/image_raw",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Image topic to subscribe to.",
            ),
        )

        self.declare_parameter(
            name="camera_info_topic",
            value="/camera/camera_info",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Camera info topic to subscribe to.",
            ),
        )

        self.declare_parameter(
            name="camera_frame",
            value="",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Camera optical frame to use.",
            ),
        )

        self.declare_parameter(
            name="robot_odom",
            value="/scout/main/odometry",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Robot pose topic to use.",
            ),
        )

        self.marker_size = (
            self.get_parameter("marker_size").get_parameter_value().double_value
        )
        self.get_logger().info(f"Marker size: {self.marker_size}")

        dictionary_id_name = (
            self.get_parameter("aruco_dictionary_id").get_parameter_value().string_value
        )
        self.get_logger().info(f"Marker type: {dictionary_id_name}")

        image_topic = (
            self.get_parameter("image_topic").get_parameter_value().string_value
        )
        self.get_logger().info(f"Image topic: {image_topic}")

        info_topic = (
            self.get_parameter("camera_info_topic").get_parameter_value().string_value
        )
        self.get_logger().info(f"Image info topic: {info_topic}")

        self.camera_frame = (
            self.get_parameter("camera_frame").get_parameter_value().string_value
        )

        self.robot_odom_topic = (
            self.get_parameter("robot_odom").get_parameter_value().string_value
        )
        # self.get_logger().info(f"Robot pose topic: {robot_odom}")

        # Make sure we have a valid dictionary id:
        try:
            dictionary_id = cv2.aruco.__getattribute__(dictionary_id_name)
            if type(dictionary_id) != type(cv2.aruco.DICT_5X5_100):
                raise AttributeError
        except AttributeError:
            self.get_logger().error(
                "bad aruco_dictionary_id: {}".format(dictionary_id_name)
            )
            options = "\n".join([s for s in dir(cv2.aruco) if s.startswith("DICT")])
            self.get_logger().error("valid options: {}".format(options))

        # Set up subscriptions
        self.info_sub = self.create_subscription(
            CameraInfo, info_topic, self.info_callback, qos_profile_sensor_data
        )

        self.create_subscription(
            Image, image_topic, self.image_callback, qos_profile_sensor_data
        )

        self.robot_pose_sub = self.create_subscription(
            Odometry, self.robot_odom_topic, self.odom_callback, qos_profile_sensor_data
        )
        
        # flags
        self.camera_info_received = False
        self.transform_flag = False
        self.odom_received = False

        # tf
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Load AR tags
        self.declare_parameter("ARTags", [
            "0 0.0 0.0 0.0"
        ])
        self.ar_tags = self._load_artags_from_string_list()
        self.get_logger().info(f"Loaded ARTags: {self.ar_tags}")
        
        # Set up publishers
        self.declare_parameter("poses_topic", "aruco_poses")
        self.declare_parameter("markers_topic", "aruco_markers")
        self.declare_parameter("transformed_poses_topic", "aruco_transformed")
        self.declare_parameter("enu_transformed_poses_topic", "enu_transformed_poses_topic")
        self.declare_parameter("enu_robot_pose_topic", "enu_robot_pose_topic")
        self.declare_parameter("robot_frame_id", "base_link")
        
        poses_topic = self.get_parameter("poses_topic").get_parameter_value().string_value
        markers_topic = self.get_parameter("markers_topic").get_parameter_value().string_value
        transformed_poses_topic = self.get_parameter("transformed_poses_topic").get_parameter_value().string_value
        enu_transformed_poses_topic = self.get_parameter("enu_transformed_poses_topic").get_parameter_value().string_value
        enu_robot_pose_topic = self.get_parameter("enu_robot_pose_topic").get_parameter_value().string_value
        self.robot_frame_id = self.get_parameter("robot_frame_id").get_parameter_value().string_value
        
        self.get_logger().info(f"AR pose in camera frame: {poses_topic}")
        self.get_logger().info(f"AR markers in camera frame: {markers_topic}")
        self.get_logger().info(f"AR pose in BASE_LINK frame: {transformed_poses_topic}")
        self.get_logger().info(f"AR pose in ENU frame: {enu_transformed_poses_topic}")
        self.get_logger().info(f"Robot pose in ENU frame: {enu_robot_pose_topic}")

        self.poses_pub = self.create_publisher(PoseArray, poses_topic, 10)
        self.markers_pub = self.create_publisher(ArucoMarkers, markers_topic, 10)
        self.transformed_pub = self.create_publisher(PoseArray, transformed_poses_topic, 10)
        self.enu_transformed_pub = self.create_publisher(PoseArray, enu_transformed_poses_topic, 10)
        self.enu_robot_pose_pub = self.create_publisher(PoseStamped, enu_robot_pose_topic, 10)
        
        self.get_logger().info(f"Robot frame ID: {self.robot_frame_id}")

        # Set up fields for camera parameters
        self.info_msg = None
        self.intrinsic_mat = None
        self.distortion = None

        self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.aruco_parameters = cv2.aruco.DetectorParameters()
        self.bridge = CvBridge()

    def info_callback(self, info_msg):
        self.info_msg = info_msg
        self.intrinsic_mat = np.reshape(np.array(self.info_msg.k), (3, 3))
        self.distortion = np.array(self.info_msg.d)
        # Assume that camera parameters will remain the same...
        self.destroy_subscription(self.info_sub)
    
    def odom_callback(self, msg):
        self.robot_position = msg.pose.pose.position
        self.robot_orientation = msg.pose.pose.orientation
        self.robot_quaternion = ([
            self.robot_orientation.x,
            self.robot_orientation.y,
            self.robot_orientation.z,
            self.robot_orientation.w])
        self.robot_orientation_euler = R.from_quat(self.robot_quaternion)
        self.odom_received = True
        
    def image_callback(self, img_msg):
        if self.info_msg is None:
            self.get_logger().warn("No camera info has been received!")
            return
        elif not self.camera_info_received:
            self.get_logger().info("Camera info received, processing images now.")
            self.camera_info_received = True
        
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="mono8")
        markers = ArucoMarkers()
        pose_array = PoseArray()
        
        markers.header = img_msg.header
        pose_array.header = img_msg.header
        
        markers.header.frame_id = img_msg.header.frame_id
        pose_array.header.frame_id = img_msg.header.frame_id

        aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dictionary, self.aruco_parameters)
        corners, marker_ids, rejected = aruco_detector.detectMarkers(cv_image)

        #corners, marker_ids, rejected = cv2.aruco.ArucoDetector.detectMarkers(
        #    cv_image, self.aruco_dictionary, parameters=self.aruco_parameters
        #)
        source_frame = img_msg.header.frame_id  
        target_frame = self.robot_frame_id

        if not self.transform_flag:
            try:
                # Lookup the transform from source_frame to target_frame
                # Use the timestamp of the received pose for accuracy
                self.transform = self.tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    rclpy.time.Time(),
                    rclpy.duration.Duration(seconds=1.0) # Timeout for the transform lookup
                )
                self.get_logger().info(f"Lookup successful for transform from {source_frame} to {target_frame}")
                self.transform_flag = True
            except Exception as e:
                self.get_logger().info(f"Source frame: {source_frame}")
                self.get_logger().info(f"Target frame: {target_frame}")
                self.get_logger().error(f"Failed to lookup transform: {e}")
                return
        
        if marker_ids is not None and self.odom_received:
            #self.get_logger().info(f"cv2 version: {cv2.__version__}")

            if version.parse(cv2.__version__) > version.parse("4.0.0"):
                # self.get_logger().info('Version larger than 4.0.0')
                # rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                #     corners, self.marker_size, self.intrinsic_mat, self.distortion
                # )
                rvecs = []
                tvecs = []

                for corner in corners:
                    success, rvec, tvec = cv2.solvePnP(
                        objectPoints=self._create_3d_marker_corners(),
                        imagePoints=corner,
                        cameraMatrix=self.intrinsic_mat,
                        distCoeffs=self.distortion
                    )
                    if success:
                        rvecs.append(rvec)
                        tvecs.append(tvec)

            else:
                rvecs, tvecs = cv2.aruco.estimatePoseSingleMarkers(
                    corners, self.marker_size, self.intrinsic_mat, self.distortion
                )
            
            valid_marker_ids = []
            valid_poses = []
            pose_array = PoseArray()
            markers = ArucoMarkers()

            for corner, marker_id in zip(corners, marker_ids):
                success, rvec, tvec = cv2.solvePnP(
                    objectPoints=self._create_3d_marker_corners(),
                    imagePoints=corner,
                    cameraMatrix=self.intrinsic_mat,
                    distCoeffs=self.distortion
                )
                if not success:
                    continue
            
                pose = Pose()
                pose.position.x = float(tvec[0])
                pose.position.y = float(tvec[1])
                pose.position.z = float(tvec[2])
            
                rot_matrix = np.eye(4)
                rot_matrix[0:3, 0:3] = cv2.Rodrigues(rvec)[0]
                quat = tf_transformations.quaternion_from_matrix(rot_matrix)
            
                pose.orientation.x = float(quat[0])
                pose.orientation.y = float(quat[1])
                pose.orientation.z = float(quat[2])
                pose.orientation.w = float(quat[3])
            
                pose_array.poses.append(pose)
                markers.poses.append(pose)
                # self.get_logger().info(f"Marker ID: {marker_id[0]}, Pose: {pose}")
                markers.marker_ids.append(marker_id[0])

            transformed_pose_array = PoseArray()
            enu_transformed_pose_array = PoseArray()
            
            for pose, marker_id in zip(pose_array.poses, markers.marker_ids):
                # load position by marker id
                # self.get_logger().info(str(type(marker_id)))
                if marker_id not in self.ar_tags:
                    #self.get_logger().info(f"Marker ID {marker_id} ignored (not listed in ARTags)")
                    continue
                else:
                    #self.get_logger().info(f"Marker ID {marker_id} in processing.")
                    xyz = self.ar_tags[marker_id]
                    self.ar_local = np.array(xyz)
                
                transformed_pose = tf2_geometry_msgs.do_transform_pose(pose, self.transform)
                transformed_pose_array.header.frame_id = self.robot_frame_id
                transformed_pose_array.poses.append(transformed_pose)
                
                # rotirati u ENU frame
                enu_transformed_pose = Pose()
                position_vector = np.array([transformed_pose.position.x,
                                            transformed_pose.position.y,
                                            transformed_pose.position.z])
                enu_pose = self.robot_orientation_euler.apply(position_vector)                
                enu_transformed_pose.position.x = enu_pose[0]
                enu_transformed_pose.position.y = enu_pose[1]
                enu_transformed_pose.position.z = enu_pose[2] 
                enu_transformed_pose_array.poses.append(enu_transformed_pose)
                
                # oduzeti od AR markera i dobiti poziciju robota u enu frame 
                enu_robot_pose = PoseStamped()
                enu_robot_pose.header.frame_id = "ENU"
                enu_robot_pose.header.stamp = img_msg.header.stamp
                enu_robot_pose.pose.position.x = -enu_transformed_pose.position.x + self.ar_local[0]
                enu_robot_pose.pose.position.y = -enu_transformed_pose.position.y + self.ar_local[1]
                enu_robot_pose.pose.position.z = -enu_transformed_pose.position.z + self.ar_local[2]
                
                # self.get_logger().info(f"{self.robot_orientation}")
                
                enu_robot_pose.pose.orientation = self.robot_orientation  # orientation from odometry   
                
                self.enu_robot_pose_pub.publish(enu_robot_pose)
                # self.get_logger().info(f"Published enu_robot_pose for marker {marker_id}")

            self.poses_pub.publish(pose_array)
            self.markers_pub.publish(markers)
            self.enu_transformed_pub.publish(enu_transformed_pose_array)
            self.transformed_pub.publish(transformed_pose_array)
            
            # self.get_logger().info(f"ar marker {self.ar_local}.")    
            # self.get_logger().info(f"x diff: {np.abs(enu_robot_pose.pose.position.x - self.robot_position.x)}")
            # self.get_logger().info(f"y diff: {np.abs(enu_robot_pose.pose.position.y - self.robot_position.y)}")
            
            # self.transform_flag = False
        self.odom_received = False

    def _create_3d_marker_corners(self):
        half_size = self.marker_size / 2.0
        return np.array([
            [-half_size,  half_size, 0],
            [ half_size,  half_size, 0],
            [ half_size, -half_size, 0],
            [-half_size, -half_size, 0]
        ], dtype=np.float32)

    def _load_artags_from_string_list(self):
        result = {}
        try:
            tag_lines = list(self.get_parameter("ARTags").get_parameter_value().string_array_value)
            for line in tag_lines:
                parts = line.strip().split()
                if len(parts) != 4:
                    self.get_logger().warn(f"Skipping malformed ARTag string: {line}")
                    continue
                marker_id = int(parts[0])
                position = [float(p) for p in parts[1:]]
                result[marker_id] = position
        except Exception as e:
            self.get_logger().error(f"Failed to parse ARTags: {e}")
        return result


def main():
    rclpy.init()
    node = ArucoNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
