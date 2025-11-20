from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, OpaqueFunction


def launch_setup(context, *args, **kwargs):
    namespace = LaunchConfiguration("namespace").perform(context)

    sailboat_joints = (
        [f"/model/{namespace}/joint/rudder_joint",
         f"/model/{namespace}/joint/sail_joint",])
    
    sailboat_arguments = (
        [f"{joint}/cmd_pos@std_msgs/msg/Float64@gz.msgs.Double" for joint in sailboat_joints]
        + [f"{joint}/ang_vel@std_msgs/msg/Float64@gz.msgs.Double" for joint in sailboat_joints]
        + [
            f"/model/{namespace}/odometry@nav_msgs/msg/Odometry@gz.msgs.Odometry",
            f"/model/{namespace}/pose@geometry_msgs/msg/PoseArray@gz.msgs.Pose_V",
            f"/model/{namespace}/imu@sensor_msgs/msg/Imu@gz.msgs.IMU",
            f"/model/{namespace}/magnetometer@sensor_msgs/msg/MagneticField@gz.msgs.Magnetometer",
            f"/world/waves/model/{namespace}/link/camera_link/sensor/camera_sensor/image@sensor_msgs/msg/Image@gz.msgs.Image",
            f"/world/waves/model/{namespace}/link/camera_link/sensor/camera_sensor/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo",
            f"/model/{namespace}/gps/fix@sensor_msgs/msg/NavSatFix@gz.msgs.Nav.Sat",
        ]
    )

    sailboat_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=sailboat_arguments,
        output="screen",
    )

    return [sailboat_bridge]

def generate_launch_description():
    args = [
        DeclareLaunchArgument(
            "namespace",
            default_value="",
            description="Namespace",
        )
    ]

    return LaunchDescription(args + [OpaqueFunction(function=launch_setup)])