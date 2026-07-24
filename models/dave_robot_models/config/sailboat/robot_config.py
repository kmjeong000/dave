from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare



def _as_bool(value):
    return value.lower() in ("1", "true", "yes", "on")


def launch_setup(context, *args, **kwargs):
    namespace = LaunchConfiguration("namespace").perform(context)
    world_name = LaunchConfiguration("world_name").perform(context)

    sailboat_joints = [
        f"/model/{namespace}/joint/rudder_joint",
        f"/model/{namespace}/joint/sail_joint",
    ]

    sailboat_arguments = (
        [f"{joint}/cmd_pos@std_msgs/msg/Float64@gz.msgs.Double" for joint in sailboat_joints]
        + [f"{joint}/base_cmd_pos@std_msgs/msg/Float64[gz.msgs.Double" for joint in sailboat_joints]
        + [f"{joint}/ang_vel@std_msgs/msg/Float64@gz.msgs.Double" for joint in sailboat_joints]
        + [
            f"/model/{namespace}/odometry@nav_msgs/msg/Odometry@gz.msgs.Odometry",
            f"/model/{namespace}/pose@geometry_msgs/msg/PoseArray@gz.msgs.Pose_V",
            f"/model/{namespace}/imu@sensor_msgs/msg/Imu@gz.msgs.IMU",
            f"/model/{namespace}/magnetometer@sensor_msgs/msg/MagneticField@gz.msgs.Magnetometer",
            f"/world/{world_name}/model/{namespace}/link/camera_link/sensor/camera_sensor/image@sensor_msgs/msg/Image@gz.msgs.Image",
            f"/world/{world_name}/model/{namespace}/link/camera_link/sensor/camera_sensor/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo",
            f"/model/{namespace}/gps/fix@sensor_msgs/msg/NavSatFix@gz.msgs.NavSat",
        ]
    )

    sailboat_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=sailboat_arguments,
        output="screen",
    )

    command_adapter = Node(
        package="dave_robot_models",
        executable="sailboat_command_adapter.py",
        name="sailboat_command_adapter",
        output="screen",
        parameters=[
            {
                "namespace": namespace,
                "residual_enabled": False,
                "base_command_timeout_s": 0.5,
                "residual_command_timeout_s": 0.5,
                "rudder_min_rad": -0.7854,
                "rudder_max_rad": 0.7854,
                "rudder_residual_limit_rad": 0.0872664626,
                "rudder_residual_rate_limit_rad_s": 0.1745329252,
                "sail_min_rad": -0.7854,
                "sail_max_rad": 0.7854,
                "sail_residual_limit_rad": 0.0872664626,
                "sail_residual_rate_limit_rad_s": 0.1745329252,
            }
        ],
    )

    actions = [sailboat_bridge, command_adapter]

    if _as_bool(LaunchConfiguration("start_sitl").perform(context)):
        ardupilot_params = LaunchConfiguration("ardupilot_params").perform(context)
        sitl_start_delay = float(LaunchConfiguration("sitl_start_delay").perform(context))
        ardupilot_home = LaunchConfiguration("ardupilot_home").perform(context)
        
        sitl_cmd = [
            "ardurover -S -w --model json:127.0.0.1 "
            + "--defaults " + ardupilot_params + " "
            + "-I0 "
            + "--home " + ardupilot_home
        ]
        
        actions.append(
            TimerAction(
                period=sitl_start_delay,
                actions=[ExecuteProcess(cmd=sitl_cmd, shell=True, output="screen")],
            )
        )
        
    if _as_bool(LaunchConfiguration("start_mavproxy").perform(context)):
        mavproxy_args = LaunchConfiguration("mavproxy_args").perform(context)
        mavproxy_start_delay = float(LaunchConfiguration("mavproxy_start_delay").perform(context))
        mavproxy_cmd = [
            "mavproxy.py " 
            + "--master=tcp:127.0.0.1:5760 "
            + mavproxy_args
        ]
        
        actions.append(
            TimerAction(
                period=mavproxy_start_delay,
                actions=[ExecuteProcess(cmd=mavproxy_cmd, shell=True, output="log")],
            )
        )
        
    if _as_bool(LaunchConfiguration("start_mavros").perform(context)):
        mavros_fcu_url = LaunchConfiguration("mavros_fcu_url").perform(context)
        mavros_start_delay = float(LaunchConfiguration("mavros_start_delay").perform(context))
        mavros_cmd = ["ros2 launch mavros apm.launch fcu_url:=" + mavros_fcu_url]
        actions.append(
            TimerAction(
                period=mavros_start_delay,
                actions=[ExecuteProcess(cmd=mavros_cmd, shell=True, output="screen")],
            )
        )

    return actions


def generate_launch_description():
    args = [
        DeclareLaunchArgument(
            "namespace",
            default_value="",
            description="Namespace",
        ),
        DeclareLaunchArgument(
            "world_name",
            default_value="waves",
            description="Gazebo world topic name",
        ),
        DeclareLaunchArgument(
            "start_sitl",
            default_value="true",
            description="Start ArduRover SITL with JSON Gazebo backend",
        ),
        DeclareLaunchArgument(
            "ardupilot_params",
            default_value=PathJoinSubstitution(
                [FindPackageShare("dave_robot_models"), "config", "sailboat", "ardurover.parm"]
            ),
            description="Path to ArduRover parameter file",
        ),
        DeclareLaunchArgument(
            "ardupilot_home",
            default_value="44.65870,-124.06556,0.0,270.0",
            description="ArduRover home position",
        ),
        DeclareLaunchArgument(
            "start_mavproxy",
            default_value="true",
            description="Start MAVProxy to fan out telemetry to QGC and MAVROS",
        ),
        DeclareLaunchArgument(
            "start_mavros",
            default_value="true",
            description="Start MAVROS using mavros apm.launch",
        ),
        DeclareLaunchArgument(
            "mavproxy_args",
            default_value="--out=udp:127.0.0.1:14550 --out=udp:127.0.0.1:14560",
            description="MAVProxy output arguments",
        ),
        DeclareLaunchArgument(
            "mavros_fcu_url",
            default_value="udp://:14560@127.0.0.1:14560",
            description="MAVROS FCU URL",
        ),
        DeclareLaunchArgument(
            "sitl_start_delay",
            default_value="5.0",
            description="Seconds to wait before starting SITL",
        ),
        DeclareLaunchArgument(
            "mavproxy_start_delay", 
            default_value="7.0",
            description="Seconds to wait before starting MAVProxy",
        ),
        DeclareLaunchArgument(
            "mavros_start_delay",
            default_value="9.0",
            description="Seconds to wait before starting MAVROS",
        )
    ]

    return LaunchDescription(args + [OpaqueFunction(function=launch_setup)])
