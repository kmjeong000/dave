from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    RegisterEventHandler,
    LogInfo,
    IncludeLaunchDescription,
)
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch.conditions import IfCondition
from launch_ros.substitutions import FindPackageShare
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    gui = LaunchConfiguration("gui")
    use_sim_time = LaunchConfiguration("use_sim_time")
    namespace = LaunchConfiguration("namespace")
    world_name = LaunchConfiguration("world_name")
    x = LaunchConfiguration("x")
    y = LaunchConfiguration("y")
    z = LaunchConfiguration("z")
    roll = LaunchConfiguration("roll")
    pitch = LaunchConfiguration("pitch")
    yaw = LaunchConfiguration("yaw")
    use_ned_frame = LaunchConfiguration("use_ned_frame")
    start_sitl = LaunchConfiguration("start_sitl")
    start_mavros = LaunchConfiguration("start_mavros")
    start_mavproxy = LaunchConfiguration("start_mavproxy")
    ardupilot_params = LaunchConfiguration("ardupilot_params")
    ardupilot_home = LaunchConfiguration("ardupilot_home")
    mavproxy_start_delay = LaunchConfiguration("mavproxy_start_delay")
    mavproxy_args = LaunchConfiguration("mavproxy_args")
    mavros_fcu_url = LaunchConfiguration("mavros_fcu_url")
    sitl_start_delay = LaunchConfiguration("sitl_start_delay")
    mavros_start_delay = LaunchConfiguration("mavros_start_delay")

    args = [
        DeclareLaunchArgument(
            "gui",
            default_value="true",
            description="Flag to indicate whether to use simulation",
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Flag to indicate whether to use sim time",
        ),
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
            "x",
            default_value="0",
            description="Initial x position",
        ),
        DeclareLaunchArgument(
            "y",
            default_value="0",
            description="Initial y position",
        ),
        DeclareLaunchArgument(
            "z",
            default_value="0.0",
            description="Initial z position",
        ),
        DeclareLaunchArgument(
            "roll",
            default_value="0.0",
            description="Initial roll",
        ),
        DeclareLaunchArgument(
            "pitch",
            default_value="0.0",
            description="Initial pitch",
        ),
        DeclareLaunchArgument(
            "yaw",
            default_value="0.0",
            description="Initial yaw",
        ),
        DeclareLaunchArgument(
            "use_ned_frame",
            default_value="false",
            description="Use North-East-Down frame",
        ),
        DeclareLaunchArgument(
            "start_sitl",
            default_value="true",
            description="Start ArduPilot SITL for robot configs that support it",
        ),
        DeclareLaunchArgument(
            "start_mavros",
            default_value="true",
            description="Start MAVROS for robot configs that support it",
        ),
        DeclareLaunchArgument(
            "start_mavproxy",
            default_value="true",
            description="Start MAVProxy to fan out telemetry to QGC and MAVROS",
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
            "mavproxy_start_delay",
            default_value="7.0",
            description="Seconds to wait before starting MAVProxy",
        ),
        DeclareLaunchArgument(
            "mavproxy_args",
            default_value="--out=udp:127.0.0.1:14550 --out=udp:127.0.0.1:14560",
            description="MAVProxy output arguments passed to SITL launch helpers",
        ),
        DeclareLaunchArgument(
            "mavros_fcu_url",
            default_value="udp://:14560@127.0.0.1:14560",
            description="MAVROS FCU URL for robot configs that support it",
        ),
        DeclareLaunchArgument(
            "sitl_start_delay",
            default_value="5.0",
            description="Seconds to wait before starting SITL",
        ),
        DeclareLaunchArgument(
            "mavros_start_delay",
            default_value="8.0",
            description="Seconds to wait before starting MAVROS",
        ),
    ]

    description_file = PathJoinSubstitution(
        [
            FindPackageShare("dave_robot_models"),
            "description",
            namespace,
            "model.sdf",
        ]
    )

    tf2_spawner = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="world_to_world_ned",
        arguments=[
            "--roll",
            "1.57",
            "--yaw",
            "3.14",
            "--frame_id",
            "world",
            "--child_frame_id",
            "world_ned",
        ],
        output="both",
        condition=IfCondition(use_ned_frame),
        parameters=[{"use_sim_time": use_sim_time}],
    )

    gz_spawner = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-name",
            namespace,
            "-file",
            description_file,
            "-x",
            x,
            "-y",
            y,
            "-z",
            z,
            "-R",
            roll,
            "-P",
            pitch,
            "-Y",
            yaw,
        ],
        output="both",
        parameters=[{"use_sim_time": use_sim_time}],
    )

    nodes = [tf2_spawner, gz_spawner]

    # Include robot_config.py based on the model name
    robot_config = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution(
                    [
                        FindPackageShare("dave_robot_models"),
                        "config",
                        namespace,
                        "robot_config.py",
                    ]
                )
            ]
        ),
        launch_arguments={
            "namespace": namespace,
            "world_name": world_name,
            "start_sitl": start_sitl,
            "start_mavros": start_mavros,
            "start_mavproxy": start_mavproxy,
            "ardupilot_params": ardupilot_params,
            "ardupilot_home": ardupilot_home,
            "mavproxy_start_delay": mavproxy_start_delay,
            "mavproxy_args": mavproxy_args,
            "mavros_fcu_url": mavros_fcu_url,
            "sitl_start_delay": sitl_start_delay,
            "mavros_start_delay": mavros_start_delay,
        }.items(),
    )

    include = [robot_config]

    event_handlers = [
        RegisterEventHandler(
            OnProcessExit(target_action=gz_spawner, on_exit=LogInfo(msg="Robot Model Uploaded"))
        )
    ]

    return LaunchDescription(args + nodes + event_handlers + include)
