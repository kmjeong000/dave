from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare


def launch_setup(context, *args, **kwargs):
    paused = LaunchConfiguration("paused")
    gui = LaunchConfiguration("gui")
    use_sim_time = LaunchConfiguration("use_sim_time")
    debug = LaunchConfiguration("debug")
    headless = LaunchConfiguration("headless")
    verbose = LaunchConfiguration("verbose")
    namespace = LaunchConfiguration("namespace")
    world_name = LaunchConfiguration("world_name")
    x = LaunchConfiguration("x")
    y = LaunchConfiguration("y")
    z = LaunchConfiguration("z")
    roll = LaunchConfiguration("roll")
    pitch = LaunchConfiguration("pitch")
    yaw = LaunchConfiguration("yaw")
    use_ned_frame = LaunchConfiguration("use_ned_frame")
    gui_client_delay = LaunchConfiguration("gui_client_delay")
    start_sitl = LaunchConfiguration("start_sitl")
    start_mavros = LaunchConfiguration("start_mavros")
    mavproxy_args = LaunchConfiguration("mavproxy_args")
    start_mavproxy = LaunchConfiguration("start_mavproxy")
    ardupilot_params = LaunchConfiguration("ardupilot_params")
    ardupilot_home = LaunchConfiguration("ardupilot_home")
    mavproxy_start_delay = LaunchConfiguration("mavproxy_start_delay")
    mavros_fcu_url = LaunchConfiguration("mavros_fcu_url")
    sitl_start_delay = LaunchConfiguration("sitl_start_delay")
    mavros_start_delay = LaunchConfiguration("mavros_start_delay")

    if world_name.perform(context) != "empty.sdf":
        world_name = LaunchConfiguration("world_name").perform(context)
        world_filename = f"{world_name}.world"
        world_filepath = PathJoinSubstitution(
            [FindPackageShare("dave_worlds"), "worlds", world_filename]
        )
        gz_args = [world_filepath]
    else:
        gz_args = [world_name]

    if headless.perform(context) == "true":
        gz_args.append(" -s")
    if paused.perform(context) == "false":
        gz_args.append(" -r")
    if debug.perform(context) == "true":
        gz_args.append(" -v ")
        gz_args.append(verbose.perform(context))

    # Include the first launch file
    gz_sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution(
                    [
                        FindPackageShare("ros_gz_sim"),
                        "launch",
                        "gz_sim.launch.py",
                    ]
                )
            ]
        ),
        launch_arguments=[
            ("gz_args", gz_args),
        ],
    )

    split_gui_client = TimerAction(
        period=float(gui_client_delay.perform(context)),
        actions=[
            ExecuteProcess(
                cmd=["gz", "sim", "-g"],
                output="screen",
            )
        ],
        condition=IfCondition(
            PythonExpression(
                [
                    "'",
                    gui,
                    "' == 'true' and '",
                    headless,
                    "' == 'true'",
                ]
            )
        ),
    )

    # Include the second launch file with model name
    robot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution(
                    [
                        FindPackageShare("dave_robot_models"),
                        "launch",
                        "upload_robot.launch.py",
                    ]
                )
            ]
        ),
        launch_arguments={
            "gui": gui,
            "use_sim_time": use_sim_time,
            "namespace": namespace,
            "world_name": "waves",
            "x": x,
            "y": y,
            "z": z,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "use_ned_frame": use_ned_frame,
            "start_sitl": start_sitl,
            "start_mavros": start_mavros,
            "mavproxy_args": mavproxy_args,
            "start_mavproxy": start_mavproxy,
            "ardupilot_params": ardupilot_params,
            "ardupilot_home": ardupilot_home,
            "mavproxy_start_delay": mavproxy_start_delay,
            "mavros_fcu_url": mavros_fcu_url,
            "sitl_start_delay": sitl_start_delay,
            "mavros_start_delay": mavros_start_delay,
        }.items(),
    )

    include = [gz_sim_launch, split_gui_client, robot_launch]

    return include


def generate_launch_description():

    # Declare the launch arguments with default values
    args = [
        DeclareLaunchArgument(
            "paused",
            default_value="false",
            description="Start the simulation continuously",
        ),
        DeclareLaunchArgument(
            "gui",
            default_value="true",
            description="Flag to enable the gazebo gui",
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Flag to indicate whether to use simulation time",
        ),
        DeclareLaunchArgument(
            "debug",
            default_value="false",
            description="Flag to enable the gazebo debug flag",
        ),
        DeclareLaunchArgument(
            "headless",
            default_value="false",
            description="Flag to enable the gazebo headless mode",
        ),
        DeclareLaunchArgument(
            "verbose",
            default_value="0",
            description="Adjust level of console verbosity",
        ),
        DeclareLaunchArgument(
            "gui_client_delay",
            default_value="2.0",
            description="Delay before attaching a separate Gazebo GUI client when gui=true and headless=true",
        ),
        DeclareLaunchArgument(
            "world_name",
            default_value="empty.sdf",
            description="Gazebo world file to launch",
        ),
        DeclareLaunchArgument(
            "namespace",
            default_value="",
            description="Namespace",
        ),
        DeclareLaunchArgument(
            "x",
            default_value="0.0",
            description="Initial x position",
        ),
        DeclareLaunchArgument(
            "y",
            default_value="0.0",
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
            description="Initial roll angle",
        ),
        DeclareLaunchArgument(
            "pitch",
            default_value="0.0",
            description="Initial pitch angle",
        ),
        DeclareLaunchArgument(
            "yaw",
            default_value="0.0",
            description="Initial yaw angle",
        ),
        DeclareLaunchArgument(
            "use_ned_frame",
            default_value="false",
            description="Flag to indicate whether to use the north-east-down frame",
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
            "mavproxy_args",
            default_value="--out=udp:127.0.0.1:14550 --out=udp:127.0.0.1:14560",
            description="MAVProxy output arguments passed to SITL launch helpers",
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
            default_value="9.0",
            description="Seconds to wait before starting MAVROS",
        ),
    ]

    return LaunchDescription(args + [OpaqueFunction(function=launch_setup)])
