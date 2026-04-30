from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    waypoints_file_path = LaunchConfiguration('waypoints_file_path')
    odom_topic = LaunchConfiguration('odom_topic')
    speed = LaunchConfiguration('speed')
    lookahead = LaunchConfiguration('lookahead')

    return LaunchDescription([
        DeclareLaunchArgument('waypoints_file_path', default_value='test.csv'),
        DeclareLaunchArgument('odom_topic', default_value='/ego_racecar/odom'),
        DeclareLaunchArgument('speed', default_value='5.0'),
        DeclareLaunchArgument('lookahead', default_value='2.0'),
        Node(
            package='pure_pursuit',
            executable='pure_pursuit_node',
            name='pure_pursuit_node',
            output='screen',
            parameters=[{
                'waypoints_file_path': waypoints_file_path,
                'odom_topic': odom_topic,
                'speed': speed,
                'lookahead': lookahead,
            }],
        ),
    ])
