from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import glob
import os


def generate_launch_description():
    # 获取包路径
    open3d_loc_share = get_package_share_directory('open3d_loc')
    workspace_root = open3d_loc_share.split('/install/')[0] if '/install/' in open3d_loc_share else ''
    source_candidates = glob.glob(os.path.join(workspace_root, 'src', '**', 'open3d_loc'), recursive=True)
    open3d_loc_dir = source_candidates[0] if source_candidates else open3d_loc_share
    log_dir = os.path.join(open3d_loc_dir, 'log')
    os.makedirs(log_dir, exist_ok=True)

    ros_log_dir = SetEnvironmentVariable('ROS_LOG_DIR', log_dir)

    # 声明 use_sim_time 参数
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )
    stamp_outputs_with_node_time_arg = DeclareLaunchArgument(
        'stamp_outputs_with_node_time',
        default_value='false',
        description='Stamp localization TF and output topics with node time. Use true for rosbag playback when input header stamps are in a different time domain.'
    )

    # 配置文件路径
    config_file = os.path.join(open3d_loc_share, 'config', 'loc_param_g1.yaml')

    # 全局定位节点
    global_localization_node = Node(
        package='open3d_loc',
        executable='global_localization_node',
        name='global_localization_node',
        output='both',
        parameters=[
            config_file,
            {
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'stamp_outputs_with_node_time': LaunchConfiguration('stamp_outputs_with_node_time')
            }
        ]
    )

    return LaunchDescription([
        ros_log_dir,
        use_sim_time_arg,
        stamp_outputs_with_node_time_arg,
        global_localization_node,
    ])
