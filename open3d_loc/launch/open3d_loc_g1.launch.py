from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    # 获取包路径
    open3d_loc_share = FindPackageShare('open3d_loc')
    open3d_loc_source_dir = '/home/kangneng/xq/code/work_space/colcon_ws/src/FAST_LIO_LOCALIZATION_HUMANOID/open3d_loc'
    log_dir = os.path.join(open3d_loc_source_dir, 'log')
    os.makedirs(log_dir, exist_ok=True)

    ros_log_dir = SetEnvironmentVariable('ROS_LOG_DIR', log_dir)

    # 声明 use_sim_time 参数
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )

    # 配置文件路径
    config_file = PathJoinSubstitution([
        open3d_loc_share,
        'config',
        'loc_param_g1.yaml'
    ])

    # 地图文件路径 - 使用绝对路径指向源码目录中的地图文件
    map_file = '/home/kangneng/xq/pcd/map1-2/global_map_downsize_rotate.pcd'

    # 全局定位节点
    global_localization_node = Node(
        package='open3d_loc',
        executable='global_localization_node',
        name='global_localization_node',
        output='both',
        parameters=[
            config_file,
            {
                'path_map': map_file,
                'pcd_queue_maxsize': 10,
                'voxelsize_coarse': 0.2,
                'voxelsize_fine': 0.2,
                'voxel_downsample_size': 0.1,
                'icp_distance_threshold': 0.15,
                'fitness_eval_threshold': 0.10,
                'normal_search_radius': 0.4,
                'threshold_fitness': 0.5,
                'threshold_fitness_init': 0.5,
                'max_icp_translation': 0.3,
                'max_icp_yaw_deg': 1.0,
                'max_init_icp_translation': 2.0,
                'max_init_icp_yaw_deg': 15.0,
                'min_init_fitness_improvement': 0.02,
                'min_source_points': 500,
                'min_target_points': 20000,
                'loc_frequence': 2.5,
                'save_scan': False,
                'hidden_removal': False,
                'maxpoints_source': 80000,
                'maxpoints_target': 400000,
                'filter_odom2map': False,
                'kalman_processVar2': 0.001,
                'kalman_estimatedMeasVar2': 0.02,
                'confidence_loc_th': 0.7,
                'dis_updatemap': 3.5,
                'use_sim_time': LaunchConfiguration('use_sim_time')
            }
        ]
    )

    # 点云转换节点
    pointcloud_transformer_node = Node(
        package='open3d_loc',
        executable='pointcloud_transformer_node',
        name='pointcloud_transformer_node',
        output='screen',
        parameters=[{
            'input_topic': '/cloud_registered_body_1',
            'output_topic': '/cloud_registered_map',
            'global_map_topic': '/global_map',
            'source_frame': 'base_link',
            'target_frame': 'map',
            'voxel_leaf_size': 0.1,
            'map_voxel_leaf_size': 0.2,
            'max_global_points': 1000000,
            'map_publish_frequency': 1.0,
            'enable_global_map': True,
            'use_sim_time': LaunchConfiguration('use_sim_time')
        }]
    )

    return LaunchDescription([
        ros_log_dir,
        use_sim_time_arg,
        global_localization_node,
        # pointcloud_transformer_node
    ])
