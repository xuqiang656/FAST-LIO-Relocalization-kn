// #include <pcl/common/transforms.h>
#include "open3d_registration/open3d_registration.h"
#include "open3d_conversions/open3d_conversions.h"
#include "global_localization.h"

#define PI 3.1415926


GloabalLocalization::GloabalLocalization() : Node("global_loc_node"),
                                             tf_buffer_(this->get_clock()),
                                             tf_listener_(std::make_shared<tf2_ros::TransformListener>(tf_buffer_))
{

    flag_exit_.store(false);
    loc_initialized_.store(false);
    mat_baselink2odom_ = Eigen::Matrix4d::Identity();
    mat_odom2map_ = Eigen::Matrix4d::Identity();
    mat_odom2map_kalman_ = Eigen::Matrix4d::Identity();
    mat_baselink2map_ = Eigen::Matrix4d::Identity();
    mat_initialpose_ = Eigen::Matrix4d::Identity();
    mat_baselink2motionlink_ = Eigen::Matrix4d::Identity();
    mat_baselink2imulink_ = Eigen::Matrix4d::Identity();
    last_loc_ = Eigen::Vector3d(0, 0, -5000);

    pcd_map_ori_.reset(new open3d::geometry::PointCloud);
    pcd_scan_cur_.reset(new open3d::geometry::PointCloud);
    pcd_map_fine_.reset(new open3d::geometry::PointCloud);
    queue_maxsize_ = 5;

    pub_baselink2map_ = this->create_publisher<nav_msgs::msg::Odometry>("/baselink2map", 1);
    pub_baselink2map_kalman_ = this->create_publisher<nav_msgs::msg::Odometry>("/baselink2map_kalman", 1);
    pub_motionlink2map_ = this->create_publisher<nav_msgs::msg::Odometry>("/motionlink2map", 1);
    pub_odom2map_ = this->create_publisher<nav_msgs::msg::Odometry>("/odom2map", 1);
    pub_odom2map_kalman_ = this->create_publisher<nav_msgs::msg::Odometry>("/odom2map_kalman", 1);

    rclcpp::QoS map_qos(rclcpp::KeepLast(1));
    map_qos.reliable();
    map_qos.transient_local();
    pub_map_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/map", map_qos);
    pub_submap_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/submap", 1);
    pub_scan2map_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/scan2map", 1);
    pub_scan_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/scan", 1);
    pub_localization_3d_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/localization_3d", 1);
    pub_localization_3d_confidence_ = this->create_publisher<std_msgs::msg::Float32>("/localization_3d_confidence", 1);
    pub_localization_3d_delay_ms_ = this->create_publisher<std_msgs::msg::Float32>("/localization_3d_delay_ms", 1);

    loc_frequence_ = 2.0; //
    loc_fitness_.store(0.0);

    // 注册回调函数
    sub_baselink2odom_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/Odometry_loc", 50, std::bind(&GloabalLocalization::CallbackBaselink2Odom, this, std::placeholders::_1));
    sub_scan_cur_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/cloud_registered_1", 50, std::bind(&GloabalLocalization::CallbackScan, this, std::placeholders::_1));
    sub_initialpose_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/initialpose", 50, std::bind(&GloabalLocalization::CallbackInitialPose, this, std::placeholders::_1));

    pose_baselink2odom_ = nav_msgs::msg::Odometry();
    pose_baselink2odom_.header.frame_id = "odom";
    pose_baselink2odom_.child_frame_id = "base_link";
    
    // geometry_msgs的Quaternion会被初始化为0,0,0,0,而不是正确的0,0,0,1
    pose_baselink2odom_.pose.pose.orientation.w = 1;
    RCLCPP_INFO(this->get_logger(), "pose baselink2odom:\nx: %f, y: %f, z: %f, qx: %f, \
                            qy: %f, qz: %f, qw: %f",
                pose_baselink2odom_.pose.pose.position.x,
                pose_baselink2odom_.pose.pose.position.y,
                pose_baselink2odom_.pose.pose.position.z,
                pose_baselink2odom_.pose.pose.orientation.x,
                pose_baselink2odom_.pose.pose.orientation.y,
                pose_baselink2odom_.pose.pose.orientation.z,
                pose_baselink2odom_.pose.pose.orientation.w);

    // 队列最大数量
    this->declare_parameter<int>("pcd_queue_maxsize", 5);
    this->declare_parameter<bool>("save_scan", false);
    this->declare_parameter<std::string>("save_scan_dir", "/tmp/open3d_loc_scan_submap");
    /// 最大点数量限制
    this->declare_parameter<int>("maxpoints_source", 50000);
    this->declare_parameter<int>("maxpoints_target", 200000);

    // 定位间隔时间
    this->declare_parameter<double>("loc_frequence", 2.0);

    /// 定位阈值
    this->declare_parameter<double>("confidence_loc_th", 0.6);

    /// 卡尔曼参数
    this->declare_parameter<std::vector<double>>("kf_baselink2map/x", std::vector<double>(2));
    this->declare_parameter<std::vector<double>>("kf_baselink2map/y", std::vector<double>(2));
    this->declare_parameter<std::vector<double>>("kf_baselink2map/z", std::vector<double>(2));

    this->declare_parameter<bool>("filter_odom2map", false);
    this->declare_parameter<double>("kalman_processVar2", 0.02);
    this->declare_parameter<double>("kalman_estimatedMeasVar2", 0.04);
    // voxelsize
    this->declare_parameter<double>("voxelsize_coarse", 0.2);
    this->declare_parameter<double>("voxel_downsample_size", 0.1);
    this->declare_parameter<double>("icp_distance_threshold", 0.15);
    this->declare_parameter<double>("fitness_eval_threshold", 0.15);
    this->declare_parameter<double>("normal_search_radius", 0.4);
    this->declare_parameter<double>("max_icp_translation", 0.3);
    this->declare_parameter<double>("max_icp_yaw_deg", 1.0);
    this->declare_parameter<double>("max_init_icp_translation", 2.0);
    this->declare_parameter<double>("max_init_icp_yaw_deg", 15.0);
    this->declare_parameter<double>("min_init_fitness_improvement", 0.02);
    this->declare_parameter<int>("min_source_points", 2500);
    this->declare_parameter<int>("min_target_points", 50000);
    this->declare_parameter<double>("threshold_fitness_init", 0.9);
    this->declare_parameter<double>("threshold_fitness", 0.9);
    this->declare_parameter<std::vector<double>>("initialpose", std::vector<double>());
    this->declare_parameter<double>("dis_updatemap", 1);
    this->declare_parameter<double>("map_publish_interval", 2.0);

    this->get_parameter("pcd_queue_maxsize", queue_maxsize_);
    this->get_parameter("save_scan", save_scan_);
    this->get_parameter("save_scan_dir", save_scan_dir_);
    if (queue_maxsize_ < 1)
    {
        RCLCPP_WARN(this->get_logger(), "pcd_queue_maxsize=%d is invalid, use 1", queue_maxsize_);
        queue_maxsize_ = 1;
    }
    this->get_parameter("maxpoints_source", maxpoints_source_);
    this->get_parameter("maxpoints_target", maxpoints_target_);
    this->get_parameter("loc_frequence", loc_frequence_);
    this->get_parameter("confidence_loc_th", confidence_loc_th_);
    this->get_parameter("kf_baselink2map/x", kf_param_x_);
    this->get_parameter("kf_baselink2map/y", kf_param_y_);
    this->get_parameter("kf_baselink2map/z", kf_param_z_);
    this->get_parameter("filter_odom2map", filter_odom2map_);
    this->get_parameter("kalman_processVar2", kalman_processVar2_);
    this->get_parameter("kalman_estimatedMeasVar2", kalman_estimatedMeasVar2_);

    // RCLCPP_INFO(this->get_logger(), "Kalman filter parameters:");
    // RCLCPP_INFO(this->get_logger(), "  kf_x: [%.6f, %.6f], size: %zu",
    //             kf_param_x_.size() >= 1 ? kf_param_x_[0] : 0.0,
    //             kf_param_x_.size() >= 2 ? kf_param_x_[1] : 0.0,
    //             kf_param_x_.size());
    // RCLCPP_INFO(this->get_logger(), "  kf_y: [%.6f, %.6f], size: %zu",
    //             kf_param_y_.size() >= 1 ? kf_param_y_[0] : 0.0,
    //             kf_param_y_.size() >= 2 ? kf_param_y_[1] : 0.0,
    //             kf_param_y_.size());
    // RCLCPP_INFO(this->get_logger(), "  kf_z: [%.6f, %.6f], size: %zu",
    //             kf_param_z_.size() >= 1 ? kf_param_z_[0] : 0.0,
    //             kf_param_z_.size() >= 2 ? kf_param_z_[1] : 0.0,
    //             kf_param_z_.size());
    // RCLCPP_INFO(this->get_logger(), "  filter_odom2map: %s", filter_odom2map_ ? "true" : "false");
    this->get_parameter("voxelsize_coarse", voxelsize_coarse_);
    this->get_parameter("voxel_downsample_size", voxel_downsample_size_);
    this->get_parameter("icp_distance_threshold", icp_distance_threshold_);
    this->get_parameter("fitness_eval_threshold", fitness_eval_threshold_);
    this->get_parameter("normal_search_radius", normal_search_radius_);
    this->get_parameter("max_icp_translation", max_icp_translation_);
    this->get_parameter("max_icp_yaw_deg", max_icp_yaw_deg_);
    this->get_parameter("max_init_icp_translation", max_init_icp_translation_);
    this->get_parameter("max_init_icp_yaw_deg", max_init_icp_yaw_deg_);
    this->get_parameter("min_init_fitness_improvement", min_init_fitness_improvement_);
    this->get_parameter("min_source_points", min_source_points_);
    this->get_parameter("min_target_points", min_target_points_);
    this->get_parameter("threshold_fitness_init", threshold_fitness_init_);
    this->get_parameter("threshold_fitness", threshold_fitness_);
    this->get_parameter("initialpose", initialpose_);
    this->get_parameter("dis_updatemap", dis_updatemap_);
    double map_publish_interval = 2.0;
    this->get_parameter("map_publish_interval", map_publish_interval);

    RCLCPP_INFO(this->get_logger(),
                "registration params: voxelsize_coarse=%.3f, voxel_downsample_size=%.3f, "
                "icp_distance_threshold=%.3f, fitness_eval_threshold=%.3f, "
                "normal_search_radius=%.3f, threshold_fitness=%.3f, threshold_fitness_init=%.3f, "
                "max_icp_translation=%.3f, max_icp_yaw_deg=%.3f, max_init_icp_translation=%.3f, "
                "max_init_icp_yaw_deg=%.3f, min_init_fitness_improvement=%.3f, min_source_points=%d, min_target_points=%d, "
                "maxpoints_source=%d, maxpoints_target=%d",
                voxelsize_coarse_, voxel_downsample_size_, icp_distance_threshold_,
                fitness_eval_threshold_, normal_search_radius_, threshold_fitness_, threshold_fitness_init_,
                max_icp_translation_, max_icp_yaw_deg_, max_init_icp_translation_, max_init_icp_yaw_deg_,
                min_init_fitness_improvement_, min_source_points_, min_target_points_,
                maxpoints_source_, maxpoints_target_);

    if (initialpose_.size() != 6)
    {
        RCLCPP_WARN(this->get_logger(),
                    "invalid initialpose parameter size=%zu, expected 6 values [x,y,z,roll,pitch,yaw], use identity",
                    initialpose_.size());
        initialpose_ = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    }
    RCLCPP_INFO(this->get_logger(),
                "initialpose param: xyz=(%.3f, %.3f, %.3f), rpy_deg=(%.3f, %.3f, %.3f)",
                initialpose_[0], initialpose_[1], initialpose_[2],
                initialpose_[3], initialpose_[4], initialpose_[5]);
    mat_initialpose_.block<3, 3>(0, 0) = Euler2Matrix3d(Eigen::Vector3d(initialpose_[3], initialpose_[4], initialpose_[5]));
    mat_initialpose_.block<3, 1>(0, 3) = Eigen::Vector3d(initialpose_[0], initialpose_[1], initialpose_[2]);

    // 读取地图
    RCLCPP_INFO(this->get_logger(), "开始读取点云地图");
    std::string path_map = "";
    this->declare_parameter<std::string>("path_map", "");
    this->get_parameter("path_map", path_map);
    open3d::io::ReadPointCloud(path_map, *pcd_map_ori_);
    if (pcd_map_ori_ == nullptr || pcd_map_ori_->IsEmpty())
    {
        RCLCPP_ERROR(this->get_logger(), "read map from path: %s failed", path_map.c_str());
        rclcpp::shutdown();
    }

    auto pcd_map_coarse = pcd_map_ori_->VoxelDownSample(voxelsize_coarse_);
    pcd_map_coarse->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(voxelsize_coarse_ * 2, 30));
    if (!pcd_map_coarse->HasColors())
    {
        pcd_map_coarse->PaintUniformColor({1, 0, 0});
    }
    map_points_count_ = pcd_map_coarse->points_.size();

    /// publish map, 用粗地图可视化，减少资源占用
    open3d_conversions::open3dToRos(*pcd_map_coarse, map_msg_);
    map_msg_.header.frame_id = "map";
    map_msg_.header.stamp = this->now();
    pub_map_->publish(map_msg_);
    if (map_publish_interval > 0.0)
    {
        auto map_publish_period = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::duration<double>(map_publish_interval));
        map_publish_timer_ = this->create_wall_timer(
            map_publish_period,
            [this]()
            {
                map_msg_.header.stamp = this->now();
                pub_map_->publish(map_msg_);
                RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 10000,
                                     "republish static map: points=%zu, frame_id=%s",
                                     map_points_count_, map_msg_.header.frame_id.c_str());
            });
        RCLCPP_INFO(this->get_logger(),
                    "map publisher uses transient_local QoS and republish interval %.3f s",
                    map_publish_interval);
    }

    pcd_map_fine_ = pcd_map_ori_->VoxelDownSample(voxel_downsample_size_);
    pcd_map_fine_->colors_.clear();
    pcd_map_fine_->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(normal_search_radius_, 30));
    pcd_map_ori_.reset();

    static_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
    
    auto publish_static_tf_from_param =
        [this](const std::string &param_name,
               const std::string &parent_frame,
               const std::string &child_frame,
               Eigen::Matrix4d *matrix_out)
    {
        const std::vector<double> default_tf = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0};
        this->declare_parameter<std::vector<double>>(param_name, default_tf);
        std::vector<double> tf_param;
        this->get_parameter(param_name, tf_param);

        bool valid = tf_param.size() == 7;
        for (double value : tf_param)
        {
            valid = valid && std::isfinite(value);
        }
        if (!valid)
        {
            RCLCPP_WARN(this->get_logger(),
                        "invalid static tf param %s, expected [x,y,z,qx,qy,qz,qw], use identity",
                        param_name.c_str());
            tf_param = default_tf;
        }

        Eigen::Quaterniond quat(tf_param[6], tf_param[3], tf_param[4], tf_param[5]);
        if (!std::isfinite(quat.norm()) || quat.norm() < 1e-6)
        {
            RCLCPP_WARN(this->get_logger(),
                        "invalid quaternion in static tf param %s, use identity rotation",
                        param_name.c_str());
            quat = Eigen::Quaterniond::Identity();
        }
        else
        {
            quat.normalize();
        }

        Eigen::Matrix4d matrix = Eigen::Matrix4d::Identity();
        matrix.block<3, 3>(0, 0) = quat.toRotationMatrix();
        matrix.block<3, 1>(0, 3) = Eigen::Vector3d(tf_param[0], tf_param[1], tf_param[2]);
        if (matrix_out != nullptr)
        {
            *matrix_out = matrix;
        }

        geometry_msgs::msg::TransformStamped transform;
        transform.header.stamp = this->now();
        transform.header.frame_id = parent_frame;
        transform.child_frame_id = child_frame;
        transform.transform.translation.x = tf_param[0];
        transform.transform.translation.y = tf_param[1];
        transform.transform.translation.z = tf_param[2];
        transform.transform.rotation.x = quat.x();
        transform.transform.rotation.y = quat.y();
        transform.transform.rotation.z = quat.z();
        transform.transform.rotation.w = quat.w();
        static_broadcaster_->sendTransform(transform);

        RCLCPP_INFO(this->get_logger(),
                    "publish static tf %s -> %s from %s: xyz=(%.3f, %.3f, %.3f), quat=(%.6f, %.6f, %.6f, %.6f)",
                    parent_frame.c_str(), child_frame.c_str(), param_name.c_str(),
                    tf_param[0], tf_param[1], tf_param[2], quat.x(), quat.y(), quat.z(), quat.w());
    };

    publish_static_tf_from_param("static_tf_base_link_to_imu_link", "base_link", "imu_link", &mat_baselink2imulink_);
    publish_static_tf_from_param("static_tf_base_link_to_motion_link", "base_link", "motion_link", &mat_baselink2motionlink_);

    RCLCPP_WARN(this->get_logger(), "initialize finished");

    br_odom2map_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

    StartLoc();
}

GloabalLocalization::~GloabalLocalization()
{
    flag_exit_.store(true);
    if (thread_loc_.joinable())
    {
        thread_loc_.join();
    }
}

Eigen::Matrix3d GloabalLocalization::Euler2Matrix3d(const Eigen::Vector3d euler)
{
    Eigen::Matrix3d mat3d;
    // convert degrees to radians
    auto eulerAngle = euler / 180 * M_PI;
    Eigen::AngleAxisd rollAngle(Eigen::AngleAxisd(eulerAngle[0], Eigen::Vector3d::UnitX()));
    Eigen::AngleAxisd pitchAngle(Eigen::AngleAxisd(eulerAngle[1], Eigen::Vector3d::UnitY()));
    Eigen::AngleAxisd yawAngle(Eigen::AngleAxisd(eulerAngle[2], Eigen::Vector3d::UnitZ()));
    mat3d = rollAngle * pitchAngle * yawAngle;
    return mat3d;
}
bool GloabalLocalization::GetTfTransformToMatrix(std::string frame_id, std::string child_frame_id, Eigen::Matrix4d &matrix)
{
    // 获取pose
    geometry_msgs::msg::TransformStamped pose_;
    try
    {
        pose_ = tf_buffer_.lookupTransform(frame_id, child_frame_id, rclcpp::Time(0));
    }
    catch (tf2::TransformException &e)
    {
        RCLCPP_ERROR(this->get_logger(), "[GetTransformMatrix]: %s", e.what());
        return false;
    }

    Eigen::Vector3d translation = Eigen::Vector3d(pose_.transform.translation.x, pose_.transform.translation.y, pose_.transform.translation.z);
    Eigen::Quaterniond quat = Eigen::Quaterniond::Identity();

    quat = Eigen::Quaterniond(pose_.transform.rotation.w,
                              pose_.transform.rotation.x,
                              pose_.transform.rotation.y,
                              pose_.transform.rotation.z);
    Eigen::Matrix3d rotation = quat.matrix();

    matrix = Eigen::Matrix4d::Identity();
    matrix.block<3, 3>(0, 0) = rotation;
    matrix.matrix().block<3, 1>(0, 3) = translation;
    return true;
}

void GloabalLocalization::CallbackBaselink2Odom(const nav_msgs::msg::Odometry::SharedPtr baselink2odom)
{
    {
        std::lock_guard<std::mutex> timestamp_lock(lock_timestamp_);
        timestamp_odom_ = baselink2odom->header.stamp;
    }
    Eigen::Isometry3d mat_current = Eigen::Isometry3d::Identity();
    tf2::fromMsg(baselink2odom->pose.pose, mat_current);
    auto mat_imulink2odom = mat_current.matrix();

    Eigen::Matrix4d mat_odom2map_snapshot = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d mat_baselink2odom_snapshot = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d mat_baselink2map_snapshot = Eigen::Matrix4d::Identity();
    {
        std::lock_guard<std::mutex> state_lock(lock_mat_odom2map_);
        mat_baselink2odom_ = mat_imulink2odom * mat_baselink2imulink_.inverse();
        mat_baselink2map_ = mat_odom2map_ * mat_baselink2odom_;
        mat_odom2map_snapshot = mat_odom2map_;
        mat_baselink2odom_snapshot = mat_baselink2odom_;
        mat_baselink2map_snapshot = mat_baselink2map_;
    }

    Eigen::Isometry3d Isometry3d_baselink2map;
    Isometry3d_baselink2map.matrix() = mat_baselink2map_snapshot;
    nav_msgs::msg::Odometry baselink2map;
    baselink2map.pose.pose = tf2::toMsg(Isometry3d_baselink2map);
    baselink2map.header.frame_id = "map";
    baselink2map.child_frame_id = "base_link";
    baselink2map.header.stamp = baselink2odom->header.stamp;
    pub_baselink2map_->publish(baselink2map);

    Eigen::Isometry3d Isometry3d_odom2map;
    Isometry3d_odom2map.matrix() = mat_odom2map_snapshot;
    nav_msgs::msg::Odometry odom2map;
    odom2map.pose.pose = tf2::toMsg(Isometry3d_odom2map);
    odom2map.header.frame_id = "map";
    odom2map.child_frame_id = "odom";
    odom2map.header.stamp = baselink2odom->header.stamp;
    pub_odom2map_->publish(odom2map);

    /// 发布tf关系
    geometry_msgs::msg::TransformStamped transform_odom2map;
    transform_odom2map.header.frame_id = "map";
    transform_odom2map.child_frame_id = "odom";
    transform_odom2map.header.stamp = baselink2odom->header.stamp;
    transform_odom2map.transform.translation.x = odom2map.pose.pose.position.x;
    transform_odom2map.transform.translation.y = odom2map.pose.pose.position.y;
    transform_odom2map.transform.translation.z = odom2map.pose.pose.position.z;
    transform_odom2map.transform.rotation = odom2map.pose.pose.orientation;
    br_odom2map_->sendTransform(transform_odom2map);

    geometry_msgs::msg::TransformStamped transform_baselink2odom;
    transform_baselink2odom.header.frame_id = "odom";
    transform_baselink2odom.child_frame_id = "base_link";
    transform_baselink2odom.header.stamp = baselink2odom->header.stamp;
    transform_baselink2odom.transform.translation.x = mat_baselink2odom_snapshot(0, 3);
    transform_baselink2odom.transform.translation.y = mat_baselink2odom_snapshot(1, 3);
    transform_baselink2odom.transform.translation.z = mat_baselink2odom_snapshot(2, 3);
    Eigen::Quaterniond quat_baselink2odom(mat_baselink2odom_snapshot.block<3, 3>(0, 0));
    quat_baselink2odom.normalize();
    transform_baselink2odom.transform.rotation.x = quat_baselink2odom.x();
    transform_baselink2odom.transform.rotation.y = quat_baselink2odom.y();
    transform_baselink2odom.transform.rotation.z = quat_baselink2odom.z();
    transform_baselink2odom.transform.rotation.w = quat_baselink2odom.w();
    br_odom2map_->sendTransform(transform_baselink2odom);

    /// 卡尔曼滤波 - 只在定位初始化完成后执行
    if (loc_initialized_.load())
    {
        Eigen::Matrix4d mat_baselink2map_kalman = Eigen::Matrix4d::Identity();

        if (filter_odom2map_)
        {
            Eigen::Matrix4d mat_odom2map_kalman_snapshot = Eigen::Matrix4d::Identity();
            {
                std::lock_guard<std::mutex> state_lock(lock_mat_odom2map_);
                mat_odom2map_kalman_snapshot = mat_odom2map_kalman_;
            }
            Eigen::Isometry3d Isometry3d_odom2map_kalman;
            Isometry3d_odom2map_kalman.matrix() = mat_odom2map_kalman_snapshot;
            nav_msgs::msg::Odometry odom2map_kalman;
            odom2map_kalman.pose.pose = tf2::toMsg(Isometry3d_odom2map_kalman);
            odom2map_kalman.header.frame_id = "map";
            odom2map_kalman.child_frame_id = "odom_kalman";
            odom2map_kalman.header.stamp = baselink2odom->header.stamp;
            pub_odom2map_kalman_->publish(odom2map_kalman);

            kf_baselink_z_.inputLatestNoisyMeasurement((mat_odom2map_kalman_snapshot * mat_baselink2odom_snapshot)(2, 3));
            mat_baselink2map_kalman = mat_odom2map_kalman_snapshot * mat_baselink2odom_snapshot;
        }
        else
        {
            double input_x = mat_baselink2map_snapshot(0, 3);
            double input_y = mat_baselink2map_snapshot(1, 3);
            double input_z = mat_baselink2map_snapshot(2, 3);

            kf_baselink_x_.inputLatestNoisyMeasurement(input_x);
            kf_baselink_y_.inputLatestNoisyMeasurement(input_y);
            kf_baselink_z_.inputLatestNoisyMeasurement(input_z);
            mat_baselink2map_kalman = mat_baselink2map_snapshot;

            RCLCPP_DEBUG(this->get_logger(), "KF input: x=%.3f, y=%.3f, z=%.3f", input_x, input_y, input_z);
        }

        double filtered_z = kf_baselink_z_.getLatestEstimatedMeasurement();

        // 验证结果是否有效（检查 NaN）
        if (std::isnan(filtered_z))
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                 "Kalman filter returned NaN (input was: %.3f), using unfiltered value",
                                 mat_baselink2map_kalman(2, 3));
            mat_baselink2map_kalman(2, 3) = mat_baselink2map_snapshot(2, 3);
        }
        else
        {
            mat_baselink2map_kalman(2, 3) = filtered_z;
        }
        Eigen::Isometry3d Isometry3d_baselink2map_kalman;
        Isometry3d_baselink2map_kalman.matrix() = mat_baselink2map_kalman;
        nav_msgs::msg::Odometry baselink2map_kalman;
        baselink2map_kalman.pose.pose = tf2::toMsg(Isometry3d_baselink2map_kalman);
        baselink2map_kalman.header.frame_id = "map";
        // baselink2map_kalman.child_frame_id = "base_link_kalman";
        baselink2map_kalman.header.stamp = baselink2odom->header.stamp;
        pub_baselink2map_kalman_->publish(baselink2map_kalman);

        Eigen::Matrix4d mat_motionlink2map = mat_baselink2map_kalman * mat_baselink2motionlink_;
        Eigen::Isometry3d Isometry3d_motionlink2map;
        Isometry3d_motionlink2map.matrix() = mat_motionlink2map;
        nav_msgs::msg::Odometry motionlink2map;
        motionlink2map.pose.pose = tf2::toMsg(Isometry3d_motionlink2map);
        motionlink2map.header.frame_id = "map";
        // baselink2map_kalman.child_frame_id = "base_link_kalman";
        motionlink2map.header.stamp = baselink2odom->header.stamp;
        pub_motionlink2map_->publish(motionlink2map);

        localization_3d_confidence_.data = static_cast<float>(loc_fitness_.load());
        pub_localization_3d_confidence_->publish(localization_3d_confidence_);
        localization_3d_delay_ms_.data = (this->now() - baselink2odom->header.stamp).seconds() * 1000.0;
        pub_localization_3d_delay_ms_->publish(localization_3d_delay_ms_);
        localization_3d_.header.frame_id = "map";
        localization_3d_.header.stamp = baselink2odom->header.stamp;
        localization_3d_.pose = motionlink2map.pose.pose;
        pub_localization_3d_->publish(localization_3d_);
    }
}
void GloabalLocalization::CallbackScan(
    const sensor_msgs::msg::PointCloud2::SharedPtr scan_in_baselink)
{
    auto pcd_received = std::make_shared<open3d::geometry::PointCloud>();
    // 单帧转换为open3d，几百us
    sensor_msgs::msg::PointCloud2::ConstSharedPtr const_scan_ptr = scan_in_baselink;
    open3d_conversions::rosToOpen3d(const_scan_ptr, *pcd_received, true);

    std::vector<std::shared_ptr<open3d::geometry::PointCloud>> scan_window;
    {
        std::lock_guard<std::mutex> scan_lock(lock_scan_);
        que_pcd_scan_.push_back(pcd_received);
        while (que_pcd_scan_.size() > static_cast<size_t>(queue_maxsize_))
        {
            que_pcd_scan_.pop_front();
        }

        if (que_pcd_scan_.size() >= static_cast<size_t>(queue_maxsize_))
        {
            scan_window.assign(que_pcd_scan_.begin(), que_pcd_scan_.end());
        }
    }

    if (!scan_window.empty())
    {
        auto combined_scan = std::make_shared<open3d::geometry::PointCloud>();
        for (const auto &scan : scan_window)
        {
            *combined_scan += *scan;
        }
        std::lock_guard<std::mutex> scan_lock(lock_scan_);
        pcd_scan_cur_ = combined_scan;
    }
}

bool GloabalLocalization::LocalizationInitialize()
{
    /// 裁剪后的地图
    std::shared_ptr<open3d::geometry::PointCloud> map_fine_crop(new open3d::geometry::PointCloud);

    /// 当前环境感知子图点云
    std::shared_ptr<open3d::geometry::PointCloud> pcd_scan(new open3d::geometry::PointCloud);
    /// 环境感知子图转换到地图坐标系
    std::shared_ptr<open3d::geometry::PointCloud> pcd_scan2map(new open3d::geometry::PointCloud);

    /// 用于配准的source target
    std::shared_ptr<open3d::geometry::PointCloud> source(new open3d::geometry::PointCloud);
    std::shared_ptr<open3d::geometry::PointCloud> target(new open3d::geometry::PointCloud);

    /// cropbox,用于裁剪地图和当前环境感知子图
    std::shared_ptr<open3d::geometry::OrientedBoundingBox> OBB_map(new open3d::geometry::OrientedBoundingBox);
    std::shared_ptr<open3d::geometry::OrientedBoundingBox> OBB_scan(new open3d::geometry::OrientedBoundingBox);

    /// 当前baselink到odom和map坐标系的关系
    Eigen::Matrix4d mat_baselink2odom_cur = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d mat_baselink2map_cur = Eigen::Matrix4d::Identity();

    /// 固定感知子图/历史地图子图大小
    OBB_map->extent_ = Eigen::Vector3d(60, 60, 40);
    OBB_map->color_ = Eigen::Vector3d(1, 0.5, 0);
    OBB_scan->extent_ = Eigen::Vector3d(60, 60, 40);
    OBB_scan->color_ = Eigen::Vector3d(0, 1, 0);

    double fitness_initial; /// overlap
    double loc_cost = 0;    /// 定位耗时(ms)
    int count_success = 0;
    bool init_success = false;
    while (rclcpp::ok() && !flag_exit_.load())
    {
        auto loc_s = std::chrono::high_resolution_clock::now(); /// 开始定位计时
        std::shared_ptr<open3d::geometry::PointCloud> scan_snapshot;
        {
            std::lock_guard<std::mutex> scan_lock(lock_scan_);
            scan_snapshot = pcd_scan_cur_;
        }
        if (scan_snapshot == nullptr || scan_snapshot->IsEmpty())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            continue;
        }
        else
        {
            /// 获取最新关系
            Eigen::Matrix4d reg_matrix = Eigen::Matrix4d::Identity();
            {
                std::lock_guard<std::mutex> state_lock(lock_mat_odom2map_);
                mat_baselink2odom_cur = mat_baselink2odom_;
                reg_matrix = mat_odom2map_;
            }
            *pcd_scan = *scan_snapshot;
            mat_baselink2map_cur = reg_matrix * mat_baselink2odom_cur;

            /// 将cropbox转换到对应位置进行裁剪点云
            OBB_map->center_ = mat_baselink2map_cur.block<3, 1>(0, 3);
            OBB_map->R_ = mat_baselink2map_cur.block<3, 3>(0, 0);
            OBB_scan->center_ = mat_baselink2odom_cur.block<3, 1>(0, 3);
            OBB_scan->R_ = mat_baselink2odom_cur.block<3, 3>(0, 0);
            *map_fine_crop = *pcd_map_fine_->Crop(*OBB_map);

            /// 配准计时
            target = map_fine_crop;
            const size_t target_before_sample_size = target->points_.size();
            if (target->points_.size() > static_cast<size_t>(maxpoints_target_))
            {
                target = target->RandomDownSample(double(maxpoints_target_) / target->points_.size());
            }
            const size_t target_after_sample_size = target->points_.size();

            source = pcd_scan->Crop(*OBB_scan);
            const size_t source_before_voxel_size = source->points_.size();
            source = source->VoxelDownSample(voxel_downsample_size_);
            const size_t source_after_voxel_size = source->points_.size();
            if (source->points_.size() > static_cast<size_t>(maxpoints_source_))
            {
                source = source->RandomDownSample(double(maxpoints_source_) / source->points_.size());
            }
            RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                  "init preprocess: target=%zu->%zu, source=%zu->%zu->%zu, target_has_normal=%s, source_has_normal=%s",
                                  target_before_sample_size, target_after_sample_size,
                                  source_before_voxel_size, source_after_voxel_size, source->points_.size(),
                                  target->HasNormals() ? "true" : "false", source->HasNormals() ? "true" : "false");

            if (source->points_.size() < static_cast<size_t>(min_source_points_) ||
                target->points_.size() < static_cast<size_t>(min_target_points_))
            {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                     "skip init icp: source=%zu (min=%d), target=%zu (min=%d), map_center=(%.3f, %.3f, %.3f), scan_center=(%.3f, %.3f, %.3f)",
                                     source->points_.size(), min_source_points_, target->points_.size(), min_target_points_,
                                     OBB_map->center_.x(), OBB_map->center_.y(), OBB_map->center_.z(),
                                     OBB_scan->center_.x(), OBB_scan->center_.y(), OBB_scan->center_.z());
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                continue;
            }

            source->Transform(reg_matrix);
            *pcd_scan2map = *source;
            auto eva_before_icp = open3d::pipelines::registration::EvaluateRegistration(*source, *target, fitness_eval_threshold_);
            RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                  "init before icp: eva_fitness=%f, inlier_rmse=%f, eval_threshold=%.3f",
                                  eva_before_icp.fitness_, eva_before_icp.inlier_rmse_, fitness_eval_threshold_);

            auto multiScale_reg_matrix = pcd_tools::RegistrationMultiScaleIcp(source, target, voxel_downsample_size_, 1, {1, 2, 4});
            reg_matrix = multiScale_reg_matrix * reg_matrix;
            source->Transform(multiScale_reg_matrix);
            auto eva_result_coarse = open3d::pipelines::registration::EvaluateRegistration(*source, *target, fitness_eval_threshold_);
            double init_delta_trans = multiScale_reg_matrix.block<3, 1>(0, 3).norm();
            double init_delta_yaw = std::atan2(multiScale_reg_matrix(1, 0), multiScale_reg_matrix(0, 0)) * 180.0 / M_PI;
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "init icp result: eva_before=%f, eva_after=%f, inlier_rmse=%f, delta_trans=%.3f, delta_yaw_deg=%.3f, threshold_fitness_init=%.3f, max_init_icp_translation=%.3f, max_init_icp_yaw_deg=%.3f",
                                 eva_before_icp.fitness_, eva_result_coarse.fitness_, eva_result_coarse.inlier_rmse_,
                                 init_delta_trans, init_delta_yaw, threshold_fitness_init_, max_init_icp_translation_, max_init_icp_yaw_deg_);
            fitness_initial = eva_result_coarse.fitness_;
            *pcd_scan2map = *source;

            bool safe_init_step = init_delta_trans <= max_init_icp_translation_ &&
                                  std::abs(init_delta_yaw) <= max_init_icp_yaw_deg_;
            bool init_fitness_improved = fitness_initial > eva_before_icp.fitness_ + min_init_fitness_improvement_;
            bool accept_init = fitness_initial > threshold_fitness_init_ && safe_init_step;
            bool update_init_candidate = safe_init_step && (accept_init || init_fitness_improved);

            if (update_init_candidate)
            {
                {
                    std::lock_guard<std::mutex> state_lock(lock_mat_odom2map_);
                    mat_odom2map_ = reg_matrix;
                }
                RCLCPP_INFO(this->get_logger(),
                            "update init candidate: eva_before=%f, eva_after=%f, improvement=%f, success=%s, odom2map_xyz=(%.3f, %.3f, %.3f)",
                            eva_before_icp.fitness_, fitness_initial, fitness_initial - eva_before_icp.fitness_,
                            accept_init ? "true" : "false", reg_matrix(0, 3), reg_matrix(1, 3), reg_matrix(2, 3));
            }
            else
            {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                     "reject init icp: eva_before=%f, eva_after=%f, threshold=%.3f, improvement=%f, min_improvement=%.3f, delta_trans=%.3f, max_delta=%.3f, delta_yaw_deg=%.3f, max_yaw_deg=%.3f, source=%zu, target=%zu",
                                     eva_before_icp.fitness_, fitness_initial, threshold_fitness_init_,
                                     fitness_initial - eva_before_icp.fitness_, min_init_fitness_improvement_,
                                     init_delta_trans, max_init_icp_translation_, init_delta_yaw, max_init_icp_yaw_deg_,
                                     source->points_.size(), target->points_.size());
            }
            auto loc_e = std::chrono::high_resolution_clock::now(); /// 结束定位计时
            loc_cost = std::chrono::duration_cast<std::chrono::microseconds>(loc_e - loc_s).count() / 1000.0;
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                 "init localization cost: %.3f ms", loc_cost);

            if (accept_init)
            {
                count_success += 1;
                /// 连续两次定位成功后定位初始化成功
                if (count_success >= 2)
                {
                    init_success = true;
                    break;
                }
            }
            else
            {
                count_success = 0;
            }
        }
    }

    if (!init_success)
    {
        RCLCPP_WARN(this->get_logger(), "localization initialize stopped before success");
        return false;
    }

    RCLCPP_INFO(this->get_logger(), "localization initialize success");
    return true;
}
void GloabalLocalization::Localization()
{
    RCLCPP_INFO(this->get_logger(), "wait for Odometry_loc");
    // 等待接收到第一条里程计消息（通过检查timestamp是否有效）
    while (rclcpp::ok() && !flag_exit_.load())
    {
        {
            std::lock_guard<std::mutex> timestamp_lock(lock_timestamp_);
            if (timestamp_odom_.seconds() != 0.0)
            {
                break;
            }
        }
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Waiting for Odometry_loc...");
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    RCLCPP_INFO(this->get_logger(), "Received Odometry_loc");

    RCLCPP_INFO(this->get_logger(), "wait for cloud_registered_1");
    // 等待接收到第一条点云消息（通过检查pcd_scan_cur_是否为空）
    while (rclcpp::ok() && !flag_exit_.load())
    {
        bool has_scan = false;
        {
            std::lock_guard<std::mutex> scan_lock(lock_scan_);
            has_scan = pcd_scan_cur_ != nullptr && !pcd_scan_cur_->IsEmpty();
        }
        if (has_scan)
        {
            break;
        }
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Waiting for cloud_registered_1...");
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    RCLCPP_INFO(this->get_logger(), "Received cloud_registered_1");
    if (flag_exit_.load())
    {
        return;
    }

    // initialize
    /****初始化定位****/
    Eigen::Matrix4d mat_baselink2odom_init = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d mat_odom2map_init = Eigen::Matrix4d::Identity();
    {
        std::lock_guard<std::mutex> state_lock(lock_mat_odom2map_);
        mat_odom2map_ = mat_initialpose_ * mat_baselink2odom_.inverse(); /// initialpose 表示 base_link 在 map 下的位姿
        mat_baselink2odom_init = mat_baselink2odom_;
        mat_odom2map_init = mat_odom2map_;
    }
    RCLCPP_INFO(this->get_logger(),
                "initial odom2map from initialpose and current odom: initial_xyz=(%.3f, %.3f, %.3f), odom_xyz=(%.3f, %.3f, %.3f), odom2map_xyz=(%.3f, %.3f, %.3f)",
                mat_initialpose_(0, 3), mat_initialpose_(1, 3), mat_initialpose_(2, 3),
                mat_baselink2odom_init(0, 3), mat_baselink2odom_init(1, 3), mat_baselink2odom_init(2, 3),
                mat_odom2map_init(0, 3), mat_odom2map_init(1, 3), mat_odom2map_init(2, 3));
    if (!LocalizationInitialize())
    {
        return;
    }

    /// 卡尔曼滤波初始化
    /// 使用当前 baselink2map 位置初始化卡尔曼滤波器
    Eigen::Matrix4d init_baselink2map = Eigen::Matrix4d::Identity();
    {
        std::lock_guard<std::mutex> state_lock(lock_mat_odom2map_);
        init_baselink2map = mat_odom2map_ * mat_baselink2odom_;
    }
    double init_x = init_baselink2map(0, 3);
    double init_y = init_baselink2map(1, 3);
    double init_z = init_baselink2map(2, 3);

    RCLCPP_INFO(this->get_logger(), "Initializing Kalman filters with position: x=%.3f, y=%.3f, z=%.3f",
                init_x, init_y, init_z);

    // 检查参数数组大小是否有效
    if (kf_param_x_.size() >= 2 && kf_param_y_.size() >= 2 && kf_param_z_.size() >= 2)
    {
        kf_baselink_x_.KalmanFilterInit(kf_param_x_[0], kf_param_x_[1], init_x, 1);
        kf_baselink_y_.KalmanFilterInit(kf_param_y_[0], kf_param_y_[1], init_y, 1);
        kf_baselink_z_.KalmanFilterInit(kf_param_z_[0], kf_param_z_[1], init_z, 1);
        RCLCPP_INFO(this->get_logger(), "Kalman filters initialized: x[%.6f,%.6f], y[%.6f,%.6f], z[%.6f,%.6f]",
                    kf_param_x_[0], kf_param_x_[1], kf_param_y_[0], kf_param_y_[1],
                    kf_param_z_[0], kf_param_z_[1]);
    }
    else
    {
        RCLCPP_ERROR(this->get_logger(), "Invalid Kalman filter parameters! x_size=%zu, y_size=%zu, z_size=%zu",
                     kf_param_x_.size(), kf_param_y_.size(), kf_param_z_.size());
        RCLCPP_ERROR(this->get_logger(), "Kalman filters will NOT be initialized - using default values");
    }

    kalman_filter_odom2map_.KalmanFilterInit(kalman_processVar2_, kalman_estimatedMeasVar2_, init_z, 1);

    loc_initialized_.store(true); /// 初始化成功

    RCLCPP_INFO(this->get_logger(), "Localization initialization complete, Kalman filters ready");

    std::shared_ptr<open3d::geometry::PointCloud> pcd_scan(new open3d::geometry::PointCloud);
    std::shared_ptr<open3d::geometry::PointCloud> pcd_scan2map(new open3d::geometry::PointCloud);
    std::shared_ptr<open3d::geometry::PointCloud> source(new open3d::geometry::PointCloud);
    std::shared_ptr<open3d::geometry::PointCloud> target(new open3d::geometry::PointCloud);
    std::shared_ptr<open3d::geometry::PointCloud> map_fine_crop(new open3d::geometry::PointCloud);
    std::shared_ptr<open3d::geometry::OrientedBoundingBox> OBB_map(new open3d::geometry::OrientedBoundingBox);
    std::shared_ptr<open3d::geometry::OrientedBoundingBox> OBB_scan(new open3d::geometry::OrientedBoundingBox);
    OBB_map->color_ = Eigen::Vector3d(1, 0.5, 0);
    OBB_map->extent_ = Eigen::Vector3d(60, 60, 40);

    OBB_scan->extent_ = Eigen::Vector3d(60, 60, 40);
    OBB_scan->color_ = Eigen::Vector3d(0, 1, 0);
    rclcpp::Time time_current;
    {
        std::lock_guard<std::mutex> timestamp_lock(lock_timestamp_);
        time_current = timestamp_odom_;
    }
    rclcpp::Time time_last = time_current - rclcpp::Duration(3, 0);

    RCLCPP_INFO(this->get_logger(), "time_last: %f", time_last.seconds());
    RCLCPP_INFO(this->get_logger(), "time_current: %f", time_current.seconds());
    int scan_count = 0;

    std::string save_path = save_scan_dir_;
    if (!save_path.empty() && save_path.back() != '/')
    {
        save_path += "/";
    }

    double time_diff_loc = 5;                                     /// 前后两次定位的时间差(s)
    std::chrono::high_resolution_clock::time_point time_last_loc; /// 上次定位的完成时间点
    std::chrono::high_resolution_clock::time_point time_this_loc; /// 当前定位的开始时间点
    double loc_cost = 0;                                          /// 定位耗时(ms)
    while (rclcpp::ok() && !flag_exit_.load())
    {

        {
            std::lock_guard<std::mutex> timestamp_lock(lock_timestamp_);
            time_current = timestamp_odom_;
        }
        auto time_diff_frame = time_current.seconds() - time_last.seconds();
        time_last = time_current;
        if (std::fabs(time_diff_frame) < 1e-6)
        {
            loc_cost = 0.0;
            continue;
        }

        time_this_loc = std::chrono::high_resolution_clock::now();
        time_diff_loc = std::chrono::duration_cast<std::chrono::microseconds>(time_this_loc - time_last_loc).count() / 1000000.0 + loc_cost / 1000.0;

        if (time_diff_loc < loc_frequence_)
        {
            int wait_time = int((loc_frequence_ - time_diff_loc) * 1000);
            RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                  "tracking wait: time_diff=%.3f s, sleep %d ms", time_diff_loc, wait_time);
            std::this_thread::sleep_for(std::chrono::milliseconds(wait_time));
        }
        else
        {
            RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                  "tracking run now: time_diff=%.3f s", time_diff_loc);
        }
        auto loc_s = std::chrono::high_resolution_clock::now(); /// 开始定位计时

        std::shared_ptr<open3d::geometry::PointCloud> scan_snapshot;
        {
            std::lock_guard<std::mutex> scan_lock(lock_scan_);
            scan_snapshot = pcd_scan_cur_;
        }
        if (scan_snapshot == nullptr || scan_snapshot->IsEmpty())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            continue;
        }
        else
        {
            Eigen::Matrix4d mat_baselink2odom_cur = Eigen::Matrix4d::Identity();
            Eigen::Matrix4d mat_baselink2map_cur = Eigen::Matrix4d::Identity();
            Eigen::Matrix4d reg_matrix = Eigen::Matrix4d::Identity();

            {
                std::lock_guard<std::mutex> state_lock(lock_mat_odom2map_);
                /// 是否对odom2map进行kalman滤波
                if (filter_odom2map_)
                {
                    kalman_filter_odom2map_.inputLatestNoisyMeasurement(mat_odom2map_(2, 3));
                    kalman_filter_odom2map_.inputLatestNoisyMeasurement(mat_odom2map_(2, 3)); /// 两次
                    mat_odom2map_kalman_ = mat_odom2map_;
                    mat_odom2map_kalman_(2, 3) = kalman_filter_odom2map_.getLatestEstimatedMeasurement();
                }
                mat_baselink2odom_cur = mat_baselink2odom_;
                reg_matrix = mat_odom2map_;
            }
            *pcd_scan = *scan_snapshot;
            mat_baselink2map_cur = reg_matrix * mat_baselink2odom_cur;

            Eigen::Vector3d cur_loc(mat_baselink2map_cur(0, 3), mat_baselink2map_cur(1, 3), mat_baselink2map_cur(2, 3));
            auto dis_motion = ComputeMotionDis(last_loc_, cur_loc);
            if (dis_motion > dis_updatemap_)
            {
                auto submap_s = std::chrono::high_resolution_clock::now();

                RCLCPP_INFO(this->get_logger(),
                            "update submap: last=(%.3f, %.3f, %.3f), current=(%.3f, %.3f, %.3f), distance=%.3f",
                            last_loc_.x(), last_loc_.y(), last_loc_.z(), cur_loc.x(), cur_loc.y(), cur_loc.z(), dis_motion);
                last_loc_ = cur_loc;
                OBB_map->center_ = mat_baselink2map_cur.block<3, 1>(0, 3);
                OBB_map->R_ = mat_baselink2map_cur.block<3, 3>(0, 0);

                /// 粗地图和精地图
                *map_fine_crop = *pcd_map_fine_->Crop(*OBB_map);

                auto submap_e = std::chrono::high_resolution_clock::now();
                auto submap_cost = std::chrono::duration_cast<std::chrono::microseconds>(submap_e - submap_s).count() / 1000.0;
                RCLCPP_DEBUG(this->get_logger(), "submap_cost: %.3f ms", submap_cost);
            }

            OBB_scan->center_ = mat_baselink2odom_cur.block<3, 1>(0, 3);
            OBB_scan->R_ = mat_baselink2odom_cur.block<3, 3>(0, 0);

            target = map_fine_crop;
            const size_t target_before_sample_size = target->points_.size();
            if (target->points_.size() > static_cast<size_t>(maxpoints_target_))
            {
                target = target->RandomDownSample(double(maxpoints_target_) / target->points_.size());
            }
            const size_t target_after_sample_size = target->points_.size();

            source = pcd_scan->Crop(*OBB_scan);
            const size_t source_before_voxel_size = source->points_.size();
            source = source->VoxelDownSample(voxel_downsample_size_);
            const size_t source_after_voxel_size = source->points_.size();
            if (source->points_.size() > static_cast<size_t>(maxpoints_source_))
            {
                source = source->RandomDownSample(double(maxpoints_source_) / source->points_.size());
            }
            RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                  "tracking preprocess: target=%zu->%zu, source=%zu->%zu->%zu, voxel=%.3f",
                                  target_before_sample_size, target_after_sample_size,
                                  source_before_voxel_size, source_after_voxel_size, source->points_.size(),
                                  voxel_downsample_size_);

            if (source->points_.size() < static_cast<size_t>(min_source_points_) ||
                target->points_.size() < static_cast<size_t>(min_target_points_))
            {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                     "skip tracking icp: source=%zu (min=%d), target=%zu (min=%d), map_center=(%.3f, %.3f, %.3f), scan_center=(%.3f, %.3f, %.3f)",
                                     source->points_.size(), min_source_points_, target->points_.size(), min_target_points_,
                                     OBB_map->center_.x(), OBB_map->center_.y(), OBB_map->center_.z(),
                                     OBB_scan->center_.x(), OBB_scan->center_.y(), OBB_scan->center_.z());
                continue;
            }

            auto eva_before_icp = open3d::pipelines::registration::EvaluateRegistration(*source, *target, fitness_eval_threshold_, reg_matrix);
            RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                  "tracking before icp: eva_fitness=%f, inlier_rmse=%f, eval_threshold=%.3f, source=%zu, target=%zu",
                                  eva_before_icp.fitness_, eva_before_icp.inlier_rmse_, fitness_eval_threshold_,
                                  source->points_.size(), target->points_.size());

            auto reg_result2 = pcd_tools::RegistrationIcp(source, target, icp_distance_threshold_, reg_matrix, 1);
            reg_matrix = reg_result2.transformation_ * reg_matrix;
            auto eva_result2 = open3d::pipelines::registration::EvaluateRegistration(*source, *target, fitness_eval_threshold_, reg_matrix);
            /// 给发布的置信度赋值
            loc_fitness_.store(eva_result2.fitness_);
            double delta_trans = reg_result2.transformation_.block<3, 1>(0, 3).norm();
            double delta_yaw = std::atan2(reg_result2.transformation_(1, 0), reg_result2.transformation_(0, 0)) * 180.0 / M_PI;
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                 "tracking icp result: reg_fitness=%f, eva_before=%f, eva_after=%f, inlier_rmse=%f, delta_trans=%.3f, delta_yaw_deg=%.3f, icp_threshold=%.3f, eval_threshold=%.3f",
                                 reg_result2.fitness_, eva_before_icp.fitness_, eva_result2.fitness_, reg_result2.inlier_rmse_,
                                 delta_trans, delta_yaw, icp_distance_threshold_, fitness_eval_threshold_);
            /// 超过阈值才更新,防止因配准结果有问题而导致定位出问题
            const double loc_fitness = loc_fitness_.load();
            bool accept_tracking = loc_fitness > threshold_fitness_ &&
                                   delta_trans <= max_icp_translation_ &&
                                   std::abs(delta_yaw) <= max_icp_yaw_deg_;
            if (accept_tracking)
            {
                std::lock_guard<std::mutex> state_lock(lock_mat_odom2map_);
                mat_odom2map_ = reg_matrix;
            }
            else
            {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                     "reject tracking icp: eva_fitness=%f, threshold=%.3f, delta_trans=%.3f, max_delta=%.3f, delta_yaw_deg=%.3f, max_yaw_deg=%.3f, source=%zu, target=%zu",
                                     loc_fitness, threshold_fitness_, delta_trans, max_icp_translation_, delta_yaw, max_icp_yaw_deg_,
                                     source->points_.size(), target->points_.size());
            }

            // save_path
            if (save_scan_)
            {
                pcd_scan->Transform(mat_baselink2odom_cur.inverse());
                pcd_scan2map->Transform(mat_baselink2map_cur.inverse());
                open3d::io::WritePointCloud(save_path + std::to_string(scan_count) + "_ori.ply", *pcd_scan);
                open3d::io::WritePointCloud(save_path + std::to_string(scan_count) + "_crop.ply", *pcd_scan2map);
                scan_count += 1;
            }

            auto loc_e = std::chrono::high_resolution_clock::now(); /// 结束定位计时
            time_last_loc = loc_e;
            loc_cost = std::chrono::duration_cast<std::chrono::microseconds>(loc_e - loc_s).count() / 1000.0;
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                 "tracking localization cost: %.3f ms", loc_cost);
        }
    }
}

void GloabalLocalization::StartLoc()
{
    thread_loc_ = std::thread(&GloabalLocalization::Localization, this);
}

void GloabalLocalization::CallbackInitialPose(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr initialpose)
{
    const double current_fitness = loc_fitness_.load();
    const bool initialized = loc_initialized_.load();
    RCLCPP_INFO(this->get_logger(), "received initialpose: current confidence=%f, loc_initialized=%s",
                current_fitness, initialized ? "true" : "false");

    if (!(initialized && current_fitness > 0.99))
    {
        RCLCPP_INFO(this->get_logger(),
                    "initialpose msg: xyz=(%.3f, %.3f, %.3f), quat=(%.6f, %.6f, %.6f, %.6f)",
                    initialpose->pose.pose.position.x, initialpose->pose.pose.position.y, initialpose->pose.pose.position.z,
                    initialpose->pose.pose.orientation.x, initialpose->pose.pose.orientation.y,
                    initialpose->pose.pose.orientation.z, initialpose->pose.pose.orientation.w);

        Eigen::Quaterniond rotation_q(
            initialpose->pose.pose.orientation.w,
            initialpose->pose.pose.orientation.x,
            initialpose->pose.pose.orientation.y,
            initialpose->pose.pose.orientation.z);
        if (!std::isfinite(rotation_q.norm()) || rotation_q.norm() < 1e-6)
        {
            RCLCPP_WARN(this->get_logger(), "invalid initialpose quaternion, use identity rotation");
            rotation_q = Eigen::Quaterniond::Identity();
        }
        else
        {
            rotation_q.normalize();
        }
        mat_initialpose_.block<3, 3>(0, 0) = rotation_q.matrix();
        mat_initialpose_.block<3, 1>(0, 3) = Eigen::Vector3d(initialpose->pose.pose.position.x, initialpose->pose.pose.position.y, initialpose->pose.pose.position.z);
        Eigen::Matrix4d mat_baselink2odom_snapshot = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d mat_odom2map_snapshot = Eigen::Matrix4d::Identity();
        {
            std::lock_guard<std::mutex> state_lock(lock_mat_odom2map_);
            mat_baselink2odom_snapshot = mat_baselink2odom_;
            mat_odom2map_ = mat_initialpose_ * mat_baselink2odom_.inverse();
            mat_odom2map_snapshot = mat_odom2map_;
        }
        RCLCPP_INFO(this->get_logger(),
                    "update odom2map from initialpose: initial_xyz=(%.3f, %.3f, %.3f), odom_xyz=(%.3f, %.3f, %.3f), odom2map_xyz=(%.3f, %.3f, %.3f)",
                    mat_initialpose_(0, 3), mat_initialpose_(1, 3), mat_initialpose_(2, 3),
                    mat_baselink2odom_snapshot(0, 3), mat_baselink2odom_snapshot(1, 3), mat_baselink2odom_snapshot(2, 3),
                    mat_odom2map_snapshot(0, 3), mat_odom2map_snapshot(1, 3), mat_odom2map_snapshot(2, 3));
    }
}

double GloabalLocalization::ComputeMotionDis(const Eigen::Vector3d &a, const Eigen::Vector3d &b)
{
    return std::sqrt(std::pow(a.x() - b.x(), 2) + std::pow(a.y() - b.y(), 2) + std::pow(a.z() - b.z(), 2));
}

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<GloabalLocalization>();

    // 使用多线程执行器，可以指定线程数
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 4);
    executor.add_node(node);
    executor.spin();

    rclcpp::shutdown();
    return 0;
}
