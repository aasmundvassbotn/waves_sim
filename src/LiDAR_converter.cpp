#include <cmath>
#include <limits>
#include <format>
#include <iostream>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <image_transport/image_transport.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <opencv2/opencv.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>

using namespace message_filters;

class DepthListener : public rclcpp::Node
{
public:
    DepthListener() : Node("LiDAR_converter"), id_(0)
    {

        int id_depth_imgs = 0;
        
        std::string front_depth_topic = std::format("/sim_camF_depth_{}/image_depth", id_depth_imgs);
        std::string back_depth_topic = std::format("/sim_camB_depth_{}/image_depth", id_depth_imgs);
        std::string right_depth_topic = std::format("/sim_camR_depth_{}/image_depth", id_depth_imgs);
        std::string left_depth_topic = std::format("/sim_camL_depth_{}/image_depth", id_depth_imgs);

        // Subscribers
        sub_front_.subscribe(this, front_depth_topic);
        sub_back_.subscribe(this, back_depth_topic);
        sub_right_.subscribe(this, right_depth_topic);
        sub_left_.subscribe(this, left_depth_topic);

        // Synchronizer
        sync_ = std::make_shared<message_filters::Synchronizer<MySyncPolicy>>(MySyncPolicy(10), sub_front_, sub_back_, sub_right_, sub_left_);
        sync_->registerCallback(std::bind(&DepthListener::depth_callback, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));

        // Publisher
        publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/sim_LiDAR_depth/points", 10);
    }

private:
    using MySyncPolicy = sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image, sensor_msgs::msg::Image, sensor_msgs::msg::Image>;

    // Subscribers
    message_filters::Subscriber<sensor_msgs::msg::Image> sub_front_, sub_back_, sub_right_, sub_left_;
    std::shared_ptr<message_filters::Synchronizer<MySyncPolicy>> sync_;

    // Publisher
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;

    int id_;
    const float WATER_LEVEL = 0.0;
    const float SENSOR_HEIGHT = 0.45; // BlueBoat
    // const float SENSOR_HEIGHT = 2.5;

    void depth_callback(const sensor_msgs::msg::Image::ConstSharedPtr &msg_front,
                        const sensor_msgs::msg::Image::ConstSharedPtr &msg_back,
                        const sensor_msgs::msg::Image::ConstSharedPtr &msg_right,
                        const sensor_msgs::msg::Image::ConstSharedPtr &msg_left)
    {
        cv_bridge::CvImagePtr cv_ptr_front, cv_ptr_back, cv_ptr_right, cv_ptr_left;
        try
        {
            cv_ptr_front = cv_bridge::toCvCopy(msg_front, "32FC1");
            cv_ptr_back = cv_bridge::toCvCopy(msg_back, "32FC1");
            cv_ptr_right = cv_bridge::toCvCopy(msg_right, "32FC1");
            cv_ptr_left = cv_bridge::toCvCopy(msg_left, "32FC1");
        }
        catch (cv_bridge::Exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "CV Bridge error: %s", e.what());
            return;
        }

        int height = cv_ptr_front->image.rows;
        int width = cv_ptr_front->image.cols;

        float fov_x = M_PI / 2.0;  // 90 degrees
        float fov_y = 2.0 * atan(tan(fov_x / 2.0) * height / width);
        float fx = width / (2.0 * tan(fov_x / 2.0));
        float fy = height / (2.0 * tan(fov_y / 2.0));
        float cx = width / 2.0, cy = height / 2.0;

        // LiDAR parameters
        // Ouster OS1 
        // min range: 0.5m, max range: 170m
        // vertical FOV: 42.4°
        // horizontal FOV: 360°
        const int num_sweeps = 32;
        const float lidar_fov_y = 42.4 * M_PI / 180.0;
        if (lidar_fov_y > fov_y) {
            RCLCPP_ERROR(this->get_logger(), "Error: LiDAR vertical FOV exceeds camera's vertical FOV!");
            return;
        }
        const int num_azimuth_bins = 1024;   // Can't be higher than the camera's width
        if (num_azimuth_bins > width * 4) {
            RCLCPP_ERROR(this->get_logger(), "Error: LiDAR azimuth bins exceed camera's width!");
            return;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

        auto processAllDepths = [&](const cv::Mat& img_front,
                            const cv::Mat& img_left,
                            const cv::Mat& img_back,
                            const cv::Mat& img_right,
                            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                            double fx, double fy, double cx, double cy,
                            int width, int height,
                            double lidar_fov_y,
                            int num_sweeps,
                            int num_azimuth_bins
                        )
        {
            for (int sweep = 0; sweep < num_sweeps; ++sweep)
            {
                double theta = -lidar_fov_y / 2.0 + sweep * (lidar_fov_y / (num_sweeps - 1));
                double v_f = cy - fy * std::tan(theta);
                int v_int = static_cast<int>(std::lround(v_f));

                if (v_int < 0 || v_int >= height)
                    continue;

                for (int bin = 0; bin < num_azimuth_bins; ++bin)
                {
                    double phi_global = -M_PI + bin * (2.0 * M_PI / num_azimuth_bins);
                    double camera_yaw = 0.0;
                    const cv::Mat* depth_image = nullptr;

                    if (phi_global >= -M_PI/4 && phi_global <= M_PI/4) {
                        depth_image = &img_front;
                        camera_yaw = 0.0;
                    } else if (phi_global > M_PI/4 && phi_global < 3*M_PI/4) {
                        depth_image = &img_left;
                        camera_yaw = M_PI / 2.0;
                    } else if (phi_global <= -M_PI/4 && phi_global > -3*M_PI/4) {
                        depth_image = &img_right;
                        camera_yaw = -M_PI / 2.0;
                    } else {
                        depth_image = &img_back;
                        camera_yaw = M_PI;
                    }

                    double phi_rel = phi_global - camera_yaw;
                    double u_f = cx + fx * std::tan(phi_rel);
                    int u_int = static_cast<int>(std::lround(u_f));

                    if (u_int < 0 || u_int >= width)
                        continue;

                    float depth = depth_image->at<float>(v_int, u_int);
                    if (depth <= 0)
                        continue;

                    double X = depth * (u_f - cx) / fx;
                    double Y = depth * (v_f - cy) / fy;
                    double Z = depth;

                    double X_rot = std::cos(camera_yaw) * X - std::sin(camera_yaw) * Z;
                    double Z_rot = std::sin(camera_yaw) * X + std::cos(camera_yaw) * Z;

                    cloud->push_back(pcl::PointXYZ(static_cast<float>(X_rot),
                                            static_cast<float>(Z_rot),
                                            static_cast<float>(-Y)));

                    // cloud->push_back(pcl::PointXYZ(static_cast<float>(Z_rot),
                    //                         static_cast<float>(-X_rot),
                    //                         static_cast<float>(-Y)));
                }
            }
        };
    
        processAllDepths(cv_ptr_front->image, cv_ptr_left->image, cv_ptr_back->image, cv_ptr_right->image, cloud, fx, fy, cx, cy, width, height, lidar_fov_y, num_sweeps, num_azimuth_bins);

        // Filter ground plane

        const float WATER_PLANE = (WATER_LEVEL + SENSOR_HEIGHT + 0.1) * -1.0;

        pcl::PointCloud<pcl::PointXYZ> filtered_cloud;

        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud(cloud);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(WATER_PLANE, std::numeric_limits<float>::max());

        pass.filter(filtered_cloud);

        sensor_msgs::msg::PointCloud2 cloud_msg; 
        pcl::toROSMsg(filtered_cloud, cloud_msg);
        cloud_msg.header = msg_front->header;
        cloud_msg.header.frame_id = "LiDAR";

        RCLCPP_INFO(this->get_logger(), "Publishing merged point cloud: %d", id_++);
        publisher_->publish(cloud_msg);
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DepthListener>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}












        // auto process_depth = [&](const cv::Mat &depth_image, float yaw_offset)
        // {
        //     for (int sweep = 0; sweep < num_sweeps; ++sweep)
        //     {
        //         // 1) Compute vertical angle for this LiDAR ring
        //         float theta = -lidar_fov_y / 2.0f + sweep * (lidar_fov_y / (num_sweeps - 1));
        
        //         // 2) Convert that angle to a pixel row using the pinhole model
        //         //    v_f = cy - fy * tan(theta)  (assuming positive theta => up)
        //         float v_f = cy - fy * std::tan(theta);
        
        //         // Round to nearest integer
        //         int v_int = static_cast<int>(std::lround(v_f));
        
        //         // Skip if out of image bounds
        //         if (v_int < 0 || v_int >= height) {
        //             continue;
        //         }
        
        //         // 3) Sweep across the horizontal dimension
        //         //    We'll sample from -fov_x/2 to +fov_x/2
        //         for (int bin = 0; bin < num_azimuth_bins; ++bin)
        //         {
        //             float phi = -fov_x / 2.0f + bin * (fov_x / (num_azimuth_bins - 1));
        //             // Convert horizontal angle to pixel column
        //             float u_f = cx + fx * std::tan(phi);
        
        //             // Round to nearest integer
        //             int u_int = static_cast<int>(std::lround(u_f));
        
        //             // Skip if out of image bounds
        //             if (u_int < 0 || u_int >= width) {
        //                 continue;
        //             }
        
        //             // 4) Read depth from the single nearest pixel
        //             float depth = depth_image.at<float>(v_int, u_int);
        
        //             if (depth > 0.0f)
        //             {
        //                 // Convert (u, v, depth) to 3D in camera frame
        //                 float X = depth * (u_f - cx) / fx;
        //                 float Y = depth * (v_f - cy) / fy;
        //                 float Z = depth;
        
        //                 // 5) Apply yaw rotation for this camera's offset
        //                 //    (assuming rotation around Y-up or Z-up depending on your convention)
        //                 float X_rot = std::cos(yaw_offset) * X - std::sin(yaw_offset) * Z;
        //                 float Z_rot = std::sin(yaw_offset) * X + std::cos(yaw_offset) * Z;
        
        //                 // 6) Insert into the cloud
        //                 //    Adjust sign of Y if needed by your coordinate system
        //                 cloud->emplace_back(X_rot, Z_rot, -Y);
        //             }
        //         }
        //     }
        // };

        // process_depth(cv_ptr_front->image, 0.0f);       // Front camera (0° yaw)
        // process_depth(cv_ptr_back->image, M_PI);      // Back camera (180° yaw)
        // process_depth(cv_ptr_right->image, -M_PI_2);   // Right camera (-90° yaw)
        // process_depth(cv_ptr_left->image, M_PI_2);     // Left camera (90° yaw)