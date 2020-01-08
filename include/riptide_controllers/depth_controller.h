#ifndef DEPTH_CONTROLLER_H
#define DEPTH_CONTROLLER_H

#include "ros/ros.h"
#include "control_toolbox/pid.h"
#include "geometry_msgs/Vector3Stamped.h"
#include "tf/transform_listener.h"
#include "riptide_msgs/Depth.h"
#include "riptide_msgs/DepthCommand.h"
#include "sensor_msgs/Imu.h"
#include "riptide_msgs/ResetControls.h"
#include "riptide_msgs/ControlStatus.h"
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
using namespace std;

class DepthController
{
  private:
    // Comms
    ros::NodeHandle nh;
    ros::Subscriber depth_sub, imu_sub, cmd_sub;
    ros::Publisher cmd_pub, status_pub;

    control_toolbox::Pid depth_controller_pid;
    geometry_msgs::Vector3Stamped cmd_force;
    double output, MAX_DEPTH, MAX_DEPTH_ERROR;

    // IIR Filter variables for error_dot
    double PID_IIR_LPF_bandwidth, dt_iir, alpha, sensor_rate;

    riptide_msgs::ControlStatus status_msg;

    tf2::Quaternion quat;

    //PID
    double depth_error, depth_error_dot;
    double current_depth, depth_cmd;
    double last_error, last_error_dot;
    double dt;

    ros::Time sample_start;
    ros::Duration sample_duration;

    bool pid_depth_active;

    void UpdateError();
    double Constrain(double current, double max);
    double SmoothErrorIIR(double input, double prev);
    void ResetDepth();

  public:
    DepthController();
    template <typename T>
    void LoadParam(string param, T &var);
    void CommandCB(const riptide_msgs::DepthCommand::ConstPtr &cmd);
    void DepthCB(const riptide_msgs::Depth::ConstPtr &depth_msg);
    void ImuCB(const sensor_msgs::Imu::ConstPtr &imu_msg);
 };

 #endif
