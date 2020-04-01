#include "riptide_controllers/thruster_controller.h"

int main(int argc, char **argv)
{
  ros::init(argc, argv, "thruster_controller");
  ThrusterController tc;
  tc.Loop();
}

ThrusterController::ThrusterController() : nh(), private_nh("~")
{
  // Load parameters from .yaml files or launch files
  //ThrusterController::LoadParam<bool>("tune", tune);
  ThrusterController::LoadParam<string>("vehicle_file", vehicle_file);

  vehicle = YAML::LoadFile(vehicle_file);
  ThrusterController::LoadVehicleProperties();
  ThrusterController::SetThrusterCoeffs();
  weightLoad_eig.setZero();
  isSubmerged = false;

  for (int i = 0; i < 6; i++)
  {
    weightLoad[i] = 0;
    transportThm[i] = 0;
    command[i] = 0;
    solver_forces[i] = 0;
  }

  for (int i = 0; i < 3; i++)
  {
    solver_cob[i] = 0;
    Fb_vector[i] = 0;
  }

  odom_sub = nh.subscribe<nav_msgs::Odometry>("odometry/filtered", 1, &ThrusterController::OdomCB, this);
  cmd_sub = nh.subscribe<riptide_msgs::NetLoad>("command/net_load", 1, &ThrusterController::NetLoadCB, this);
  // cob_pub = nh.advertise<geometry_msgs::Vector3Stamped>("vehicle/cob", 1);
  cmd_pub = nh.advertise<riptide_msgs::ThrustStamped>("command/thrust", 1);
  enable_sub = nh.subscribe<riptide_msgs::ControllerEnable>("command/controller_switch", 1, &ThrusterController::ControllerCB, this);
  controller = 0;
  ThrusterController::InitDynamicReconfigure();
  ThrusterController::InitThrustMsg();

  // EOM problem
  problemEOM.AddResidualBlock(new ceres::AutoDiffCostFunction<EOM, 6, 8>(new EOM(numThrusters, thrustCoeffs, inertia, weightLoad, transportThm, command)), NULL, solver_forces);
  optionsEOM.max_num_iterations = 100;
  optionsEOM.linear_solver_type = ceres::DENSE_QR;

  // Buoyancy Problem
  problemBuoyancy.AddResidualBlock(new ceres::AutoDiffCostFunction<FindCoB, 3, 3>(new FindCoB(numThrusters, thrustCoeffs, Fb_vector, solver_forces)), NULL, solver_cob);
  optionsBuoyancy.max_num_iterations = 100;
  optionsBuoyancy.linear_solver_type = ceres::DENSE_QR;
}

// Load parameter from namespace
template <typename T>
void ThrusterController::LoadParam(std::string param, T &var)
{
  try
  {
    if (!nh.getParam(param, var))
    {
      throw 0;
    }
  }
  catch (int e)
  {
    std::string ns = nh.getNamespace();
    ROS_ERROR("Thruster Controller Namespace: %s", ns.c_str());
    ROS_ERROR("Critical! Param \"%s/%s\" does not exist or is not accessed correctly. Shutting down.", ns.c_str(), param.c_str());
    ros::shutdown();
  }
}

void ThrusterController::LoadVehicleProperties()
{
  mass = vehicle["mass"].as<double>();
  double comX = vehicle["com"][0].as<double>();
  double comY = vehicle["com"][1].as<double>();
  double comZ =  vehicle["com"][2].as<double>();
  center_of_mass[0] = comX;
  center_of_mass[1] = comY;
  center_of_mass[2] = comZ;
  Fg = mass * GRAVITY;

  // TODO: Use TF to detect when thrusters not submerged
  depth_fully_submerged = -0.3;

  Ixx = vehicle["inertia"][0].as<double>();
  Iyy = vehicle["inertia"][1].as<double>();
  Izz = vehicle["inertia"][2].as<double>();

  inertia[0] = mass;
  inertia[1] = mass;
  inertia[2] = mass;
  inertia[3] = Ixx;
  inertia[4] = Iyy;
  inertia[5] = Izz;
}

void ThrusterController::InitDynamicReconfigure()
{
  // Reset server
  param_reconfig_server.reset(new DynamicReconfigServer(param_reconfig_mutex, private_nh));

  // Now, we set the callback
  param_reconfig_callback = boost::bind(&ThrusterController::DynamicReconfigCallback, this, _1, _2);
  param_reconfig_server->setCallback(param_reconfig_callback);

  // Set Defaults
  riptide_controllers::VehiclePropertiesConfig config;
  config.Volume = vehicle["volume"].as<double>();
  config.Buoyancy_X_POS = vehicle["cob"][0].as<double>();
  config.Buoyancy_Y_POS = vehicle["cob"][1].as<double>();
  config.Buoyancy_Z_POS = vehicle["cob"][2].as<double>();
  param_reconfig_server->updateConfig(config);
  ThrusterController::DynamicReconfigCallback(config, 1);
}

void ThrusterController::SetThrusterCoeffs()
{
  numThrusters = vehicle["thrusters"].size();
  for (int i = 0; i < numThrusters; i++)
  {
    //bool enabled = vehicle["thrusters"][i]["enable"].as<bool>();
    thrustersEnabled.push_back((int)true);
  }

  // Each COLUMN contains a thruster's info
  thrusters.resize(5, numThrusters);
  thrustCoeffs.resize(6, numThrusters);
  thrusters.setZero();
  thrustCoeffs.setZero();

  for (int i = 0; i < numThrusters; i++)
    if (thrustersEnabled[i])
    {
      for (int j = 0; j < 3; j++)
      {
        // Transform X, Y, Z to COM reference frame
        thrusters(j, i) = vehicle["thrusters"][i]["pose"][j].as<double>() - center_of_mass[j];
      }
      thrusters(3, i) = vehicle["thrusters"][i]["pose"][4].as<double>();
      thrusters(4, i) = vehicle["thrusters"][i]["pose"][5].as<double>();
    }

  for (int i = 0; i < numThrusters; i++)
  {
    if (thrustersEnabled[i])
    {
      float theta = thrusters(3, i);
      float psi = thrusters(4, i);
      thrustCoeffs(0, i) = cos(psi) * cos(theta); // Effective contrbution along X-axis
      thrustCoeffs(1, i) = sin(psi) * cos(theta); // Effective contrbution along Y-axis
      thrustCoeffs(2, i) = -sin(theta);           // Effective contrbution along Z-axis

      // Cross-product
      // Determine the effective moment arms for each thruster about the B-frame axes
      thrustCoeffs.block<3, 1>(3, i) = thrusters.block<3, 1>(0, i).cross(thrustCoeffs.block<3, 1>(0, i));
    }
  }
}

void ThrusterController::InitThrustMsg()
{
  thrust_msg.header.stamp = ros::Time::now();
  thrust_msg.force.vector_port_fwd = 0;
  thrust_msg.force.vector_stbd_fwd = 0;
  thrust_msg.force.vector_port_aft = 0;
  thrust_msg.force.vector_stbd_aft = 0;
  thrust_msg.force.heave_port_fwd = 0;
  thrust_msg.force.heave_stbd_fwd = 0;
  thrust_msg.force.heave_port_aft = 0;
  thrust_msg.force.heave_stbd_aft = 0;

  cmd_pub.publish(thrust_msg);
}

// Callback for dynamic reconfigure
void ThrusterController::DynamicReconfigCallback(riptide_controllers::VehiclePropertiesConfig &config, uint32_t levels)
{
  CoB(0) = config.Buoyancy_X_POS - center_of_mass[0];
  CoB(1) = config.Buoyancy_Y_POS - center_of_mass[1];
  CoB(2) = config.Buoyancy_Z_POS - center_of_mass[2];
  Fb = config.Volume * WATER_DENSITY * GRAVITY;
}

void ThrusterController::ControllerCB(const riptide_msgs::ControllerEnable::ConstPtr &controller_msg){
  controller = controller_msg->controller;
}
void ThrusterController::OdomCB(const nav_msgs::Odometry::ConstPtr &odom_msg){
  isSubmerged = (bool)(odom_msg->pose.pose.position.z < depth_fully_submerged);

  tf2::Quaternion quat;
  tf2::fromMsg(odom_msg->pose.pose.orientation, quat);
  double yaw, pitch, roll;
  tf2::Matrix3x3 mat(quat);
  mat.getRPY(roll, pitch, yaw);
  
  Vector3d angular_vel;
  angular_vel[0] = odom_msg->twist.twist.angular.x;
  angular_vel[1] = odom_msg->twist.twist.angular.y;
  angular_vel[2] = odom_msg->twist.twist.angular.z;

  transportThm[3] = -angular_vel[1] * angular_vel[2] * (Izz - Iyy);
  transportThm[4] = -angular_vel[0] * angular_vel[2] * (Ixx - Izz);
  transportThm[5] = -angular_vel[0] * angular_vel[1] * (Iyy - Ixx);

  

  Vector3d Fb_eig;
  Fb_eig(0) = -Fb * sin(pitch);
  Fb_eig(1) = Fb * sin(roll) * cos(pitch);
  Fb_eig(2) = Fb * cos(roll) * cos(pitch);

  weightLoad_eig(0) = (Fg - Fb) * sin(pitch);
  weightLoad_eig(1) = -(Fg - Fb) * sin(roll) * cos(pitch);
  weightLoad_eig(2) = -(Fg - Fb) * cos(roll) * cos(pitch);
  weightLoad_eig.segment<3>(3) = CoB.cross(Fb_eig);
  weightLoad_eig = weightLoad_eig * ((int)(isSubmerged));

  // Convert Eigen::VectorXd to c++ double[X]
  Map<RowMatrixXd>(&weightLoad[0], weightLoad_eig.rows(), weightLoad_eig.cols()) = weightLoad_eig;
  Map<Vector3d>(&Fb_vector[0], Fb_eig.rows(), Fb_eig.cols()) = Fb_eig;
}

void ThrusterController::NetLoadCB(const riptide_msgs::NetLoad::ConstPtr &load_msg)
{
  command[0] = load_msg->force.x;
  command[1] = load_msg->force.y;
  command[2] = load_msg->force.z;
  command[3] = load_msg->moment.x;
  command[4] = load_msg->moment.y;
  command[5] = load_msg->moment.z;

  // These initial guesses don't make much of a difference.
  for (int i = 0; i < numThrusters; i++)
    solver_forces[i] = 0.0;

  // Solve all my problems
  ceres::Solve(optionsEOM, &problemEOM, &summaryEOM);

  thrust_msg.header.stamp = ros::Time::now();
  thrust_msg.force.vector_port_fwd = solver_forces[thrust_msg.force.VPF];
  thrust_msg.force.vector_stbd_fwd = solver_forces[thrust_msg.force.VSF];
  thrust_msg.force.vector_port_aft = solver_forces[thrust_msg.force.VPA];
  thrust_msg.force.vector_stbd_aft = solver_forces[thrust_msg.force.VSA];
  thrust_msg.force.heave_port_fwd = solver_forces[thrust_msg.force.HPF];
  thrust_msg.force.heave_stbd_fwd = solver_forces[thrust_msg.force.HSF];
  thrust_msg.force.heave_port_aft = solver_forces[thrust_msg.force.HPA];
  thrust_msg.force.heave_stbd_aft = solver_forces[thrust_msg.force.HSA];
  //disable if another controller is taking over
  if(controller == 0){
    cmd_pub.publish(thrust_msg);
  }
  

  // Tune Buoyancy - locate the center of buoyancy
  // The output will only make sense if the depth, roll, and pitch controllers
  // are initialized, and the vehicle is roughly stationary in the water.
  // The output should contain non-zero distances so long as the the vehicle is
  // unable to reach a target orientation along any axis.
  // The depth controller is used only to keep the vehicle fully submerged
  // Initialize values
  // solver_cob[0] = 0.0;
  // solver_cob[1] = 0.0;
  // solver_cob[2] = 0.0;

  // ceres::Solve(optionsBuoyancy, &problemBuoyancy, &summaryBuoyancy);
  // cob_msg.header.stamp = ros::Time::now();
  // cob_msg.vector.x = solver_cob[0];
  // cob_msg.vector.y = solver_cob[1];
  // cob_msg.vector.z = solver_cob[2];
  // cob_pub.publish(cob_msg);
}

void ThrusterController::Loop()
{
  ros::Rate rate(200);
  while (!ros::isShuttingDown())
  {
    ros::spinOnce();
    rate.sleep();
  }
}
