#include "riptide_controllers/linear_controller.h"
#define VELOCITY_P 2.0
#define DRAG_COEFF 0
#define MAX_FORCE 30
// #! /usr/bin/env python


int main(int argc, char **argv)
{
  ros::init(argc, argv, "linear_controller");
  LinearController lc;
  lc.Loop();
}

LinearController::LinearController() : nh("~")
{
    velocityCmd = nullptr;
    force = 0;

    command_x_sub = nh.subscribe<riptide_msgs::LinearCommand>("/command/x", 1, &LinearController::LinearCommand, this);
    // command_y_sub = nh.subscribe<riptide_msgs::LinearCommand>("/command/x", 1, &LinearController::LinearCommand, this);
    dvl_sub = nh.subscribe<riptide_msgs::Dvl>("/state/dvl", 1, &LinearController::Dvl, this);
}

// class LinearController():

//     VELOCITY_P = 2.0
//     DRAG_COEFF = 0
//     MAX_FORCE = 30

//     velocityCmd = None
//     force = 0

//     def __init__(self, publisher):
//         self.publisher = publisher

//     def cmdCb(self, msg):
//         if msg.mode == LinearCommand.VELOCITY:
//             self.velocityCmd = msg.value
//         elif msg.mode == LinearCommand.FORCE:
//             self.force = msg.value
//             self.velocityCmd = None
//             self.publisher.publish(self.force)

//     def updateState(self, velocity):

//         # If there is a desired velocity
//         if self.velocityCmd != None:
//             # Set force porportional to velocity error
//             self.force = 0
//             if not math.isnan(velocity):
//                 self.force = max(-self.MAX_FORCE, min(self.MAX_FORCE, self.VELOCITY_P * (self.velocityCmd - velocity) + self.DRAG_COEFF * velocity * abs(velocity)))

//     def reconfigure(self, config, name):
//         self.VELOCITY_P = config[name + "_velocity_p"]
//         self.DRAG_COEFF = config[name + "_drag_coeff"]
        
        

// XPub = rospy.Publisher("/command/force_x", Float64, queue_size=5)
// YPub = rospy.Publisher("/command/force_y", Float64, queue_size=5)

// xController = LinearController(XPub)
// yController = LinearController(YPub)

void LinearController::DvlCB(const riptide_msgs::Dvl::ConstPtr &dvl_msg)
{
  xController.updateState(dvl_msg->velocity.x);
  yController.updateState(dvl_msg->velocity.y);
}

// def dvlCb(msg):
//     xController.updateState(msg.velocity.x)
//     yController.updateState(msg.velocity.y)

//     # Publish new forces
//     XPub.publish(xController.force)
//     YPub.publish(yController.force)


// def dynamicReconfigureCb(config, level):
//     # On dynamic reconfiguration
//     xController.reconfigure(config, "x")
//     yController.reconfigure(config, "y")
//     return config


// if __name__ == '__main__':

//     rospy.init_node("linear_controller")

//     # Set subscribers
//     rospy.Subscriber("/command/x", LinearCommand, xController.cmdCb)
//     rospy.Subscriber("/command/y", LinearCommand, yController.cmdCb)
//     rospy.Subscriber("/state/dvl", Dvl, dvlCb)
    
//     Server(LinearControllerConfig, dynamicReconfigureCb)

//     rospy.spin()


void LinearController::Loop()
{
  ros::Rate rate(200);
  while (!ros::isShuttingDown())
  {
    ros::spinOnce();
    rate.sleep();
  }
}
