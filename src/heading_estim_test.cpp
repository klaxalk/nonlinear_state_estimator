/* include //{ */

#include <memory>
#include <ros/ros.h>
#include <nodelet/nodelet.h>

#include <mrs_lib/ukf.h>
#include <mrs_lib/param_loader.h>
#include <mrs_lib/mutex.h>
#include <mrs_lib/transformer.h>
#include <mrs_lib/subscribe_handler.h>

#include <mrs_msgs/UavState.h>
#include <mrs_msgs/ControlManagerDiagnostics.h>

#include <nav_msgs/Odometry.h>

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Vector3Stamped.h>

#include <random>

#include <dynamic_reconfigure/server.h>
#include <heading_estim_test/heading_estim_testConfig.h>

//}

/* UKF helpers //{ */

namespace mrs_lib
{
const int n_states       = 7;
const int n_inputs       = 0;
const int n_measurements = 4;

using ukf_t = UKF<n_states, n_inputs, n_measurements>;
}  // namespace mrs_lib

// Some helpful aliases to make writing of types shorter
using namespace mrs_lib;
using Q_t         = ukf_t::Q_t;
using tra_model_t = ukf_t::transition_model_t;
using obs_model_t = ukf_t::observation_model_t;
using x_t         = ukf_t::x_t;
using P_t         = ukf_t::P_t;
using u_t         = ukf_t::u_t;
using z_t         = ukf_t::z_t;
using R_t         = ukf_t::R_t;
using statecov_t  = ukf_t::statecov_t;

// helper enum to simplify addressing of UKF states
enum UKF_X
{
  X_X                   = 0,
  X_Y                   = 1,
  X_SPEED               = 2,
  X_HEADING             = 3,
  X_HEADING_RATE_TRUE   = 4,
  X_HEADING_RATE_BIASED = 5,
  X_HEADING_RATE_BIAS   = 6,
};

// helper enum to simplify addressing of UKF measurements
enum UKF_Z
{
  Z_SPEED        = 0,
  Z_X            = 1,
  Z_Y            = 2,
  Z_HEADING_RATE = 3,
};

//}

namespace heading_estim_test
{

using namespace Eigen;

// --------------------------------------------------------------
// |                          the class                         |
// --------------------------------------------------------------

/* class HeadingEstimTest //{ */

class HeadingEstimTest : public nodelet::Nodelet {

public:
  virtual void onInit();

private:
  ros::NodeHandle nh_;

  bool is_initialized_ = false;

  std::string _uav_name_;

  double ukf_alpha_;
  double ukf_kappa_;
  double ukf_beta_;

  double _ukf_bias_covariance_;

  double _ukf_measurement_covariance_position_;
  double _ukf_measurement_covariance_speed_;
  double _ukf_measurement_covariance_heading_rate_;

  std::shared_ptr<ukf_t> ukf_;

  statecov_t statecov_;
  std::mutex mutex_statecov_;

  // model covaritance
  Q_t Q_;

  // measurement covariance
  R_t R_;

  // | --------------------------- ukf -------------------------- |

  // This function implements the state transition
  x_t ufkModelTransition(const x_t& x, const u_t& u, const double dt);

  // This function implements the observation generation from a state
  ukf_t::z_t ukfModelObservation(const ukf_t::x_t& x);

  // | --------------------- noise generator -------------------- |

  std::default_random_engine noise_generator_position_;
  std::default_random_engine noise_generator_speed_;
  std::default_random_engine noise_generator_heading_rate_;

  std::normal_distribution<double> normal_distribution_position_;
  std::normal_distribution<double> normal_distribution_speed_;
  std::normal_distribution<double> normal_distribution_heading_rate_;

  double _measurement_position_sigma_;
  double _measurement_velocity_sigma_;
  double _measurement_heading_rate_sigma_;

  // | ----------------------- publishers ----------------------- |

  ros::Publisher pub_estimate_;
  ros::Publisher pub_measurement_;

  // | ----------------------- subscribers ---------------------- |

  mrs_lib::SubscribeHandler<mrs_msgs::UavState> sh_uav_state_;
  void                                          callbackUavState(mrs_lib::SubscribeHandler<mrs_msgs::UavState>& wrp);

  mrs_lib::SubscribeHandler<mrs_msgs::ControlManagerDiagnostics> sh_control_manager_diag_;

  ros::Time uav_state_last_time;
  bool      got_uav_state_ = false;

  // | ----------------------- transformer ---------------------- |

  mrs_lib::Transformer transformer_;

  // | --------------- dynamic reconfigure server --------------- |

  boost::recursive_mutex                               mutex_drs_;
  typedef heading_estim_test::heading_estim_testConfig DrsParams_t;
  typedef dynamic_reconfigure::Server<DrsParams_t>     Drs_t;
  boost::shared_ptr<Drs_t>                             drs_;
  void                                                 callbackDrs(heading_estim_test::heading_estim_testConfig& params, uint32_t level);
  DrsParams_t                                          params_;
  std::mutex                                           mutex_params_;
};

//}

/* onInit() //{ */

void HeadingEstimTest::onInit() {

  nh_ = nodelet::Nodelet::getMTPrivateNodeHandle();

  ros::Time::waitForValid();

  // | --------------------- load the params -------------------- |

  mrs_lib::ParamLoader param_loader(nh_, "HeadingEstimTest");

  param_loader.loadParam("artificial_gyro_bias", params_.gyro_bias);

  param_loader.loadParam("measurement/position_sigma", _measurement_position_sigma_);
  param_loader.loadParam("measurement/velocity_sigma", _measurement_velocity_sigma_);
  param_loader.loadParam("measurement/heading_rate_sigma", _measurement_heading_rate_sigma_);

  param_loader.loadParam("ukf/alpha", ukf_alpha_);
  param_loader.loadParam("ukf/kappa", ukf_kappa_);
  param_loader.loadParam("ukf/beta", ukf_beta_);

  param_loader.loadParam("ukf/bias_covariance", _ukf_bias_covariance_);

  param_loader.loadParam("ukf/measurement_covariance/position", _ukf_measurement_covariance_position_);
  param_loader.loadParam("ukf/measurement_covariance/speed", _ukf_measurement_covariance_speed_);
  param_loader.loadParam("ukf/measurement_covariance/heading_rate", _ukf_measurement_covariance_heading_rate_);

  param_loader.loadParam("uav_name", _uav_name_);

  if (!param_loader.loadedSuccessfully()) {
    ROS_ERROR("[HeadingEstimTest]: Could not load all parameters!");
    ros::shutdown();
  }

  // | ------------------- dynamic reconfigure ------------------ |

  drs_.reset(new Drs_t(mutex_drs_, nh_));
  drs_->updateConfig(params_);
  Drs_t::CallbackType f = boost::bind(&HeadingEstimTest::callbackDrs, this, _1, _2);
  drs_->setCallback(f);

  // | -------------------- measurement noise ------------------- |

  normal_distribution_position_     = std::normal_distribution<double>(0, _measurement_position_sigma_);
  normal_distribution_speed_        = std::normal_distribution<double>(0, _measurement_velocity_sigma_);
  normal_distribution_heading_rate_ = std::normal_distribution<double>(0, _measurement_heading_rate_sigma_);

  // | --------------------------- ukf -------------------------- |

  // bind the transition and observation methods
  tra_model_t tra_model(std::bind(&HeadingEstimTest::ufkModelTransition, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
  obs_model_t obs_model(std::bind(&HeadingEstimTest::ukfModelObservation, this, std::placeholders::_1));

  // clang-format off

  // Initialize the process noise matrix
  Q_ <<
    1, 0, 0, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0,
    0, 0, 0, 1, 0, 0, 0,
    0, 0, 0, 0, 1, 0, 0,
    0, 0, 0, 0, 0, 1, 0,
    0, 0, 0, 0, 0, 0, _ukf_bias_covariance_;

  // Initialize the measurement noise matrix
  R_ <<
    _ukf_measurement_covariance_position_, 0, 0, 0,
    0, _ukf_measurement_covariance_position_, 0, 0,
    0, 0, _ukf_measurement_covariance_speed_, 0,
    0, 0, 0, _ukf_measurement_covariance_heading_rate_;

  // Generate initial state and covariance
  x_t x0; x0 <<
    0,
    0,
    0,
    0,
    0,
    0,
    0;

  // clang-format on

  // initialize the ukf internal covariance
  P_t P_tmp = P_t::Identity();

  statecov_.x = x0;
  statecov_.P = P_t::Identity(6, 6);

  // instantiate the filter
  ukf_ = std::make_shared<ukf_t>(tra_model, obs_model, ukf_alpha_, ukf_kappa_, ukf_beta_);

  // | ----------------------- subscribers ---------------------- |

  mrs_lib::SubscribeHandlerOptions shopts;
  shopts.nh                 = nh_;
  shopts.node_name          = "HeadingEstimTest";
  shopts.no_message_timeout = mrs_lib::no_timeout;
  shopts.threadsafe         = true;
  shopts.autostart          = true;
  shopts.queue_size         = 10;
  shopts.transport_hints    = ros::TransportHints().tcpNoDelay();

  sh_uav_state_            = mrs_lib::SubscribeHandler<mrs_msgs::UavState>(shopts, "uav_state_in", &HeadingEstimTest::callbackUavState, this);
  sh_control_manager_diag_ = mrs_lib::SubscribeHandler<mrs_msgs::ControlManagerDiagnostics>(shopts, "control_manager_diag_in");

  // | ----------------------- publishers ----------------------- |

  pub_estimate_    = nh_.advertise<nav_msgs::Odometry>("estimate_out", 1);
  pub_measurement_ = nh_.advertise<geometry_msgs::PoseStamped>("measurement_out", 1);

  // | ----------------------- transformer ---------------------- |

  transformer_ = mrs_lib::Transformer("HeadingEstimTest", _uav_name_);

  // | ---------------------- finish innit ---------------------- |

  is_initialized_ = true;

  ROS_INFO("[HeadingEstimTest]: initialized");
}

//}

// --------------------------------------------------------------
// |                          callbacks                         |
// --------------------------------------------------------------

/* //{ callbackUavState() */

void HeadingEstimTest::callbackUavState(mrs_lib::SubscribeHandler<mrs_msgs::UavState>& wrp) {

  if (!is_initialized_) {
    return;
  }

  if (!sh_control_manager_diag_.hasMsg()) {
    ROS_INFO_THROTTLE(1.0, "[HeadingEstimTest]: waiting for control manager diagnostics");
    return;
  }

  auto params = mrs_lib::get_mutexed(mutex_params_, params_);

  // get the incoming UAV STATE from the subscribe handler
  mrs_msgs::UavState uav_state = *wrp.getMsg();

  ROS_INFO_ONCE("[HeadingEstimTest]: getting uav state");

  if (!got_uav_state_) {
    got_uav_state_      = true;
    uav_state_last_time = ros::Time::now();
    return;
  }

  // we need dt
  double dt           = (ros::Time::now() - uav_state_last_time).toSec();
  uav_state_last_time = ros::Time::now();

  // fallback for invalid dt (happens in simulation)
  if (dt < 1e-3) {

    // the expected rate is 100 Hz
    dt = 1e-2;
  }

  // to simulate measurements, we will need the following variables
  double uav_position_x;
  double uav_position_y;
  double uav_forward_speed;
  double uav_heading_rate;

  // fill in the state variables
  {
    uav_position_x = uav_state.pose.position.x;
    uav_position_y = uav_state.pose.position.y;

    // convert velocity to the body frame
    geometry_msgs::Vector3Stamped velocity_world;
    velocity_world.header = uav_state.header;

    velocity_world.vector.x = uav_state.velocity.linear.x;
    velocity_world.vector.y = uav_state.velocity.linear.y;
    velocity_world.vector.z = uav_state.velocity.linear.z;

    auto res = transformer_.transformSingle(_uav_name_ + "/fcu_untilted", velocity_world);

    if (!res) {
      ROS_WARN("[HeadingEstimTest]: could not transform velocity to the world frame");
      return;
    }

    uav_forward_speed = res.value().vector.x;

    // extract the heading rate from the angular rate vector and the orientation
    uav_heading_rate = mrs_lib::AttitudeConverter(uav_state.pose.orientation).getHeadingRate(uav_state.velocity.angular);
  }

  geometry_msgs::PoseStamped measurement_debug;

  // create the input vector ... we don't have an input, so it's empty
  u_t u;

  // create the measurement vector
  z_t z;

  // fill in the measurement vector
  {
    // | ------------------------ position ------------------------ |

    z(Z_X) = uav_position_x + normal_distribution_position_(noise_generator_position_);
    z(Z_Y) = uav_position_y + normal_distribution_position_(noise_generator_position_);

    // | -------------------------- speed ------------------------- |

    // extract the x component of the body velocity, that is the "nonholonomic-robot" speed
    z(Z_SPEED) = uav_forward_speed + normal_distribution_speed_(noise_generator_speed_);

    // | ---------------------- heading rate ---------------------- |

    // combine the heading rate + noise + bias
    z(Z_HEADING_RATE) = uav_heading_rate + params.gyro_bias + normal_distribution_heading_rate_(noise_generator_heading_rate_);

    if (!sh_control_manager_diag_.getMsg()->tracker_status.have_goal) {
      ROS_WARN_THROTTLE(1.0, "[HeadingEstimTest]: stationary");
      z(Z_SPEED) = 0;
    }
  }

  // print useful info
  // print it as a warning when the bias is not estimated correctly
  if (abs(statecov_.x(X_HEADING_RATE_BIAS) - params.gyro_bias) < 0.1) {
    ROS_INFO_THROTTLE(0.1, "[HeadingEstimTest]: UKF iteration: hdg rate bias %.2f (should be %.2f), rate %.2f (should be %.2f)",
                      statecov_.x(X_HEADING_RATE_BIAS), params.gyro_bias, statecov_.x(X_HEADING_RATE_TRUE),
                      mrs_lib::AttitudeConverter(uav_state.pose.orientation).getHeadingRate(uav_state.velocity.angular));
  } else {
    ROS_WARN_THROTTLE(0.1, "[HeadingEstimTest]: UKF iteration: hdg rate bias %.2f (should be %.2f), rate %.2f (should be %.2f)",
                      statecov_.x(X_HEADING_RATE_BIAS), params.gyro_bias, statecov_.x(X_HEADING_RATE_TRUE),
                      mrs_lib::AttitudeConverter(uav_state.pose.orientation).getHeadingRate(uav_state.velocity.angular));
  }

  // There should be a try-catch here to prevent program crashes
  // in case of numerical instabilities (which are possible with UKF)
  try {
    // Apply the prediction step
    statecov_ = ukf_->predict(statecov_, u, Q_, dt);

    // Apply the correction step
    statecov_ = ukf_->correct(statecov_, z, R_);
  }
  catch (const std::exception& e) {
    // In case of error, alert the user
    ROS_ERROR("UKF failed: %s", e.what());
  }

  // | ---------------- publish the current state --------------- |

  {
    nav_msgs::Odometry odom_out;
    odom_out.header.frame_id = uav_state.header.frame_id;
    odom_out.child_frame_id  = _uav_name_ + "/fcu";
    odom_out.header.stamp    = ros::Time::now();

    odom_out.pose.pose.position.x = statecov_.x(X_X);
    odom_out.pose.pose.position.y = statecov_.x(X_Y);
    odom_out.pose.pose.position.z = uav_state.pose.position.z;

    odom_out.twist.twist.linear.x = statecov_.x(X_SPEED);

    odom_out.twist.twist.angular.z = statecov_.x(X_HEADING_RATE_BIASED);

    odom_out.pose.pose.orientation = mrs_lib::AttitudeConverter(0, 0, statecov_.x(X_HEADING));

    pub_estimate_.publish(odom_out);
  }

  // | ----------- publish the measurement (for debug) ---------- |
  {
    measurement_debug.header = uav_state.header;

    measurement_debug.pose.position.x = z(Z_X);
    measurement_debug.pose.position.y = z(Z_Y);
    measurement_debug.pose.position.z = uav_state.pose.position.z;

    measurement_debug.pose.orientation = uav_state.pose.orientation;

    pub_measurement_.publish(measurement_debug);
  }
}

//}

/* callbackDrs() //{ */

void HeadingEstimTest::callbackDrs(heading_estim_test::heading_estim_testConfig& params, [[maybe_unused]] uint32_t level) {

  mrs_lib::set_mutexed(mutex_params_, params, params_);

  ROS_INFO("[CircleFlier]: DRS updated");
}

//}

// --------------------------------------------------------------
// |                             UKF                            |
// --------------------------------------------------------------

/* ukfModelTransition() //{ */

x_t HeadingEstimTest::ufkModelTransition(const x_t& x, const u_t& u, const double dt) {

  x_t ret;

  auto params = mrs_lib::get_mutexed(mutex_params_, params_);

  // if we are "moving"
  if (sh_control_manager_diag_.getMsg()->tracker_status.have_goal) {

    // position from speed and heading
    ret(X_X) = x(X_X) + dt * std::cos(x(X_HEADING)) * X_SPEED;
    ret(X_Y) = x(X_Y) + dt * std::sin(x(X_HEADING)) * X_SPEED;

    // speed remains speed
    ret(X_SPEED) = x(X_SPEED);

    // heading is an integration of the heading rate and bias
    ret(X_HEADING) = x(X_HEADING) + dt * (x(X_HEADING_RATE_TRUE));

    // if we are stationary
  } else {

    // position remains
    ret(X_X) = x(X_X);
    ret(X_Y) = x(X_Y);

    // speed should be 0
    ret(X_SPEED) = 0;

    // true heading reate is 0
    ret(X_HEADING_RATE_TRUE) = 0;

    // heading is an integration of the heading rate and bias
    ret(X_HEADING) = x(X_HEADING);
  }

  // bias compensation on?
  if (params.bias_compensation) {

    ret(X_HEADING_RATE_BIAS)   = x(X_HEADING_RATE_BIAS);
    ret(X_HEADING_RATE_BIASED) = x(X_HEADING_RATE_BIASED);

    // true heading rate
    ret(X_HEADING_RATE_TRUE) = x(X_HEADING_RATE_BIASED) - x(X_HEADING_RATE_BIAS);

  } else {


    ret(X_HEADING_RATE_BIAS)   = 0;
    ret(X_HEADING_RATE_BIASED) = 0;
    ret(X_HEADING_RATE_TRUE)   = x(X_HEADING_RATE_TRUE);
  }

  return ret;
}

//}

/* ukfmodelObservation() //{ */

ukf_t::z_t HeadingEstimTest::ukfModelObservation(const ukf_t::x_t& x) {

  auto params = mrs_lib::get_mutexed(mutex_params_, params_);

  z_t ret;

  ret(Z_X)     = x(X_X);
  ret(Z_Y)     = x(X_Y);
  ret(Z_SPEED) = x(X_SPEED);

  if (params.bias_compensation) {

    // if we are "moving" then expect a bias in gyro measurement
    if (sh_control_manager_diag_.getMsg()->tracker_status.have_goal) {

      ret(Z_HEADING_RATE) = x(X_HEADING_RATE_BIASED);

      // if we are "stationary", expect an "artificial sensor measurement with =0 rad/s"
    } else {

      ret(Z_HEADING_RATE) = x(X_HEADING_RATE_BIAS);
    }

  } else {
    ret(Z_HEADING_RATE) = x(X_HEADING_RATE_TRUE);
  }

  return ret;
}

//}

}  // namespace heading_estim_test

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(heading_estim_test::HeadingEstimTest, nodelet::Nodelet)
