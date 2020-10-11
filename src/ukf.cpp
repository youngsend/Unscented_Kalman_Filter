#include <iostream>
#include "ukf.h"
#include "Eigen/Dense"

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // state dimension is 5: px, py, v, yaw, yaw_rate
  n_x_ = 5;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd::Identity(n_x_, n_x_);
  // finally, the following two parameters handled the overshoot of vy.
  P_(3, 3) = 0.15;
  P_(4, 4) = 0.15;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 3;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 1;

  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
   * End DO NOT MODIFY section for measurement noise values
   */

  is_initialized_ = false;

  // state dimension + longitudinal acc noise, yaw_acc noise, sensorタイプに関係ないはず
  n_aug_ = 7;
  lambda_ = 3 - n_aug_;

  // weights initialization, used in mean and covariance calculation from predicted sigma points.
  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_.fill(0.5 / (lambda_ + n_aug_));
  weights_[0] = lambda_ / (lambda_ + n_aug_);

  // radar measurement noise covariance
  R_radar_ = MatrixXd(n_z_radar, n_z_radar);
  R_radar_ << std_radr_*std_radr_, 0, 0,
          0, std_radphi_*std_radphi_, 0,
          0, 0, std_radrd_*std_radrd_;

  // lidar measurement model H_
  H_lidar_ = MatrixXd(n_z_lidar, n_x_);
  H_lidar_ << 1, 0, 0, 0, 0,
          0, 1, 0, 0, 0;

  // lidar measurement noise covariance matrix
  R_lidar_ = MatrixXd(n_z_lidar, n_z_lidar);
  R_lidar_ << std_laspx_*std_laspx_, 0,
          0, std_laspy_*std_laspy_;
}

UKF::~UKF() {}

/**
 * 1. If not initialized, initialize x_ using measurements and update timestamp.
 * 2. UKF Prediction step.
 * 3. UKF Update step for radar and normal KF Update for lidar.
 * @param meas_package
 */
void UKF::ProcessMeasurement(const MeasurementPackage& meas_package) {
  /**
   * measurements.
   */
  if (!is_initialized_) {
    // initialize state with measurement if uninitialized.
    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      // lidarだったら
      x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
      P_(0, 0) = std_laspx_ * std_laspx_;
      P_(1, 1) = std_laspy_ * std_laspy_;
    } else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      auto rho = meas_package.raw_measurements_[0];
      auto phi = meas_package.raw_measurements_[1];
      x_ << rho * cos(phi), rho * sin(phi), 0, 0, 0;
      P_(0, 0) = std_radr_ * std_radr_;
      P_(1, 1) = std_radr_ * std_radr_;
    }
    is_initialized_ = true;
    time_us_ = meas_package.timestamp_;
    return;
  }

  // prediction step using motion model
  auto delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;
  Prediction(delta_t);
  time_us_ = meas_package.timestamp_;

  // update step
  if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
    UpdateLidar(meas_package);
  } else if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
    UpdateRadar(meas_package);
  }
}

/**
 * UKF Prediction step:
 * 1. Augment state and covariance, and generate sigma points using previous state and covariance.
 * 2. Process sigma points through ctrv motion model, get predicted sigma points after delta_t.
 * 3. Update state and covariance using predicted sigma points.
 * @param delta_t
 */
void UKF::Prediction(double delta_t) {
  // 1. Augmentation initialization and sigma points generation
  MatrixXd Xsig_aug = AugmentationAndSigmaPointsGeneration(x_, P_);

  // 2. Predict sigma points using ctrv motion model
  Xsig_pred_ = SigmaPointsPrediction(Xsig_aug, delta_t);

  // 3. Calculate and update mean and covariance from predicted sigma points
  UKFPredict(Xsig_pred_, x_, P_);

  std::cout << "After UKF Prediction" << x_ << std::endl;

}

/**
 * KF update step for linear lidar measurement model.
 * @param meas_package
 */
void UKF::UpdateLidar(const MeasurementPackage& meas_package) {
  // lidar measurement model is linear, so use the normal kalman filter update step.
  VectorXd z_pred = H_lidar_ * x_;
  VectorXd y = meas_package.raw_measurements_ - z_pred;
  MatrixXd S = H_lidar_ * P_ * H_lidar_.transpose() + R_lidar_;
  MatrixXd K = P_ * H_lidar_.transpose() * S.inverse();

  // new estimate
  x_ = x_ + K * y;
  MatrixXd I = MatrixXd::Identity(n_x_, n_x_);
  P_ = (I - K * H_lidar_) * P_;

  std::cout << "After Lidar KF Update" << x_ << std::endl;
}

/**
 * UKF update step for nonlinear radar measurement model.
 * 1. Convert state space sigma points to measurement space sigma points.
 * 2. Calculate mean and covariance of sigma points in measurement space.
 * 3. Update state and state covariance using measurement space sigma points and measurement.
 * @param meas_package
 */
void UKF::UpdateRadar(const MeasurementPackage& meas_package) {
  // 1. predict radar measurements from predicted sigma points
  // matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z_radar, 2 * n_aug_ + 1);
  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z_radar);
  // innovation covariance matrix
  MatrixXd S = MatrixXd(n_z_radar, n_z_radar);
  PredictRadarMeasurements(Zsig, z_pred, S);

  // 2. UKF update using radar measurement
  UKFUpdate(Zsig, z_pred, S, meas_package.raw_measurements_, x_, P_);

  std::cout << "After Radar UKF update" << x_ << std::endl;
}

MatrixXd UKF::AugmentationAndSigmaPointsGeneration(const VectorXd& x, const MatrixXd& P) {
  VectorXd x_aug = VectorXd(n_aug_);
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  // sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  // augmented mean state初期化
  x_aug.head(n_x_) = x;
  x_aug(5) = 0;
  x_aug(6) = 0;

  // augmented covariance matrix初期化
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P;
  P_aug.bottomRightCorner(2, 2) << std_a_*std_a_, 0, 0, std_yawdd_*std_yawdd_;

  // square root matrix of covariance matrix
  MatrixXd A = P_aug.llt().matrixL();

  // augmented sigma points
  Xsig_aug.col(0) = x_aug;
  for(int i=0; i<n_aug_; i++){
    Xsig_aug.col(i+1) = x_aug + sqrt(3) * A.col(i);
    Xsig_aug.col(i+n_aug_+1) = x_aug - sqrt(3) * A.col(i);
  }

  return Xsig_aug;
}

MatrixXd UKF::SigmaPointsPrediction(const MatrixXd& Xsig_aug, double delta_t) {
  // ctrv motion model:
  // https://github.com/youngsend/LearningSelfDrivingCars/blob/master/Sensor-Fusion_Udacity/Kalman-Filters/l4-Unscented-Kalman-Filters.md#sigma-point-prediction

  MatrixXd Xsig_pred = MatrixXd(n_x_, 2 * n_aug_ + 1);

  for (int i = 0; i < 2*n_aug_+1; i++){
    // extract values for better readability
    double px = Xsig_aug(0, i);
    double py = Xsig_aug(1, i);
    double v = Xsig_aug(2, i);
    double yaw = Xsig_aug(3, i);
    double yawd = Xsig_aug(4, i);
    double nu_a = Xsig_aug(5, i);
    double nu_yawdd = Xsig_aug(6, i);

    // predicted state values
    double px_p, py_p;

    // avoid division by zero
    if (fabs(yawd) > 0.001) {
      px_p = px + v/yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
      py_p = py + v/yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
    } else {
      px_p = px + v * cos(yaw) * delta_t;
      py_p = py + v * sin(yaw) * delta_t;
    }

    double v_p = v;
    double yaw_p = yaw + yawd * delta_t;
    double yawd_p = yawd;

    // add noise
    px_p = px_p + 0.5 * delta_t * delta_t * cos(yaw) * nu_a;
    py_p = py_p + 0.5 * delta_t * delta_t * sin(yaw) * nu_a;
    v_p = v_p + delta_t * nu_a;

    yaw_p = yaw_p + 0.5 * delta_t * delta_t * nu_yawdd;
    yawd_p = yawd_p + delta_t * nu_yawdd;

    // write predicted sigma point into right column
    Xsig_pred(0, i) = px_p;
    Xsig_pred(1, i) = py_p;
    Xsig_pred(2, i) = v_p;
    Xsig_pred(3, i) = yaw_p;
    Xsig_pred(4, i) = yawd_p;
  }

  return Xsig_pred;
}

void UKF::UKFPredict(const MatrixXd &Xsig_pred, VectorXd& x_out, MatrixXd& P_out) {
  // predict state mean
  VectorXd x = Xsig_pred * weights_;

  MatrixXd P = MatrixXd(n_x_, n_x_);
  P.fill(0.0);
  // predict state covariance matrix
  for(int i=0; i<Xsig_pred.cols(); i++){
    // state difference
    VectorXd x_diff = Xsig_pred.col(i) - x;
    // angle normalization
    x_diff(3) = NormalizeAngle(x_diff(3));

    P = P + weights_[i] * x_diff * x_diff.transpose();
  }

  // Update state and covariance
  x_out = x;
  P_out = P;
}

/**
 * Radar measurement model:
 * https://github.com/youngsend/LearningSelfDrivingCars/blob/master/Sensor-Fusion_Udacity/Kalman-Filters/l4-Unscented-Kalman-Filters.md#measurement-prediction
 * @param z_pred
 * @param S
 */
void UKF::PredictRadarMeasurements(MatrixXd& Zsig, VectorXd &z_pred, MatrixXd &S) {
  // transform sigma points into measurement space
  for(int i=0; i<2*n_aug_+1; i++){
    // extract values for better readability
    double px = Xsig_pred_(0, i);
    double py = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);

    double v1 = cos(yaw) * v;
    double v2 = sin(yaw) * v;

    // radar measurement model: rho, phi, rho_dot
    Zsig(0, i) = sqrt(px * px + py * py);
    Zsig(1, i) = atan2(py, px);
    Zsig(2, i) = (px * v1 + py * v2) / sqrt(px * px + py * py);
  }

  // mean predicted measurement
  z_pred = Zsig * weights_;

  S.fill(0.0);
  // innovation covariance matrix S
  for (int i=0; i<2*n_aug_+1; i++){
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // angle normalization
    z_diff(1) = NormalizeAngle(z_diff(1));

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  // add measurement noise covariance matrix
  S = S + R_radar_;
}

/**
 * UKF update: 各センサー共用する
 * https://github.com/youngsend/LearningSelfDrivingCars/blob/master/Sensor-Fusion_Udacity/Kalman-Filters/l4-Unscented-Kalman-Filters.md#ukf-update
 * @param z_pred
 * @param S
 * @param z
 * @param x
 * @param P
 */
void UKF::UKFUpdate(const MatrixXd &Zsig, const VectorXd &z_pred, const MatrixXd &S,
                    const VectorXd &z, VectorXd &x, MatrixXd &P) {
  // matrix for cross correlation between sigma points in state space and measurement space
  MatrixXd Tc = MatrixXd(n_x_, z.size());
  Tc.fill(0.0);

  // calculate cross correlation matrix
  for(int i=0; i<2*n_aug_+1; i++){
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    // angle normalization
    z_diff(1) = NormalizeAngle(z_diff(1));

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x;
    // angle normalization
    x_diff(3) = NormalizeAngle(x_diff(3));

    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  // Kalman gain K
  MatrixXd K = Tc * S.inverse();

  // residual
  VectorXd z_diff = z - z_pred;
  // angle normalization
  z_diff(1) = NormalizeAngle(z_diff(1));

  // update state mean and covariance matrix
  x = x + K * z_diff;
  P = P - K * S * K.transpose();
}

double UKF::NormalizeAngle(double angle_rad) {
  // angle normalization
  while (angle_rad > M_PI)
    angle_rad -= 2*M_PI;
  while (angle_rad < -M_PI)
    angle_rad += 2*M_PI;
  return angle_rad;
}
