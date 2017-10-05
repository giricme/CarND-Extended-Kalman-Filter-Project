#include "kalman_filter.h"
#include <iostream>
#include "tools.h"
using namespace std;

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
   * predict the state
   */
    x_ = F_*x_;
    P_ = F_*P_*F_.transpose() + Q_;
}

void KalmanFilter::NormalizeAngle(double& phi) {
    phi = atan2(sin(phi), cos(phi));
}

void KalmanFilter::UpdateCommon(const VectorXd &y) {
    const MatrixXd PHt = P_ * H_.transpose();
    const MatrixXd S = H_ * PHt + R_;
    const MatrixXd K = PHt * S.inverse();
    
    x_ += K * y;
    P_ -= K * H_ * P_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
   * update the state by using Kalman Filter equations
   */
    VectorXd z_pred = H_ * x_;
    VectorXd y = z - z_pred;
    
    UpdateCommon(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
   * update the state by using Extended Kalman Filter equations
   */
    float px = x_(0);
    float py = x_(1);
    float vx = x_(2);
    float vy = x_(3);
    float rho = sqrt(px*px + py*py);
    float phi = 0.0;
    if (py != 0 and px != 0) {
        phi = atan2(py, px);
    }
    float rho_dot;
    if (fabs(rho) < EPSILON2) {
        rho_dot = EPSILON2;
    } else {
        rho_dot = (px*vx + py*vy)/rho;
    }
    VectorXd z_pred(3);
    z_pred << rho, phi, rho_dot;

    VectorXd y = z - z_pred;
    NormalizeAngle(y(1));
    UpdateCommon(y);
}
