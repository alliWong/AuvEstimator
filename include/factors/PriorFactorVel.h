/**
 * @file PriorFactorVel.h
 */

#pragma once

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Pose3.h>


namespace gtsam {

class PriorFactorVel: public gtsam::NoiseModelFactor2<gtsam::Pose3, Vector> {

private:

  // measurement
  const Vector b_velocity_;

public:

  /**
   * Constructor
   * @param poseKey     associated pose variable key
   * @param velKey      associated velocity variable key
   * @param model       noise model for sensor
   * @param b_velocity  body frame velocity measurement
  **/
  PriorFactorVel(gtsam::Key poseKey, gtsam::Key velKey, const Vector& b_velocity, gtsam::SharedNoiseModel model) :
      gtsam::NoiseModelFactor2<gtsam::Pose3, Vector>(model, poseKey, velKey), b_velocity_(b_velocity) {}

  // Error function
  // @param p    the pose in Pose3
  // @param v    the velocity in Vector
  // @param H    the optional Jacobian matrix, which use boost optional and has default null pointer
  gtsam::Vector evaluateError(const gtsam::Pose3& p,
                              const Vector& v, 
                              boost::optional<gtsam::Matrix&> H1 = boost::none,
                              boost::optional<gtsam::Matrix&> H2 = boost::none) const {
    // ---------- Error Model ---------- //
    // Error = estimated_body_vel - measured_body_vel 
    // estimated_body_vel = inv(R[B->N]) * n_vel

    // ---------- Method 1 ---------- //
    gtsam::Matrix36 Hrot__pose; // drot/dx
    gtsam::Rot3 w_R_b = p.rotation(Hrot__pose); // inv(R[B->N]) = R[N->B]
    gtsam::Matrix33 Hvel__rot; // dvel/drot
    gtsam::Vector3 vec_b = w_R_b.unrotate(v, Hvel__rot, H2); // transform world frame velocity into body frame
    if (H1) *H1 = Hvel__rot * Hrot__pose; // derr/dx

    // std::cout << "\n*****VELOCITY FACTOR EVAL*****" << std::endl;
    // std::cout << "POSE: \n" << p << std::endl;
    // std::cout << "VEL: \n" << v << std::endl;

    // return error vector
    return (gtsam::Vector3() << vec_b - b_velocity_).finished(); // return velocity error

    // ---------- Method 2 ---------- //
    // if(H1&&H2)
    // {
    //   gtsam::Rot3 w_R_b = p.rotation(H1);
    //   gtsam::Matrix3 Hvel__rot;
    //   gtsam::Vector3 hx = w_R_b.unrotate(v,Hvel__rot,H2);
    //   (*H1) = Hvel__rot* (*H1);
    //   return (hx-b_velocity_);
    // }
    // else
    // {
    //   gtsam::Rot3 w_R_b = p.rotation();
    //   gtsam::Vector3 hx = w_R_b.unrotate(v,boost::none,H2);
    //   return (hx-b_velocity_);
    // }

  }

};

} // namespace gtsam
