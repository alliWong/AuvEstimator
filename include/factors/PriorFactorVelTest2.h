/**
 * @file PriorFactorVelTest2.h
 */

#pragma once

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Pose3.h>


namespace gtsam {

// template<class VALUE>
class PriorFactorVelTest2: public gtsam::NoiseModelFactor2<gtsam::Pose3, Vector> {

private:
  // measurement information
  const Vector b_velocity_;

public:

  /**
   * Constructor
   * @param poseKey     associated pose variable key
   * @param velKey      associated velocity variable key
   * @param model       noise model for sensor
   * @param b_velocity  body frame velocity measurement
   */
  PriorFactorVelTest2(gtsam::Key poseKey, gtsam::Key velKey, const Vector& b_velocity, gtsam::SharedNoiseModel model) :
      gtsam::NoiseModelFactor2<gtsam::Pose3, Vector>(model, poseKey, velKey), b_velocity_(b_velocity) {}

  // error function
  // @param p    the pose in Pose3
  // @param v    the velocity in Vector
  // @param H    the optional Jacobian matrix, which use boost optional and has default null pointer
  gtsam::Vector evaluateError(const gtsam::Pose3& p,
                              const Vector& v, 
                              boost::optional<gtsam::Matrix&> H1 = boost::none,
                              boost::optional<gtsam::Matrix&> H2 = boost::none) const {

    // Method 1
    gtsam::Matrix36 Hrot__pose; // pose H matrix
    gtsam::Rot3 w_R_b = p.rotation(Hrot__pose).inverse(); // Retrieve rotation from pose
    gtsam::Matrix33 Hvel__rot; // retrieve H velocity
    gtsam::Vector3 vec_b = w_R_b.rotate(v, Hvel__rot, H2); // transform world frame velocity into body frame
    if (H1) {
      *H1 = Hvel__rot * Hrot__pose; // compute corresponding H
    }
    // if (H2) *H2 = gtsam::Matrix33::Identity();
    // return error vector
    return (gtsam::Vector3() << vec_b - b_velocity_).finished(); // return velocity error

    // Method 2
    // if(H1&&H2)
    // {
    //   gtsam::Rot3 w_R_b = p.rotation(H1);
    //   gtsam::Matrix3 Hvel__rot;
    //   gtsam::Vector3 hx = w_R_b.rotate(v,Hvel__rot,H2);
    //   (*H1) = Hvel__rot* (*H1);
    //   return (hx-b_velocity_);
    // }
    // else
    // {
    //   gtsam::Rot3 w_R_b = p.rotation();
    //   gtsam::Vector3 hx = w_R_b.rotate(v,boost::none,H2);
    //   return (hx-b_velocity_);
    // }

  }

};

} // namespace gtsam


































// /**
//  * @file PriorFactorVelTest2.h
//  * A prior factor for velocity measurements in body/robot frame.
//  */

// #pragma once

// #include <gtsam/nonlinear/NonlinearFactor.h>
// #include <gtsam/base/Matrix.h>
// #include <gtsam/base/Vector.h>
// #include <gtsam/geometry/Point3.h>
// #include <gtsam/geometry/Pose3.h>
// #include <gtsam/base/numericalDerivative.h>


// namespace gtsam {

// // template<class VALUE>
// class PriorFactorVelTest2: public gtsam::NoiseModelFactor2<gtsam::Pose3, Vector> {

// private:
//   const gtsam::Point3 b_measVelocity_; // velocity measurement (body/robot frame)

//   // shorthand
//   typedef gtsam::NoiseModelFactor2<gtsam::Pose3, Vector> Base;

// public:

//   /**
//    * Constructor
//    * @param velKey          associated velocity varible key
//    * @param model           noise model for sensor
//    * @param b_measVelocity  body frame velocity measurement
//    */
//   PriorFactorVelTest2(gtsam::Key poseKey, gtsam::Key velKey, const Vector& b_measVelocity, gtsam::SharedNoiseModel model) :
//       Base(model, poseKey, velKey), b_measVelocity_(b_measVelocity) {}

//   /**
//    * Evaluate error
//    * @param p    pose rotation and translation containing information to transform a point 
//    * @param v    the velocity vector in navigatoin/world frame 
//    * @param H    the optional Jacobian matrix, which use boost optional and has default null pointer
//    */
//   gtsam::Vector evaluateError(const gtsam::Pose3& w_t_b,
//                               const Vector& vec_w, 
//                               boost::optional<gtsam::Matrix&> H1 = boost::none,
//                               boost::optional<gtsam::Matrix&> H2 = boost::none) const {
//     // ----- compute for body velocity ----- //
//     gtsam::Vector3 vec_b = bodyVelocity(w_t_b, vec_w);
//     // --- handle derivatives --- //
//     if (H1){
//       gtsam::Matrix36 derr_dx;
//       bodyVelocity(w_t_b, vec_w, derr_dx);
//       (*H1) = derr_dx;
//     }
//     if (H2){
//       // gtsam::Matrix33 dx;
//       // bodyVelocity(w_t_b, b_measVelocity_, dx);
//       (*H2) = gtsam::Matrix33::Identity();
//     }
    
//     std::cout << "\n*****VELOCITY FACTOR EVAL*****" << std::endl;
//     std::cout << "B_PRED_VEL: \n" << vec_b << std::endl;
//     std::cout << "B_MEAS_DVL: \n" << b_measVelocity_ << std::endl;
//     std::cout << "B_ERR_VEL: \n" << vec_b - b_measVelocity_ << std::endl;
//     std::cout << "N_PRED_VEL: \n" << vec_w << std::endl;
//     std::cout << "N_PRED_POSE: \n" << w_t_b << std::endl;

//     // ----- compute for error ----- //
//     gtsam::Vector3 errVec;
//     errVec = vec_b - b_measVelocity_;
//     return errVec;
//   }

//   /**
//    * Error model function
//    * @param p    pose rotation and translation containing information to transform a point 
//    * @param v    the velocity vector in navigatoin/world frame 
//    * @param H    the optional Jacobian matrix, which use boost optional and has default null pointer
//    */
//   gtsam::Vector3 bodyVelocity(const Pose3& w_t_b,
//                               const Vector3& vec_w,
//                               OptionalJacobian<3, 6> Hpose = boost::none,
//                               OptionalJacobian<3, 3> Hvel = boost::none) const {
//     Matrix36 Hrot__pose;
//     Rot3 w_R_b = w_t_b.rotation(Hrot__pose);
//     Matrix33 Hvel__rot;
//     Vector3 vec_b = w_R_b.unrotate(vec_w, Hvel__rot, Hvel);
//     if (Hpose) {
//       *Hpose = Hvel__rot * Hrot__pose;
//     }
//     return vec_b;
   


//     // // Method 2
//     // if (Hpose)
//     // {
//     //   gtsam::Matrix36 Hrot__pose;
//     //   gtsam::Rot3 w_R_b = w_t_b.rotation(Hrot__pose); // R[B->N] --> RzRyRx
//     //   gtsam::Matrix33 Hvel__rot;
//     //   gtsam::Vector3 vec_b = w_R_b.unrotate(vec_w, Hvel__rot, boost::none);
//     //   (*Hpose) = Hvel__rot * Hrot__pose;
//     //   return vec_b;
//     // }
//     // else
//     // {
//     //   gtsam::Rot3 w_R_b = w_t_b.rotation(); // R[B->N] --> RzRyRx
//     //   gtsam::Vector3 vec_b = w_R_b.unrotate(vec_w, boost::none, boost::none);
//     //   return vec_b;
//     // }

//   }

// };

// } // namespace gtsam


// // Method 2 separate velocity and rotation