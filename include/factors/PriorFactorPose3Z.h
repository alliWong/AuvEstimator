/**
 * @file PriorFactorPose3Z.h
 * A prior factor for linear position in the z-axis measurements in navigation/global frame.
 */

#pragma once

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Pose3.h>


namespace gtsam {

/**
  * Prior linear z position factor
*/

class PriorFactorPose3Z: public gtsam::NoiseModelFactor1<gtsam::Pose3> {

private:
  // measurement information
  double measZ_;

public:

  /**
   * Constructor
   * @param poseKey    associated pose varible key
   * @param model      noise model for sensor
   * @param measZ      double z measurement
   */
  PriorFactorPose3Z(gtsam::Key poseKey, const double& measZ, gtsam::SharedNoiseModel model) :
      gtsam::NoiseModelFactor1<gtsam::Pose3>(model, poseKey), measZ_(measZ) {}

  // error function
  // @param p    the pose in Pose3
  // @param H    the optional Jacobian matrix, which use boost optional and has default null pointer
  gtsam::Vector evaluateError(const gtsam::Pose3& p, boost::optional<gtsam::Matrix&> H = boost::none) const {
  
    // note that use boost optional like a pointer
    // only calculate jacobian matrix when non-null pointer exists
    if (H) *H = (gtsam::Matrix16() << 0.0, 0.0, 0.0, 0.0, 0.0, 1.0).finished();
    
    // return error vector
    return (gtsam::Vector1() << p.z() - measZ_).finished();
  }

};

} // namespace gtsam
