# EKF Estimate Covariance Message Structure

# Position variance
float64 x
float64 y
float64 z

# Attitude variance
float64 roll
float64 pitch
float64 yaw

# Velocity variance
float64 vx
float64 vy
float64 vz

# Acceleration Bias variance
float64 bias_ax
float64 bias_ay
float64 bias_az

# Gyro Bias variance
float64 bias_gx
float64 bias_gy
float64 bias_gz
