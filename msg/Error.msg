# EKF error message structure

# Header
Header header

# Error
# Position
float64 x
float64 y
float64 z

# Attitude
float64 roll
float64 pitch
float64 yaw

# Velocity
float64 vx
float64 vy
float64 vz

# RMSE
# Position
float64 rmse_x
float64 rmse_y
float64 rmse_z

# Attitude
float64 rmse_roll
float64 rmse_pitch
float64 rmse_yaw

# Velocity
float64 rmse_vx
float64 rmse_vy
float64 rmse_vz

# Distance evaluation
float64 dist_error
float64 dist_traveled
float64 rmse_dist
