pip install ultralytics

cd ~/zed_ws
ros2 topic echo /aruco_single/pose
# 建议删除旧的 build 记录以防万一，或者直接 build
colcon build --symlink-install --packages-select zed_components zed_wrapper

rm -rf build/ install/ log/
colcon build --symlink-install
source install/setup.bash
# 激活环境
source install/setup.bash
ros2 run rqt_image_view rqt_image_view

meshlab /home/lmz/graspnet/runs/zed_pointcloud_diagnose/camera_cloud_depth_range.ply
meshlab /home/lmz/graspnet/runs/zed_pointcloud_diagnose/camera_cloud_masked.ply
python zed_pointcloud_diagnose.py \
  --output_dir runs/zed_pointcloud_diagnose \
  --depth_mode NEURAL \
  --min_depth_m 0.15 \
  --max_depth_m 1.20 \
  --center_crop_ratio 0.45 \
  --largest_component_only \
  --min_component_pixels 300 \
  --component_selection center \
  --skip_robot
python zed_piper_grasp.py \
  --output_dir runs/zed_piper_grasp \
  --visualize
bash can_activate.sh can0 1000000
# 启动节点
ros2 run piper piper_single_ctrl --ros-args -p can_port:=can0 -p auto_enable:=false -p gripper_exist:=true -p rviz_ctrl_flag:=true
# 启动launch
ros2 launch piper start_single_piper.launch.py can_port:=can0 auto_enable:=false gripper_exist:=false rviz_ctrl_flag:=true
# 或，会以默认参数运行
ros2 launch piper start_single_piper.launch.py
# 也可以用rviz开启控制,需要更改的参数如上
ros2 launch piper start_single_piper_rviz.launch.py
启动 ZED
source ~/zed_ws/install/setup.bash 
ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed 
启动 ArUco
source ~/handeye/install/setup.bash 
ros2 run aruco_ros single --ros-args -p image_is_rectified:=true -p marker_size:=0.15 -p marker_id:=582 -p camera_frame:=zed_left_camera_frame_optical -p marker_frame:=aruco_marker_frame -r /image:=/zed/zed_node/rgb/color/rect/image -r /camera_info:=/zed/zed_node/rgb/color/rect/camera_info 
启动 Piper
bash ~/piper_ros/can_activate.sh 
source ~/piper_ros/install/setup.sh 
ros2 launch piper start_single_piper.launch.py can_port:=can0 
启动手眼标定
source ~/handeye/install/setup.bash 
ros2 run handeye_calibration_ros handeye_calibration --ros-args -p piper_topic:=/end_pose -p marker_topic:=/aruco_single/pose -p mode:=eye_in_hand
