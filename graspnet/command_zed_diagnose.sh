export LD_LIBRARY_PATH="/usr/local/zed/lib:${LD_LIBRARY_PATH}"

# Step 1: only verify ZED depth and the camera-frame point cloud.
python zed_pointcloud_diagnose.py \
  --output_dir runs/zed_pointcloud_diagnose \
  --depth_mode QUALITY \
  --min_depth_m 0.15 \
  --max_depth_m 1.20 \
  --center_crop_ratio 0.45 \
  --largest_component_only \
  --min_component_pixels 300 \
  --component_selection center \
  --skip_robot

# Step 2: if the camera-frame cloud looks correct, compare both hand-eye directions.
# python zed_pointcloud_diagnose.py \
#   --output_dir runs/zed_pointcloud_diagnose_compare \
#   --depth_mode QUALITY \
#   --min_depth_m 0.15 \
#   --max_depth_m 1.20 \
#   --center_crop_ratio 0.45 \
#   --largest_component_only \
#   --min_component_pixels 300 \
#   --component_selection center \
#   --calibration 2026-04-02_16-16-43_calibration.json \
#   --compare_directions
