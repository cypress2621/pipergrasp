"""Probe a clicked ZED pixel and transform it into the robot base frame."""

import argparse
import json
import os
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from piper_sdk import C_PiperInterface


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="runs/zed_handeye_probe")
    parser.add_argument("--resolution", choices=["HD720", "HD1080"], default="HD720")
    parser.add_argument("--depth_mode", choices=["PERFORMANCE", "QUALITY", "ULTRA", "NEURAL", "NEURAL_PLUS"], default="QUALITY")
    parser.add_argument("--calibration", default="2026-04-02_16-16-43_calibration.json")
    parser.add_argument("--calibration_direction", choices=["ee_to_camera", "camera_to_ee"], default="ee_to_camera")
    parser.add_argument("--can_name", default="can0")
    parser.add_argument("--click_x", type=int, default=-1)
    parser.add_argument("--click_y", type=int, default=-1)
    return parser.parse_args()


def capture_zed_frame(output_dir, resolution, depth_mode):
    import pyzed.sl as sl

    os.makedirs(output_dir, exist_ok=True)
    zed = sl.Camera()
    init_params = sl.InitParameters()
    depth_mode_map = {
        "PERFORMANCE": sl.DEPTH_MODE.PERFORMANCE,
        "QUALITY": sl.DEPTH_MODE.QUALITY,
        "ULTRA": sl.DEPTH_MODE.ULTRA,
        "NEURAL": sl.DEPTH_MODE.NEURAL,
        "NEURAL_PLUS": sl.DEPTH_MODE.NEURAL_PLUS,
    }
    init_params.depth_mode = depth_mode_map[depth_mode]
    init_params.coordinate_units = sl.UNIT.METER
    init_params.sdk_verbose = False
    init_params.camera_resolution = sl.RESOLUTION.HD1080 if resolution == "HD1080" else sl.RESOLUTION.HD720

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"ZED 打开失败: {repr(err)}")

    image = sl.Mat()
    depth = sl.Mat()
    runtime = sl.RuntimeParameters()
    for _ in range(10):
        zed.grab(runtime)

    if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
        zed.close()
        raise RuntimeError("ZED 抓帧失败。")

    zed.retrieve_image(image, sl.VIEW.LEFT)
    zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
    color_bgra = image.get_data()
    depth_m = depth.get_data().astype(np.float32)
    calib = zed.get_camera_information().camera_configuration.calibration_parameters.left_cam
    zed.close()

    if color_bgra.ndim == 3 and color_bgra.shape[2] == 4:
        color_bgr = cv2.cvtColor(color_bgra, cv2.COLOR_BGRA2BGR)
    else:
        color_bgr = color_bgra.copy()

    depth_m[~np.isfinite(depth_m)] = 0.0
    depth_m[depth_m < 0] = 0.0

    cv2.imwrite(os.path.join(output_dir, "probe_rgb.png"), color_bgr)
    np.save(os.path.join(output_dir, "probe_depth.npy"), depth_m)
    return color_bgr, depth_m, calib


def get_live_base_T_ee(can_name):
    piper = C_PiperInterface(can_name=can_name)
    piper.ConnectPort()
    time.sleep(1.0)
    pose = piper.GetArmEndPoseMsgs().end_pose

    xyz = np.array([pose.X_axis, pose.Y_axis, pose.Z_axis], dtype=np.float64) / 1e6
    rpy = np.deg2rad(np.array([pose.RX_axis, pose.RY_axis, pose.RZ_axis], dtype=np.float64) / 1000.0)

    base_T_ee = np.eye(4, dtype=np.float64)
    base_T_ee[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
    base_T_ee[:3, 3] = xyz
    return base_T_ee


def load_calibration_transform(calibration_path):
    with open(calibration_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    position = np.asarray(data["position"], dtype=np.float64)
    quaternion = np.asarray(data["orientation"], dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = R.from_quat(quaternion).as_matrix()
    transform[:3, 3] = position
    return transform


def build_base_T_camera(base_T_ee, handeye_transform, calibration_direction):
    ee_T_camera = handeye_transform if calibration_direction == "ee_to_camera" else np.linalg.inv(handeye_transform)
    return base_T_ee @ ee_T_camera


def pixel_to_camera_xyz(u, v, depth, calib):
    z = float(depth[v, u])
    if z <= 0:
        raise RuntimeError(f"像素 ({u}, {v}) 的深度无效。")
    x = (u - float(calib.cx)) * z / float(calib.fx)
    y = (v - float(calib.cy)) * z / float(calib.fy)
    return np.array([x, y, z], dtype=np.float64)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    color_bgr, depth_m, calib = capture_zed_frame(args.output_dir, args.resolution, args.depth_mode)
    h, w = depth_m.shape

    if args.click_x < 0 or args.click_y < 0:
        u = w // 2
        v = h // 2
    else:
        u = max(0, min(args.click_x, w - 1))
        v = max(0, min(args.click_y, h - 1))

    color_marked = color_bgr.copy()
    cv2.circle(color_marked, (u, v), 8, (0, 0, 255), -1)
    cv2.imwrite(os.path.join(args.output_dir, "probe_rgb_marked.png"), color_marked)

    cam_xyz = pixel_to_camera_xyz(u, v, depth_m, calib)
    base_T_ee = get_live_base_T_ee(args.can_name)
    handeye_transform = load_calibration_transform(args.calibration)
    base_T_camera = build_base_T_camera(base_T_ee, handeye_transform, args.calibration_direction)

    base_xyz = base_T_camera[:3, :3] @ cam_xyz + base_T_camera[:3, 3]
    result = {
        "pixel_uv": [int(u), int(v)],
        "camera_xyz_m": [float(x) for x in cam_xyz],
        "base_xyz_m": [float(x) for x in base_xyz],
        "calibration_direction": args.calibration_direction,
    }

    with open(os.path.join(args.output_dir, "probe_result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
