"""Capture and diagnose ZED point clouds before running grasp inference."""

import argparse
import json
import os
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy import ndimage
from scipy.spatial.transform import Rotation as R

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, "utils"))

from data_utils import CameraInfo, create_point_cloud_from_depth_image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="runs/zed_pointcloud_diagnose")
    parser.add_argument("--resolution", choices=["HD720", "HD1080"], default="HD720")
    parser.add_argument(
        "--depth_mode",
        choices=["PERFORMANCE", "QUALITY", "ULTRA", "NEURAL", "NEURAL_PLUS"],
        default="QUALITY",
    )
    parser.add_argument("--min_depth_m", type=float, default=0.15)
    parser.add_argument("--max_depth_m", type=float, default=1.20)
    parser.add_argument("--center_crop_ratio", type=float, default=1.0)
    parser.add_argument("--roi_xmin", type=float, default=0.0)
    parser.add_argument("--roi_xmax", type=float, default=1.0)
    parser.add_argument("--roi_ymin", type=float, default=0.0)
    parser.add_argument("--roi_ymax", type=float, default=1.0)
    parser.add_argument("--largest_component_only", action="store_true")
    parser.add_argument("--min_component_pixels", type=int, default=300)
    parser.add_argument("--component_selection", choices=["largest", "center"], default="center")
    parser.add_argument("--full_cloud_voxel", type=float, default=0.005)
    parser.add_argument("--masked_cloud_voxel", type=float, default=0.003)
    parser.add_argument("--skip_robot", action="store_true")
    parser.add_argument("--calibration", default="2026-04-02_16-16-43_calibration.json")
    parser.add_argument(
        "--calibration_direction",
        choices=["ee_to_camera", "camera_to_ee"],
        default="ee_to_camera",
    )
    parser.add_argument("--compare_directions", action="store_true")
    parser.add_argument("--can_name", default="can0")
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
    init_params.camera_resolution = (
        sl.RESOLUTION.HD1080 if resolution == "HD1080" else sl.RESOLUTION.HD720
    )

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
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    depth_m[~np.isfinite(depth_m)] = 0.0
    depth_m[depth_m < 0] = 0.0

    camera = CameraInfo(
        int(depth_m.shape[1]),
        int(depth_m.shape[0]),
        float(calib.fx),
        float(calib.fy),
        float(calib.cx),
        float(calib.cy),
        1.0,
    )

    cv2.imwrite(os.path.join(output_dir, "color.png"), color_bgr)
    np.save(os.path.join(output_dir, "depth_m.npy"), depth_m)
    with open(os.path.join(output_dir, "camera_intrinsics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "width": int(camera.width),
                "height": int(camera.height),
                "fx": float(camera.fx),
                "fy": float(camera.fy),
                "cx": float(camera.cx),
                "cy": float(camera.cy),
                "scale": float(camera.scale),
            },
            f,
            indent=2,
        )

    return color_rgb, depth_m, camera


def compute_input_mask(depth, center_crop_ratio, min_depth_m, max_depth_m, args):
    mask = (depth > min_depth_m) & (depth < max_depth_m)
    crop_bounds = []

    if center_crop_ratio < 1.0:
        height, width = depth.shape
        crop_ratio = max(0.05, min(center_crop_ratio, 1.0))
        crop_w = int(width * crop_ratio)
        crop_h = int(height * crop_ratio)
        x0 = max(0, (width - crop_w) // 2)
        y0 = max(0, (height - crop_h) // 2)
        x1 = min(width, x0 + crop_w)
        y1 = min(height, y0 + crop_h)
        crop_mask = np.zeros_like(mask, dtype=bool)
        crop_mask[y0:y1, x0:x1] = True
        mask &= crop_mask
        crop_bounds.append(((x0, y0, x1, y1), (0, 255, 0)))

    height, width = depth.shape
    rx0 = int(width * max(0.0, min(args.roi_xmin, 1.0)))
    rx1 = int(width * max(0.0, min(args.roi_xmax, 1.0)))
    ry0 = int(height * max(0.0, min(args.roi_ymin, 1.0)))
    ry1 = int(height * max(0.0, min(args.roi_ymax, 1.0)))
    if rx1 > rx0 and ry1 > ry0 and (
        args.roi_xmin > 0.0 or args.roi_xmax < 1.0 or args.roi_ymin > 0.0 or args.roi_ymax < 1.0
    ):
        roi_mask = np.zeros_like(mask, dtype=bool)
        roi_mask[ry0:ry1, rx0:rx1] = True
        mask &= roi_mask
        crop_bounds.append(((rx0, ry0, rx1, ry1), (255, 0, 0)))

    return mask, crop_bounds


def keep_depth_component(mask, min_component_pixels, selection_mode):
    labeled, num = ndimage.label(mask.astype(np.uint8))
    if num <= 0:
        return mask

    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    valid_labels = [label for label in range(1, num + 1) if counts[label] >= min_component_pixels]
    if not valid_labels:
        return mask

    if selection_mode == "largest":
        best_label = max(valid_labels, key=lambda label: counts[label])
        return labeled == best_label

    height, width = mask.shape
    center_y = (height - 1) / 2.0
    center_x = (width - 1) / 2.0

    def center_score(label):
        ys, xs = np.where(labeled == label)
        cy = float(np.mean(ys))
        cx = float(np.mean(xs))
        dist2 = (cx - center_x) ** 2 + (cy - center_y) ** 2
        area_bonus = counts[label] * 1e-3
        return dist2 - area_bonus

    best_label = min(valid_labels, key=center_score)
    return labeled == best_label


def save_input_debug_figure(output_dir, color_rgb, depth_m, input_mask, crop_bounds):
    color_vis = (np.clip(color_rgb, 0.0, 1.0) * 255).astype(np.uint8)
    depth_vis = depth_m.copy()
    valid_depth = depth_vis[depth_vis > 0]
    if valid_depth.size > 0:
        depth_max = np.percentile(valid_depth, 95)
        depth_vis = np.clip(depth_vis / max(depth_max, 1e-6), 0.0, 1.0)
    overlay = color_vis.copy()
    overlay[~input_mask] = (overlay[~input_mask] * 0.25).astype(np.uint8)

    for bounds, color in crop_bounds:
        x0, y0, x1, y1 = bounds
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 2)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(color_vis)
    axes[0].set_title("RGB")
    axes[1].imshow(depth_vis, cmap="viridis")
    axes[1].set_title("Depth")
    axes[2].imshow(input_mask, cmap="gray")
    axes[2].set_title("Input Mask")
    axes[3].imshow(overlay)
    axes[3].set_title("RGB + Mask Overlay")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig_path = os.path.join(output_dir, "input_debug.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path


def save_depth_visualization(output_dir, depth_m):
    depth_vis = depth_m.copy()
    valid_depth = depth_vis[depth_vis > 0]
    if valid_depth.size > 0:
        upper = max(np.percentile(valid_depth, 95), 1e-6)
        depth_vis = np.clip(depth_vis / upper, 0.0, 1.0)
    depth_u8 = (depth_vis * 255.0).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_VIRIDIS)
    path = os.path.join(output_dir, "depth_vis.png")
    cv2.imwrite(path, depth_color)
    return path


def make_open3d_cloud(points, colors=None, voxel_size=0.0):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float32))
    if colors is not None:
        cloud.colors = o3d.utility.Vector3dVector(np.clip(colors.astype(np.float32), 0.0, 1.0))
    if voxel_size > 0:
        cloud = cloud.voxel_down_sample(voxel_size)
    return cloud


def save_cloud(path, points, colors=None, voxel_size=0.0):
    if len(points) == 0:
        return None
    cloud = make_open3d_cloud(points, colors, voxel_size)
    o3d.io.write_point_cloud(path, cloud)
    return path


def summarize_depth(depth_m, valid_mask):
    valid_depth = depth_m[valid_mask]
    summary = {
        "image_height": int(depth_m.shape[0]),
        "image_width": int(depth_m.shape[1]),
        "valid_depth_pixels": int(np.count_nonzero(valid_mask)),
        "valid_depth_ratio": float(np.count_nonzero(valid_mask) / depth_m.size),
    }
    if valid_depth.size == 0:
        return summary

    summary.update(
        {
            "depth_min_m": float(valid_depth.min()),
            "depth_median_m": float(np.median(valid_depth)),
            "depth_p95_m": float(np.percentile(valid_depth, 95)),
            "depth_max_m": float(valid_depth.max()),
        }
    )
    return summary


def summarize_points(points):
    if len(points) == 0:
        return {
            "count": 0,
        }

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    mean = points.mean(axis=0)
    return {
        "count": int(len(points)),
        "min_xyz_m": [float(v) for v in mins],
        "max_xyz_m": [float(v) for v in maxs],
        "mean_xyz_m": [float(v) for v in mean],
    }


def load_calibration_transform(calibration_path):
    with open(calibration_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    position = np.asarray(data["position"], dtype=np.float64)
    quaternion = np.asarray(data["orientation"], dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = R.from_quat(quaternion).as_matrix()
    transform[:3, 3] = position
    return transform


def get_live_base_T_ee(can_name):
    from piper_sdk import C_PiperInterface

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


def build_base_T_camera(base_T_ee, handeye_transform, calibration_direction):
    if calibration_direction == "ee_to_camera":
        ee_T_camera = handeye_transform
    else:
        ee_T_camera = np.linalg.inv(handeye_transform)
    return base_T_ee @ ee_T_camera


def transform_points(points, transform):
    rotated = points @ transform[:3, :3].T
    return rotated + transform[:3, 3]


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    color_rgb, depth_m, camera = capture_zed_frame(args.output_dir, args.resolution, args.depth_mode)
    cloud = create_point_cloud_from_depth_image(depth_m, camera, organized=True)

    valid_mask = depth_m > 0
    depth_range_mask = valid_mask & (depth_m > args.min_depth_m) & (depth_m < args.max_depth_m)
    input_mask, crop_bounds = compute_input_mask(
        depth_m,
        args.center_crop_ratio,
        args.min_depth_m,
        args.max_depth_m,
        args,
    )
    if args.largest_component_only:
        input_mask = keep_depth_component(
            input_mask,
            args.min_component_pixels,
            args.component_selection,
        )

    full_points = cloud[valid_mask]
    full_colors = color_rgb[valid_mask]
    depth_range_points = cloud[depth_range_mask]
    depth_range_colors = color_rgb[depth_range_mask]
    masked_points = cloud[input_mask]
    masked_colors = color_rgb[input_mask]

    save_depth_visualization(args.output_dir, depth_m)
    save_input_debug_figure(args.output_dir, color_rgb, depth_m, input_mask, crop_bounds)

    full_cloud_path = save_cloud(
        os.path.join(args.output_dir, "camera_cloud_full.ply"),
        full_points,
        full_colors,
        voxel_size=args.full_cloud_voxel,
    )
    depth_range_cloud_path = save_cloud(
        os.path.join(args.output_dir, "camera_cloud_depth_range.ply"),
        depth_range_points,
        depth_range_colors,
        voxel_size=args.masked_cloud_voxel,
    )
    masked_cloud_path = save_cloud(
        os.path.join(args.output_dir, "camera_cloud_masked.ply"),
        masked_points,
        masked_colors,
        voxel_size=args.masked_cloud_voxel,
    )

    summary = {
        "depth_summary": summarize_depth(depth_m, valid_mask),
        "camera_cloud_full": summarize_points(full_points),
        "camera_cloud_depth_range": summarize_points(depth_range_points),
        "camera_cloud_masked": summarize_points(masked_points),
        "files": {
            "color_png": os.path.join(args.output_dir, "color.png"),
            "depth_npy": os.path.join(args.output_dir, "depth_m.npy"),
            "depth_vis_png": os.path.join(args.output_dir, "depth_vis.png"),
            "input_debug_png": os.path.join(args.output_dir, "input_debug.png"),
            "camera_intrinsics_json": os.path.join(args.output_dir, "camera_intrinsics.json"),
            "camera_cloud_full_ply": full_cloud_path,
            "camera_cloud_depth_range_ply": depth_range_cloud_path,
            "camera_cloud_masked_ply": masked_cloud_path,
        },
        "mask": {
            "depth_range_pixels": int(np.count_nonzero(depth_range_mask)),
            "depth_range_ratio": float(np.count_nonzero(depth_range_mask) / depth_m.size),
            "selected_pixels": int(np.count_nonzero(input_mask)),
            "selected_ratio": float(np.count_nonzero(input_mask) / depth_m.size),
            "min_depth_m": float(args.min_depth_m),
            "max_depth_m": float(args.max_depth_m),
            "center_crop_ratio": float(args.center_crop_ratio),
            "roi": [
                float(args.roi_xmin),
                float(args.roi_xmax),
                float(args.roi_ymin),
                float(args.roi_ymax),
            ],
        },
    }

    if not args.skip_robot:
        base_T_ee = get_live_base_T_ee(args.can_name)
        handeye_transform = load_calibration_transform(args.calibration)
        directions = (
            ["ee_to_camera", "camera_to_ee"]
            if args.compare_directions
            else [args.calibration_direction]
        )
        summary["base_T_ee"] = base_T_ee.tolist()
        summary["handeye_calibration_file"] = args.calibration
        summary["base_clouds"] = {}

        for direction in directions:
            base_T_camera = build_base_T_camera(base_T_ee, handeye_transform, direction)
            base_points = transform_points(masked_points, base_T_camera)
            base_cloud_path = save_cloud(
                os.path.join(args.output_dir, f"base_cloud_{direction}.ply"),
                base_points,
                masked_colors,
                voxel_size=args.masked_cloud_voxel,
            )
            summary["base_clouds"][direction] = {
                "base_T_camera": base_T_camera.tolist(),
                "summary": summarize_points(base_points),
                "base_cloud_ply": base_cloud_path,
            }

    summary_path = os.path.join(args.output_dir, "diagnostic_summary.json")
    save_json(summary_path, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
