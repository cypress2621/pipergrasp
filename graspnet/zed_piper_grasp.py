"""Single-frame grasp pipeline for ZED + GraspNet + Piper.

This script captures one RGB-D frame from a ZED camera, runs GraspNet to
predict grasps, transforms the top candidates into the robot base frame using
eye-in-hand calibration plus the live Piper end-effector pose, and can
optionally execute a conservative top-down pick with Piper.
"""

import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from scipy import ndimage

import torch
from ultralytics import YOLO
from ultralytics.models.sam import Predictor as SAMPredictor
from graspnetAPI import GraspGroup

DEFAULT_YOLO_WEIGHT = "yolov8s-world.pt"
DEFAULT_SAM_WEIGHT = "sam_b.pt"

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, "models"))
sys.path.append(os.path.join(ROOT_DIR, "dataset"))
sys.path.append(os.path.join(ROOT_DIR, "utils"))

from graspnet import GraspNet, pred_decode
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image
from piper_sdk import C_PiperInterface


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", default="logs/log_rs/checkpoint-rs.tar")
    parser.add_argument("--calibration", default="2026-04-17_16-56-08_calibration.json")
    parser.add_argument(
        "--calibration_direction",
        choices=["ee_to_camera", "camera_to_ee"],
        default="ee_to_camera",
    )
    parser.add_argument("--compare_directions", action="store_true")
    parser.add_argument("--output_dir", default="runs/zed_piper_grasp")
    parser.add_argument("--resolution", choices=["HD720", "HD1080"], default="HD720")
    parser.add_argument(
        "--depth_mode",
        choices=["PERFORMANCE", "QUALITY", "ULTRA", "NEURAL", "NEURAL_PLUS"],
        default="NEURAL",
    )
    parser.add_argument("--num_point", type=int, default=20000)
    parser.add_argument("--num_view", type=int, default=300)
    parser.add_argument("--collision_thresh", type=float, default=0.01)
    parser.add_argument("--voxel_size", type=float, default=0.01)
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
    parser.add_argument("--mask_mode", choices=["depth", "yolo_sam", "hybrid"], default="hybrid")
    parser.add_argument("--target_class", default="bottle")
    parser.add_argument("--yolo_weight", default=DEFAULT_YOLO_WEIGHT)
    parser.add_argument("--sam_weight", default=DEFAULT_SAM_WEIGHT)
    parser.add_argument("--semantic_conf", type=float, default=0.25)
    parser.add_argument(
        "--semantic_box_selection",
        choices=["highest_conf", "center"],
        default="highest_conf",
    )
    parser.add_argument("--remove_plane", action="store_true")
    parser.add_argument("--plane_distance_thresh", type=float, default=0.01)
    parser.add_argument("--object_height_min", type=float, default=0.01)
    parser.add_argument("--object_height_max", type=float, default=0.20)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--min_score", type=float, default=0.05)
    parser.add_argument("--max_grasp_distance_m", type=float, default=1.0)
    parser.add_argument("--z_min", type=float, default=0.02)
    parser.add_argument("--z_max", type=float, default=0.80)
    parser.add_argument("--x_min", type=float, default=-0.50)
    parser.add_argument("--x_max", type=float, default=0.50)
    parser.add_argument("--y_min", type=float, default=-0.50)
    parser.add_argument("--y_max", type=float, default=0.50)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--can_name", default="can0")
    parser.add_argument("--gripper_length_m", type=float, default=0.07)
    parser.add_argument("--pregrasp_offset", type=float, default=0.10)
    parser.add_argument("--lift_offset", type=float, default=0.12)
    parser.add_argument("--travel_z_margin", type=float, default=0.03)
    parser.add_argument(
        "--execution_orientation_mode",
        choices=["current", "fixed"],
        default="current",
    )
    parser.add_argument("--tool_roll", type=float, default=math.pi)
    parser.add_argument("--tool_pitch", type=float, default=0.0)
    parser.add_argument("--tool_yaw", type=float, default=0.0)
    parser.add_argument("--gripper_open_m", type=float, default=0.09)
    parser.add_argument("--gripper_extra_close_m", type=float, default=0.005)
    parser.add_argument("--gripper_effort", type=int, default=1500)
    parser.add_argument("--speed_percent", type=int, default=30)
    parser.add_argument("--settle_time", type=float, default=2.0)
    return parser.parse_args()


def ensure_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "当前会话里没有可用 CUDA 设备，GraspNet baseline 的 pointnet2 算子无法在 CPU 上运行。"
        )


def get_net(args):
    ensure_cuda()
    net = GraspNet(
        input_feature_dim=0,
        num_view=args.num_view,
        num_angle=12,
        num_depth=4,
        cylinder_radius=0.05,
        hmin=-0.02,
        hmax_list=[0.01, 0.02, 0.03, 0.04],
        is_training=False,
    )
    device = torch.device("cuda:0")
    net.to(device)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()
    print("Loaded checkpoint:", args.checkpoint_path)
    return net, device

def move_to_init_pose(can_name):
    from piper_sdk import C_PiperInterface_V2
    import time

    print(f"正在连接 Piper ({can_name}) 并移动到初始观测位...")
    
    # 1. 初始化接口 (使用 V2 接口)
    piper = C_PiperInterface_V2(can_name)
    piper.ConnectPort()
    
    # 2. 使能机械臂
    while not piper.EnablePiper():
        time.sleep(0.01)
    
    # 3. 设置运动模式：关节控制模式，速度 30%
    # ModeCtrl 参数含义通常为：模式, 运动指令类型, 速度, 加速度
    piper.ModeCtrl(0x01, 0x01, 30, 0x00)
    
    # 4. 计算关节目标值
    # 单位转换因子：rad -> deg * 1000
    factor = 57295.7795 
    

    position = [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    joint_0 = round(position[0] * factor)
    joint_1 = round(position[1] * factor)
    joint_2 = round(position[2] * factor)
    joint_3 = round(position[3] * factor)
    joint_4 = round(position[4] * factor)
    joint_5 = round(position[5] * factor)
    # 夹爪通常使用微米单位
    joint_6 = round(position[6] * 1000 * 1000)
    
    # 5. 发送控制指令
    piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
    piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)
    
    # 6. 等待机械臂到达（根据位移估算，通常 2-3 秒足够）
    time.sleep(3.0)
    print("已到达初始观测位置。")
    
    # 注意：此处不需要关闭端口，否则后续采集 EndPose 可能会报错
    # 我们可以把这个 piper 对象返回，或者让后续代码复用
    return piper

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

    cv2.imwrite(os.path.join(output_dir, "color.png"), color_bgr)
    np.save(os.path.join(output_dir, "depth_m.npy"), depth_m)
    with open(os.path.join(output_dir, "camera_intrinsics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "width": int(depth_m.shape[1]),
                "height": int(depth_m.shape[0]),
                "fx": float(calib.fx),
                "fy": float(calib.fy),
                "cx": float(calib.cx),
                "cy": float(calib.cy),
                "scale": 1.0,
            },
            f,
            indent=2,
        )

    camera = CameraInfo(
        float(depth_m.shape[1]),
        float(depth_m.shape[0]),
        float(calib.fx),
        float(calib.fy),
        float(calib.cx),
        float(calib.cy),
        1.0,
    )
    return color_rgb, depth_m, camera


def choose_sam_predictor(sam_weight):
    overrides = dict(
        task='segment',
        mode='predict',
        model=sam_weight,
        conf=0.01,
        save=False,
    )
    return SAMPredictor(overrides=overrides)


def yolo_detect_objects(image_bgr, args, yolo_model):
    if yolo_model is None:
        return None
    if args.target_class:
        yolo_model.set_classes([args.target_class])

    results = yolo_model.predict(image_bgr, imgsz=640, conf=args.semantic_conf)
    if len(results) == 0:
        return None
    return results[0]


def select_yolo_mask(color_rgb, args, yolo_model=None, sam_predictor=None):
    if yolo_model is None:
        return None, None

    image_bgr = cv2.cvtColor(np.clip(color_rgb * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    result = yolo_detect_objects(image_bgr, args, yolo_model)
    if result is None:
        return None, None

    boxes = result.boxes.xyxy.cpu().numpy() if len(result.boxes) > 0 else np.empty((0, 4))
    class_ids = np.asarray(result.boxes.cls.cpu().numpy(), dtype=np.int32) if len(result.boxes) > 0 else np.empty((0,), dtype=np.int32)
    confs = np.asarray(result.boxes.conf.cpu().numpy(), dtype=np.float32) if len(result.boxes) > 0 else np.empty((0,), dtype=np.float32)
    names = result.names
    candidates = []

    for idx, box in enumerate(boxes):
        class_id = int(class_ids[idx]) if idx < len(class_ids) else -1
        label = names.get(class_id, str(class_id))
        if args.target_class and args.target_class != label and args.target_class != str(class_id):
            continue
        score = float(confs[idx]) if idx < len(confs) else 0.0
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        x0 = max(0, min(x0, image_rgb.shape[1] - 1))
        x1 = max(0, min(x1, image_rgb.shape[1] - 1))
        y0 = max(0, min(y0, image_rgb.shape[0] - 1))
        y1 = max(0, min(y1, image_rgb.shape[0] - 1))
        if x1 <= x0 or y1 <= y0:
            continue
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        candidates.append(
            {
                "box": (x0, y0, x1, y1),
                "score": score,
                "center_dist2": (cx - image_rgb.shape[1] / 2) ** 2 + (cy - image_rgb.shape[0] / 2) ** 2,
                "label": label,
                "class_id": class_id,
            }
        )

    if not candidates:
        return None, None

    if args.semantic_box_selection == "center":
        candidates.sort(key=lambda item: item["center_dist2"])
    else:
        candidates.sort(key=lambda item: item["score"], reverse=True)

    best = candidates[0]
    bbox = best["box"]

    if sam_predictor is not None:
        sam_predictor.set_image(image_rgb)
        results = sam_predictor(bboxes=[bbox])
        if results and results[0].masks:
            mask_obj = results[0].masks.data[0]
            mask = mask_obj.cpu().numpy().astype(bool)
            if mask.ndim != 2:
                mask = mask.squeeze()
            return mask, bbox

    mask = np.zeros((image_rgb.shape[0], image_rgb.shape[1]), dtype=bool)
    x0, y0, x1, y1 = bbox
    mask[y0:y1, x0:x1] = True
    return mask, bbox


def compute_input_mask(depth, center_crop_ratio, min_depth_m, max_depth_m, color_rgb, args, yolo_model=None, sam_predictor=None):
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

    if args.mask_mode in ("yolo_sam", "hybrid"):
        semantic_mask, semantic_box = select_yolo_mask(color_rgb, args, yolo_model, sam_predictor)
        if semantic_mask is not None:
            if args.mask_mode == "hybrid":
                mask &= semantic_mask
            else:
                mask = semantic_mask
            crop_bounds.append((semantic_box, (0, 255, 255)))
        else:
            print("YOLO/SAM 未检测到目标，继续使用深度掩码作为回退。")

    return mask, crop_bounds


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
    print("Saved input debug figure:", fig_path)


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


def build_end_points(
    color,
    depth,
    camera,
    num_point,
    device,
    min_depth_m,
    max_depth_m,
    center_crop_ratio,
    args,
    yolo_model=None,
    sam_predictor=None,
):
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
    mask, crop_bounds = compute_input_mask(
        depth,
        center_crop_ratio,
        min_depth_m,
        max_depth_m,
        color,
        args,
        yolo_model=yolo_model,
        sam_predictor=sam_predictor,
    )
    if args.largest_component_only:
        mask = keep_depth_component(mask, args.min_component_pixels, args.component_selection)
    save_input_debug_figure(args.output_dir, color, depth, mask, crop_bounds)

    cloud_masked = cloud[mask]
    color_masked = color[mask]

    if len(cloud_masked) == 0:
        raise RuntimeError("深度图中没有落在设定深度范围内的有效点云。")

    if args.remove_plane:
        plane_cloud = o3d.geometry.PointCloud()
        plane_cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
        plane_model, inliers = plane_cloud.segment_plane(
            distance_threshold=args.plane_distance_thresh,
            ransac_n=3,
            num_iterations=1000,
        )
        if len(inliers) > 0:
            inlier_mask = np.zeros(len(cloud_masked), dtype=bool)
            inlier_mask[np.asarray(inliers, dtype=np.int64)] = True
            outlier_mask = ~inlier_mask

            plane_points = cloud_masked[inlier_mask]
            object_points = cloud_masked[outlier_mask]
            object_colors = color_masked[outlier_mask]

            if len(object_points) > 0:
                plane_normal = np.asarray(plane_model[:3], dtype=np.float64)
                plane_normal = plane_normal / (np.linalg.norm(plane_normal) + 1e-8)
                plane_offset = float(plane_model[3])

                signed_dist = object_points @ plane_normal + plane_offset
                abs_dist = np.abs(signed_dist)
                height_mask = (
                    (abs_dist >= args.object_height_min)
                    & (abs_dist <= args.object_height_max)
                )
                if np.count_nonzero(height_mask) > 0:
                    cloud_masked = object_points[height_mask]
                    color_masked = object_colors[height_mask]

    if len(cloud_masked) == 0:
        raise RuntimeError("点云经过地面过滤后为空，请放宽过滤参数。")

    if len(cloud_masked) >= num_point:
        idxs = np.random.choice(len(cloud_masked), num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), num_point - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)

    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    vis_cloud = o3d.geometry.PointCloud()
    vis_cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    vis_cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))

    end_points = {
        "point_clouds": torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to(device),
        "cloud_colors": color_sampled,
    }
    return end_points, vis_cloud


def infer_grasps(net, end_points, vis_cloud, voxel_size, collision_thresh):
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg = GraspGroup(grasp_preds[0].detach().cpu().numpy())
    if collision_thresh > 0:
        detector = ModelFreeCollisionDetector(np.asarray(vis_cloud.points), voxel_size=voxel_size)
        collision_mask = detector.detect(gg, approach_dist=0.05, collision_thresh=collision_thresh)
        gg = gg[~collision_mask]
    gg.nms()
    gg.sort_by_score()
    return gg


def load_calibration_transform(calibration_path):
    with open(calibration_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    position = np.asarray(data["position"], dtype=np.float64)
    quaternion = np.asarray(data["orientation"], dtype=np.float64)
    rotation = R.from_quat(quaternion).as_matrix()

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = position
    return transform


def transform_grasp_to_base(grasp, base_T_camera):
    cam_R_grasp = grasp.rotation_matrix
    cam_t_grasp = grasp.translation

    base_R_camera = base_T_camera[:3, :3]
    base_t_camera = base_T_camera[:3, 3]

    base_R_grasp = base_R_camera @ cam_R_grasp
    base_t_grasp = base_R_camera @ cam_t_grasp + base_t_camera

    return base_R_grasp, base_t_grasp


def get_live_base_pose(can_name):
    piper = C_PiperInterface(can_name=can_name)
    piper.ConnectPort()
    time.sleep(1.0)
    pose = piper.GetArmEndPoseMsgs().end_pose

    xyz = np.array([pose.X_axis, pose.Y_axis, pose.Z_axis], dtype=np.float64) / 1e6
    rpy = np.deg2rad(np.array([pose.RX_axis, pose.RY_axis, pose.RZ_axis], dtype=np.float64) / 1000.0)

    base_T_ee = np.eye(4, dtype=np.float64)
    base_T_ee[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
    base_T_ee[:3, 3] = xyz
    return base_T_ee, xyz, rpy


def get_live_base_T_ee(can_name):
    base_T_ee, _, _ = get_live_base_pose(can_name)
    return base_T_ee


def build_base_T_camera(base_T_ee, handeye_transform, calibration_direction):
    if calibration_direction == "ee_to_camera":
        ee_T_camera = handeye_transform
    else:
        ee_T_camera = np.linalg.inv(handeye_transform)
    return base_T_ee @ ee_T_camera


def choose_candidates(gg, base_T_camera, args):
    candidates = []
    raw_debug = []
    for idx in range(len(gg)):
        grasp = gg[idx]
        base_R_grasp, base_t_grasp = transform_grasp_to_base(grasp, base_T_camera)
        x, y, z = base_t_grasp.tolist()
        euler_xyz = R.from_matrix(base_R_grasp).as_euler("xyz", degrees=False)
        width_limit = args.gripper_open_m
        effective_width = min(float(grasp.width), width_limit)
        width_was_clamped = bool(float(grasp.width) > width_limit)

        raw_item = {
            "index": idx,
            "score": float(grasp.score),
            "width_m": float(grasp.width),
            "effective_width_m": effective_width,
            "width_was_clamped": width_was_clamped,
            "depth_m": float(grasp.depth),
            "camera_distance_m": float(np.linalg.norm(grasp.translation)),
            "translation_base_m": [float(v) for v in base_t_grasp],
            "rotation_base_rpy_rad": [float(v) for v in euler_xyz],
            "width_limit_m": float(width_limit),
        }
        if len(raw_debug) < max(args.top_k, 20):
            raw_debug.append(raw_item)

        if grasp.score < args.min_score:
            continue
        if np.linalg.norm(grasp.translation) > args.max_grasp_distance_m:
            continue

        if not (args.x_min <= x <= args.x_max):
            continue
        if not (args.y_min <= y <= args.y_max):
            continue
        if not (args.z_min <= z <= args.z_max):
            continue

        candidates.append(
            {
                "index": idx,
                "score": float(grasp.score),
                "width_m": float(grasp.width),
                "effective_width_m": effective_width,
                "width_was_clamped": width_was_clamped,
                "depth_m": float(grasp.depth),
                "translation_base_m": [float(v) for v in base_t_grasp],
                "rotation_base_matrix": base_R_grasp.tolist(),
                "rotation_base_rpy_rad": [float(v) for v in euler_xyz],
            }
        )
        if len(candidates) >= args.top_k:
            break
    return candidates, raw_debug


def save_json(output_dir, filename, payload):
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return output_path


def run_candidate_selection(gg, base_T_camera, args, output_dir, prefix):
    candidates, raw_debug = choose_candidates(gg, base_T_camera, args)
    raw_path = save_json(output_dir, f"{prefix}_grasp_candidates_raw.json", raw_debug)
    print("Saved raw candidates:", raw_path)
    if not candidates:
        print(f"No valid candidates for {prefix}.")
        return None, raw_debug

    output_path = save_json(output_dir, f"{prefix}_grasp_candidates.json", candidates)
    print("Saved candidates:", output_path)
    print(f"Best candidate ({prefix}):", json.dumps(candidates[0], indent=2))
    return candidates, raw_debug


def visualize_candidate(vis_cloud, gg, candidate, base_T_camera, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    grasp = gg[candidate["index"]]
    gripper = grasp.to_open3d_geometry()

    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
    base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.10)
    base_frame.transform(base_T_camera)

    world_T_grasp = np.eye(4, dtype=np.float64)
    world_T_grasp[:3, :3] = grasp.rotation_matrix
    world_T_grasp[:3, 3] = grasp.translation
    grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.06)
    grasp_frame.transform(world_T_grasp)

    vis_path = os.path.join(output_dir, "preview_cloud.ply")
    o3d.io.write_point_cloud(vis_path, vis_cloud)
    print("Saved preview cloud:", vis_path)
    print("Opening Open3D preview window...")
    o3d.visualization.draw_geometries([vis_cloud, gripper, camera_frame, base_frame, grasp_frame])


def pose_to_piper_units(position_m, rpy_rad):
    factor = 180.0 / math.pi
    x = int(round(position_m[0] * 1_000_000))
    y = int(round(position_m[1] * 1_000_000))
    z = int(round(position_m[2] * 1_000_000))
    rx = int(round(rpy_rad[0] * factor * 1000))
    ry = int(round(rpy_rad[1] * factor * 1000))
    rz = int(round(rpy_rad[2] * factor * 1000))
    return x, y, z, rx, ry, rz


def gripper_m_to_units(width_m):
    width_m = max(0.0, width_m)
    return int(round(width_m * 1_000_000))


def move_end_pose(piper, position_m, rpy_rad, speed_percent, settle_time):
    piper.MotionCtrl_2(0x01, 0x00, speed_percent)
    piper.EndPoseCtrl(*pose_to_piper_units(position_m, rpy_rad))
    time.sleep(settle_time)


def execute_topdown_pick(candidate, args):
    piper = C_PiperInterface(can_name=args.can_name)
    piper.ConnectPort()
    time.sleep(0.2)
    piper.EnablePiper()
    piper.EnableArm(7)
    time.sleep(1.0)

    # 初始打开夹爪
    piper.GripperCtrl(gripper_m_to_units(args.gripper_open_m), args.gripper_effort, 0x01, 0)
    time.sleep(1.0)

    _, current_xyz, current_rpy = get_live_base_pose(args.can_name)
    grasp_center = np.asarray(candidate["translation_base_m"], dtype=np.float64)
    base_R_grasp = np.asarray(candidate["rotation_base_matrix"], dtype=np.float64)
    approach_axis = base_R_grasp[:, 0]
    approach_axis = approach_axis / (np.linalg.norm(approach_axis) + 1e-8)

    # --- 核心修改部分：选择姿态 ---
    if args.execution_orientation_mode == "current":
        tool_rpy = current_rpy.copy()
    elif args.execution_orientation_mode == "fixed":
        tool_rpy = np.array([args.tool_roll, args.tool_pitch, args.tool_yaw], dtype=np.float64)
    # ---------------------------

    tcp_target = grasp_center - approach_axis * args.gripper_length_m
    tcp_pregrasp = tcp_target - approach_axis * args.pregrasp_offset
    lift = tcp_target.copy()
    lift[2] += args.lift_offset

    safe_travel_z = max(
        float(current_xyz[2]),
        float(tcp_pregrasp[2]),
        float(tcp_target[2]),
        float(lift[2]),
    ) + args.travel_z_margin

    current_lift = current_xyz.copy()
    current_lift[2] = safe_travel_z
    approach_xy = tcp_pregrasp.copy()
    approach_xy[2] = safe_travel_z
    tcp_pregrasp[2] = max(tcp_pregrasp[2], tcp_target[2])
    lift[2] = max(lift[2], safe_travel_z)

    print("Execute candidate:", json.dumps(candidate, indent=2))
    print(
        "Execution pose:",
        json.dumps(
            {
                "execution_orientation_mode": args.execution_orientation_mode,
                "tool_rpy_rad": [float(v) for v in tool_rpy],
                "current_xyz_m": [float(v) for v in current_xyz],
                "safe_travel_z_m": float(safe_travel_z),
                "approach_xy_m": [float(v) for v in approach_xy],
                "tcp_pregrasp_m": [float(v) for v in tcp_pregrasp],
                "tcp_target_m": [float(v) for v in tcp_target],
                "lift_m": [float(v) for v in lift],
            },
            indent=2,
        ),
    )
    print(f"DEBUG: 目标 RPY (度): {np.rad2deg(tool_rpy)}")
    j_msgs = piper.GetArmJointMsgs()
    print(f"当前关节位置: {j_msgs}")
    # 依次执行路径点
    move_end_pose(piper, current_lift, tool_rpy, args.speed_percent, args.settle_time)
    move_end_pose(piper, approach_xy, tool_rpy, args.speed_percent, args.settle_time)
    move_end_pose(piper, tcp_pregrasp, tool_rpy, args.speed_percent, args.settle_time)
    move_end_pose(piper, tcp_target, tool_rpy, args.speed_percent, args.settle_time)

    # 关闭夹爪进行抓取
    target_width = float(candidate.get("effective_width_m", candidate["width_m"]))
    close_width = max(0.0, target_width - args.gripper_extra_close_m)
    piper.GripperCtrl(gripper_m_to_units(close_width), args.gripper_effort, 0x01, 0)
    time.sleep(1.2)

    # 提起物体
    move_end_pose(piper, lift, tool_rpy, args.speed_percent, args.settle_time)

    # 放下物体并松开
    move_end_pose(piper, tcp_target, tool_rpy, args.speed_percent, args.settle_time)
    piper.GripperCtrl(gripper_m_to_units(args.gripper_open_m), args.gripper_effort, 0x01, 0)
    time.sleep(1.0)
    move_end_pose(piper, current_lift, tool_rpy, args.speed_percent, args.settle_time)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    try:
        # 为了不和脚本后面原有的 piper 逻辑冲突，你可以选择在这里移动完就结束
        move_to_init_pose(args.can_name)
    except Exception as e:
        print(f"无法移动到初始位置: {e}")
    net, device = get_net(args)
    time.sleep(args.settle_time)
    color, depth, camera = capture_zed_frame(args.output_dir, args.resolution, args.depth_mode)
    yolo_model = None
    sam_predictor = None
    if args.mask_mode in ("yolo_sam", "hybrid"):
        print(f"正在加载 YOLO 模型: {args.yolo_weight}")
        try:
            yolo_model = YOLO(args.yolo_weight)
        except Exception as e:
            raise RuntimeError(
                "无法加载 YOLO 权重，请先安装 ultralytics 并确保权重路径正确。"
            ) from e
        print(f"正在加载 SAM 模型: {args.sam_weight}")
        try:
            sam_predictor = choose_sam_predictor(args.sam_weight)
        except Exception as e:
            raise RuntimeError(
                "无法加载 SAM 权重，请先安装 ultralytics 并确保权重路径正确。"
            ) from e

    end_points, vis_cloud = build_end_points(
        color,
        depth,
        camera,
        args.num_point,
        device,
        args.min_depth_m,
        args.max_depth_m,
        args.center_crop_ratio,
        args,
        yolo_model=yolo_model,
        sam_predictor=sam_predictor,
    )
    gg = infer_grasps(net, end_points, vis_cloud, args.voxel_size, args.collision_thresh)

    base_T_ee = get_live_base_T_ee(args.can_name)
    handeye_transform = load_calibration_transform(args.calibration)
    base_T_camera = build_base_T_camera(
        base_T_ee,
        handeye_transform,
        args.calibration_direction,
    )
    if args.compare_directions:
        compare_results = {}
        for direction in ["ee_to_camera", "camera_to_ee"]:
            compare_base_T_camera = build_base_T_camera(base_T_ee, handeye_transform, direction)
            candidates, raw_debug = run_candidate_selection(
                gg,
                compare_base_T_camera,
                args,
                args.output_dir,
                direction,
            )
            compare_results[direction] = {
                "has_candidates": candidates is not None,
                "best_candidate": candidates[0] if candidates else None,
            }
            if args.visualize and candidates:
                visualize_candidate(
                    vis_cloud,
                    gg,
                    candidates[0],
                    compare_base_T_camera,
                    os.path.join(args.output_dir, direction),
                )

        compare_path = save_json(args.output_dir, "direction_compare_summary.json", compare_results)
        print("Saved direction comparison:", compare_path)
        return

    candidates, raw_debug = run_candidate_selection(
        gg,
        base_T_camera,
        args,
        args.output_dir,
        "selected",
    )
    if not candidates:
        raise RuntimeError("没有筛选出满足工作空间约束的抓取候选。")

    # Keep legacy filenames for downstream inspection.
    save_json(args.output_dir, "grasp_candidates_raw.json", raw_debug)
    save_json(args.output_dir, "grasp_candidates.json", candidates)

    if args.visualize:
        visualize_candidate(vis_cloud, gg, candidates[0], base_T_camera, args.output_dir)

    if args.execute:
        execute_topdown_pick(candidates[0], args)
        print("Piper execution complete.")


if __name__ == "__main__":
    main()
