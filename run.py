from pathlib import Path
import torch
import argparse
import os
import cv2
import math
import numpy as np
import pickle
import time
from tqdm import tqdm

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer
from hamer.utils import recursive_to
from hamer.utils.geometry import perspective_projection
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer
from typing import List, Tuple


openpose_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
gt_indices = openpose_indices

print("!!!!!!!!!!!!!!!! third-party/hamer/run.py (simplified, pkl-driven) !!!!!!!!!!!!!!!!!!!")


HAND_SKELETON: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)

RIGHT_COLOR  = (50, 180, 255)
RIGHT_COLOR_ = (0, 0, 200)
LEFT_COLOR  = (255, 90, 120)
LEFT_COLOR_ = (204, 0, 0)

BBOX_LEFT_COLOR  = (0, 0, 255)   # 红色 (BGR)
BBOX_RIGHT_COLOR = (255, 0, 0)   # 蓝色 (BGR)


def draw_bbox_and_label(img, bbox, label, color):
    """
    bbox: [x1, y1, x2, y2]
    label: 'left' or 'right'
    """
    x1, y1, x2, y2 = map(int, bbox[:4])
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    cv2.putText(
        img, label, (x1, y1 - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3
    )
    return img



def load_hand_track_file(path):
    """
    Auto-detect pkl or json, and convert to unified:
        [
            {
                "frame": int,
                "bbox": [x1,y1,x2,y2],   <-- XYXY format
                "lr": "left"/"right"
            }
        ]
    """
    suffix = os.path.splitext(path)[1].lower()
    hand_bbox_all = []

    # ---------------------------
    # Case 1: PKL
    # ---------------------------
    if suffix == ".pkl":
        with open(path, "rb") as f:
            hand_bbox_all = pickle.load(f)

        print(f"[INFO] Loaded PKL: {len(hand_bbox_all)} boxes")
        return hand_bbox_all

    # ---------------------------
    # Case 2: JSON
    # ---------------------------
    elif suffix == ".json":
        import json
        with open(path, "r") as f:
            data = json.load(f)

        for frame_key, hands in data.items():
            frame = int(frame_key)

            if "left_hand" in hands:
                x1, y1, x2, y2 = hands["left_hand"]["box_2d"]
                hand_bbox_all.append({
                    "frame": frame,
                    "bbox": [x1, y1, x2, y2],
                    "lr": "left",
                })

            if "right_hand" in hands:
                x1, y1, x2, y2 = hands["right_hand"]["box_2d"]
                hand_bbox_all.append({
                    "frame": frame,
                    "bbox": [x1, y1, x2, y2],
                    "lr": "right",
                })

        print(f"[INFO] Loaded JSON: {len(hand_bbox_all)} boxes")
        return hand_bbox_all

    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def _draw_hand(
    image: np.ndarray,
    keypoints: np.ndarray,
    valid: np.ndarray,
    color: Tuple[int, int, int],
    circle_radius: int,
    line_thickness: int,
) -> None:
    pts = keypoints.astype(np.int32)
    for i, j in HAND_SKELETON:
        if valid[i] and valid[j]:
            cv2.line(image, tuple(pts[i]), tuple(pts[j]), color, thickness=line_thickness, lineType=cv2.LINE_AA)
    for idx, is_valid in enumerate(valid):
        if is_valid:
            color_ = RIGHT_COLOR_ if color[0] == RIGHT_COLOR[0] else LEFT_COLOR_
            cv2.circle(image, tuple(pts[idx]), circle_radius, color_, thickness=-1, lineType=cv2.LINE_AA)



def get_keypoints_rectangle(keypoints: np.array, threshold: float) -> Tuple[float, float, float]:
    """
    Compute rectangle enclosing keypoints above the threshold.
    Args:
        keypoints (np.array): Keypoint array of shape (N, 3).
        threshold (float): Confidence visualization threshold.
    Returns:
        Tuple[float, float, float]: Rectangle width, height and area.
    """
    print(keypoints.shape)
    valid_ind = keypoints[:, -1] > threshold
    if valid_ind.sum() > 0:
        valid_keypoints = keypoints[valid_ind][:, :-1]
        max_x = valid_keypoints[:,0].max()
        max_y = valid_keypoints[:,1].max()
        min_x = valid_keypoints[:,0].min()
        min_y = valid_keypoints[:,1].min()
        width = max_x - min_x
        height = max_y - min_y
        area = width * height
        return width, height, area
    else:
        return 0,0,0

def render_keypoints(img: np.array,
                     keypoints: np.array,
                     pairs: List,
                     colors: List,
                     thickness_circle_ratio: float,
                     thickness_line_ratio_wrt_circle: float,
                     pose_scales: List,
                     threshold: float = 0.1,
                     alpha: float = 1.0) -> np.array:
    """
    Render keypoints on input image.
    Args:
        img (np.array): Input image of shape (H, W, 3) with pixel values in the [0,255] range.
        keypoints (np.array): Keypoint array of shape (N, 3).
        pairs (List): List of keypoint pairs per limb.
        colors: (List): List of colors per keypoint.
        thickness_circle_ratio (float): Circle thickness ratio.
        thickness_line_ratio_wrt_circle (float): Line thickness ratio wrt the circle.
        pose_scales (List): List of pose scales.
        threshold (float): Only visualize keypoints with confidence above the threshold.
    Returns:
        (np.array): Image of shape (H, W, 3) with keypoints drawn on top of the original image. 
    """
    img_orig = img.copy()
    width, height = img.shape[1], img.shape[2]
    area = width * height

    lineType = 8
    shift = 0
    numberColors = len(colors)
    thresholdRectangle = 0.1

    person_width, person_height, person_area = get_keypoints_rectangle(keypoints, thresholdRectangle)
    if person_area > 0:
        ratioAreas = min(1, max(person_width / width, person_height / height))
        thicknessRatio = np.maximum(np.round(math.sqrt(area) * thickness_circle_ratio * ratioAreas), 2)
        thicknessCircle = np.maximum(1, thicknessRatio if ratioAreas > 0.05 else -np.ones_like(thicknessRatio))
        thicknessLine = np.maximum(1, np.round(thicknessRatio * thickness_line_ratio_wrt_circle))
        radius = thicknessRatio / 2

        img = np.ascontiguousarray(img.copy())
        for i, pair in enumerate(pairs):
            index1, index2 = pair
            if keypoints[index1, -1] > threshold and keypoints[index2, -1] > threshold:
                thicknessLineScaled = int(round(min(thicknessLine[index1], thicknessLine[index2]) * pose_scales[0]))
                colorIndex = index2
                color = colors[colorIndex % numberColors]
                keypoint1 = keypoints[index1, :-1].astype(np.int32)
                keypoint2 = keypoints[index2, :-1].astype(np.int32)
                cv2.line(img, tuple(keypoint1.tolist()), tuple(keypoint2.tolist()), tuple(color.tolist()), thicknessLineScaled, lineType, shift)
        for part in range(len(keypoints)):
            faceIndex = part
            if keypoints[faceIndex, -1] > threshold:
                radiusScaled = int(round(radius[faceIndex] * pose_scales[0]))
                thicknessCircleScaled = int(round(thicknessCircle[faceIndex] * pose_scales[0]))
                colorIndex = part
                color = colors[colorIndex % numberColors]
                center = keypoints[faceIndex, :-1].astype(np.int32)
                cv2.circle(img, tuple(center.tolist()), radiusScaled, tuple(color.tolist()), thicknessCircleScaled, lineType, shift)
    return img

def render_hand_keypoints(img, right_hand_keypoints, threshold=0.1, use_confidence=False, map_fn=lambda x: np.ones_like(x), alpha=1.0):
    if use_confidence and map_fn is not None:
        #thicknessCircleRatioLeft = 1./50 * map_fn(left_hand_keypoints[:, -1])
        thicknessCircleRatioRight = 1./50 * map_fn(right_hand_keypoints[:, -1])
    else:
        #thicknessCircleRatioLeft = 1./50 * np.ones(left_hand_keypoints.shape[0])
        thicknessCircleRatioRight = 1./50 * np.ones(right_hand_keypoints.shape[0])
    thicknessLineRatioWRTCircle = 0.75
    pairs = [0,1,  1,2,  2,3,  3,4,  0,5,  5,6,  6,7,  7,8,  0,9,  9,10,  10,11,  11,12,  0,13,  13,14,  14,15,  15,16,  0,17,  17,18,  18,19,  19,20]
    pairs = np.array(pairs).reshape(-1,2)

    colors = [100.,  100.,  100.,
              100.,    0.,    0.,
              150.,    0.,    0.,
              200.,    0.,    0.,
              255.,    0.,    0.,
              100.,  100.,    0.,
              150.,  150.,    0.,
              200.,  200.,    0.,
              255.,  255.,    0.,
                0.,  100.,   50.,
                0.,  150.,   75.,
                0.,  200.,  100.,
                0.,  255.,  125.,
                0.,   50.,  100.,
                0.,   75.,  150.,
                0.,  100.,  200.,
                0.,  125.,  255.,
              100.,    0.,  100.,
              150.,    0.,  150.,
              200.,    0.,  200.,
              255.,    0.,  255.]
    colors = np.array(colors).reshape(-1,3)
    #colors = np.zeros_like(colors)
    poseScales = [1]
    img = render_keypoints(img, right_hand_keypoints, pairs, colors, thicknessCircleRatioRight, thicknessLineRatioWRTCircle, poseScales, threshold, alpha=alpha)
    return img


def render_openpose(img: np.array,
                    hand_keypoints: np.array) -> np.array:
    """
    Render keypoints in the OpenPose format on input image.
    Args:
        img (np.array): Input image of shape (H, W, 3) with pixel values in the [0,255] range.
        body_keypoints (np.array): Keypoint array of shape (N, 3); 3 <====> (x, y, confidence).
    Returns:
        (np.array): Image of shape (H, W, 3) with keypoints drawn on top of the original image. 
    """
    #img = render_body_keypoints(img, body_keypoints)
    img = render_hand_keypoints(img, hand_keypoints)
    return img


def cam_crop_to_full(cam_bbox, box_center, box_size, img_size, focal_length=5000.0):
    """Convert cropped camera parameters back to full-image camera coordinates."""
    img_w, img_h = img_size[:, 0], img_size[:, 1]
    cx, cy, b = box_center[:, 0], box_center[:, 1], box_size
    w_2, h_2 = img_w / 2.0, img_h / 2.0
    bs = b * cam_bbox[:, 0] + 1e-9
    tz = 2 * focal_length / bs
    tx = (2 * (cx - w_2) / bs) + cam_bbox[:, 1]
    ty = (2 * (cy - h_2) / bs) + cam_bbox[:, 2]
    full_cam = torch.stack([tx, ty, tz], dim=-1)
    return full_cam


LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)


def convert_crop_coords_to_orig_img(bbox, keypoints, crop_size):
    """
    Convert keypoints from crop-normalized coordinates back to original image coordinates.

    bbox: (N, 4) [cx, cy, h, w]
    keypoints: (N, K, 3), coordinates in crop space [0, crop_size]
    """
    cx, cy, h = bbox[:, 0], bbox[:, 1], bbox[:, 2]

    # Rescale to original crop size
    keypoints *= h[..., None, None] / crop_size

    # Shift to original full-image coordinates
    keypoints[:, :, 0] = (cx - h / 2)[..., None] + keypoints[:, :, 0]
    keypoints[:, :, 1] = (cy - h / 2)[..., None] + keypoints[:, :, 1]
    return keypoints


def save_video(path, out_dir, out_name):
    """Save all images in `path` as an mp4 video."""
    print("saving to :", out_name + ".mp4")
    img_array = []
    height, width = 0, 0

    for filename in tqdm(sorted(os.listdir(path))):
        img = cv2.imread(os.path.join(path, filename))
        if img is None:
            continue
        if height != 0:
            img = cv2.resize(img, (width, height))
        height, width, _ = img.shape
        size = (width, height)
        img_array.append(img)

    if len(img_array) == 0:
        print("[WARN] No frames found for video:", path)
        return

    out_path = os.path.join(out_dir, out_name + ".mp4")
    out = cv2.VideoWriter(out_path, 0x7634706D, 30, size)
    for frame in img_array:
        out.write(frame)
    out.release()
    print("done:", out_path)


def main():
    parser = argparse.ArgumentParser(description="HaMeR demo code (pkl-driven, no detector)")
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to pretrained model checkpoint')
    parser.add_argument('--img_folder', type=str, default='images', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='out_demo', help='Output folder to save rendered results')
    parser.add_argument('--res_folder', type=str, help='Output folder to save rendered results')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=True, help='If set, render all people together also')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'], help='List of file extensions to consider')
    parser.add_argument('--conf', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--type', type=str, default='EgoDexter', help='Path to pretrained model checkpoint')
    parser.add_argument('--render', dest='render', action='store_true', default=False, help='If set, render side view also')
    parser.add_argument(
        "--hands_track_pkl",
        type=str,
        required=True,
        help="Path to hands_track.pkl (list of dicts with 'frame', 'bbox', 'lr')",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    print("args.checkpoint:", args.checkpoint)
    download_models(args.checkpoint)
    model, model_cfg = load_hamer(args.checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # ------------------------------------------------------------------
    # 2. Setup renderer and output dirs
    # ------------------------------------------------------------------
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    args.out_folder = args.out_folder + "_" + str(model_cfg.EXTRA.FOCAL_LENGTH)

    # Base directory for saving results
    if args.res_folder is not None:
        base_dir = os.path.dirname(args.res_folder)
        if base_dir == "":
            base_dir = "."
    else:
        base_dir = args.out_folder

    render_save_path = os.path.join(base_dir, f"render_all_{model_cfg.EXTRA.FOCAL_LENGTH}")
    joint2d_save_path = os.path.join(base_dir, f"joint2d_{model_cfg.EXTRA.FOCAL_LENGTH}")
    vit_save_path = os.path.join(base_dir, f"vit_{model_cfg.EXTRA.FOCAL_LENGTH}")
    mesh_dir = os.path.join(base_dir, f"mesh_{model_cfg.EXTRA.FOCAL_LENGTH}")
    keypoint_json_save_path = os.path.join(base_dir, f"keypoint_json_{model_cfg.EXTRA.FOCAL_LENGTH}")

    os.makedirs(render_save_path, exist_ok=True)
    os.makedirs(joint2d_save_path, exist_ok=True)
    os.makedirs(vit_save_path, exist_ok=True)
    os.makedirs(mesh_dir, exist_ok=True)
    os.makedirs(keypoint_json_save_path, exist_ok=True)
    os.makedirs(base_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 3. Collect image paths
    # ------------------------------------------------------------------
    img_paths = [img for end in args.file_type for img in Path(args.img_folder).glob(end)]
    img_paths = sorted(img_paths)
    print("Total frames:", len(img_paths))

    # ------------------------------------------------------------------
    # 4. Load hand bbox tracking results from pkl
    # ------------------------------------------------------------------
    # print("Loading hands_track_pkl from:", args.hands_track_pkl)
    # with open(args.hands_track_pkl, "rb") as f:
    #     hand_bbox_all = pickle.load(f)
    hand_bbox_all = load_hand_track_file(args.hands_track_pkl)
    print("Total hand bbox entries:", len(hand_bbox_all))

    # Build frame -> list of hands mapping for fast lookup
    frame2hands = {}
    for h in hand_bbox_all:
        frame_id = int(h["frame"])
        frame2hands.setdefault(frame_id, []).append(h)

    # ------------------------------------------------------------------
    # 5. Run inference over all frames
    # ------------------------------------------------------------------
    results_dict = {}
    big_all_verts = []
    big_all_cam_t = []
    big_all_joints = []
    big_all_right = []

    tid = []
    tracked_time = [0, 0]  # index 0: left, 1: right

    start_time_all = time.time()

    for idx, img_path in enumerate(tqdm(img_paths)):
        a = time.time()
        img_path = str(img_path)
        img_cv2 = cv2.imread(img_path)
        if img_cv2 is None:
            print("[WARN] Failed to read image:", img_path)
            continue

        print("Processing frame", idx, ":", img_path)

        results_dict[img_path] = {
            "mano": [],
            "cam_trans": [],
            "tracked_ids": [],
            "tracked_time": [],
            "extra_data": [],
        }

        # ---------------------------
        # get hands from pkl for this frame
        # ---------------------------
        hands = frame2hands.get(idx+1500, [])
        bboxes = []
        is_right_list = []
        vit_keypoints_list = []

        for hand in hands:
            # hand['bbox'] assumed [cx, cy, h, w] (consistent with your original code)
            bboxes.append(hand["bbox"][:4])
            lr = hand["lr"]
            is_right_list.append(1 if lr == "right" else 0)
            # No ViTPose now: provide dummy 21 keypoints with zero confidence
            vit_keypoints_list.append(np.zeros((21, 3), dtype=np.float32))

        if len(bboxes) == 0:
            # No hand detected/tracked for this frame
            results_dict[img_path]["tid"] = tid
            results_dict[img_path]["tracked_time"] = []
            results_dict[img_path]["shot"] = 0
            tracked_time[0] += 1
            tracked_time[1] += 1
            for i in tid:
                results_dict[img_path]["tracked_time"].append(tracked_time[i])
            for idx_t, t in enumerate(tracked_time):
                if t > 50 and (idx_t in tid):
                    tid.remove(idx_t)
            print("No hand in pkl for frame", idx, "| tid:", results_dict[img_path]["tid"])
            continue

        # numpy arrays
        boxes = np.stack(bboxes)  # (N, 4)
        right = np.stack(is_right_list)  # (N,)
        vit_keypoints = np.stack(vit_keypoints_list)  # (N, 21, 3)

        # ensure left(0) before right(1), keep at most 2 hands
        sort_idx = np.argsort(right)
        boxes = boxes[sort_idx][:2]
        right = right[sort_idx][:2]
        vit_keypoints = vit_keypoints[sort_idx][:2]

        print("Hands used in frame:", right)

        # ------------------------------------------------------------------
        # Build dataset and dataloader for this frame
        # ------------------------------------------------------------------
        dataset = ViTDetDataset(
            model_cfg,
            img_cv2,
            boxes,
            right,
            vit_keypoints,
            rescale_factor=args.rescale_factor,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )

        all_verts = []
        all_cam_t = []
        all_joints = []
        all_pred_3d = []
        all_right = []
        all_pred_2d = []
        all_bboxes = []

        left_flag = False
        right_flag = False

        b = time.time()
        print("preprocess time:", b - a)

        # To be reused outside loop
        scaled_focal_length = None
        img_size_current = None

        for batch in dataloader:
            batch = recursive_to(batch, device)

            with torch.no_grad():
                out = model(batch)

            multiplier = (2 * batch["right"] - 1)
            pred_cam = out["pred_cam"]
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]

            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            multiplier = (2 * batch["right"] - 1)
            # scaled_focal_length = (
            #     model_cfg.EXTRA.FOCAL_LENGTH
            #     / model_cfg.MODEL.IMAGE_SIZE
            #     * img_size.max()
            # )

            scaled_focal_length = 716
            pred_cam_t_full = cam_crop_to_full(
                pred_cam,
                box_center,
                box_size,
                img_size,
                scaled_focal_length,
            )

            pred_cam_t_full = pred_cam_t_full.detach().cpu().numpy()
            batch_size = batch["img"].shape[0]

            for n in range(batch_size):
                img_fn, _ = os.path.splitext(os.path.basename(img_path))
                person_id = int(batch["personid"][n])

                verts = out["pred_vertices"][n].detach().cpu().numpy()
                pred_joints_2d = out["pred_keypoints_2d"][n].detach().cpu().numpy()
                pred_3d = out["pred_keypoints_3d"][n].detach().cpu().numpy()

                is_right = int(batch["right"][n].cpu().numpy())

                # Some datasets only use one hand side
                if args.type == "EgoPAT3D" and is_right == 0:
                    print("skip left hand for EgoPAT3D")
                    continue
                if args.type == "EgoDexter" and is_right == 1:
                    print("skip right hand for EgoDexter")
                    continue

                # ensure at most one left & one right per frame
                if is_right == 1:
                    if right_flag:
                        continue
                    right_flag = True
                else:
                    if left_flag:
                        continue
                    left_flag = True

                # Mirror coordinates for left/right consistency
                pred_joints_2d[:, 0] = (2 * is_right - 1) * pred_joints_2d[:, 0]
                verts[:, 0] = (2 * is_right - 1) * verts[:, 0]
                pred_3d[:, 0] = (2 * is_right - 1) * pred_3d[:, 0]

                # add dummy confidence for 2D joints
                v_conf = np.ones((pred_joints_2d.shape[0], 1), dtype=np.float32)
                pred_joints_2d_conf = np.concatenate((pred_joints_2d, v_conf), axis=-1)

                cam_t = pred_cam_t_full[n]
                img_size_current = img_size[n].detach().cpu().numpy()

                all_pred_3d.append(pred_3d)
                all_joints.append(pred_joints_2d_conf)
                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right)
                all_pred_2d.append(pred_joints_2d_conf)
                all_bboxes.append(batch["bbox"][n].detach().cpu().numpy())

                # tracking info
                tracked_time[is_right] = 0
                results_dict[img_path]["tracked_ids"].append(is_right)
                if is_right not in tid:
                    tid.append(is_right)

                out["pred_mano_params"][n]["is_right"] = is_right
                results_dict[img_path]["mano"].append(out["pred_mano_params"][n])
                results_dict[img_path]["cam_trans"].append(cam_t)

        # accumulate for possible mesh exporting
        big_all_joints.append(all_pred_3d)
        big_all_verts.append(all_verts)
        big_all_cam_t.append(all_cam_t)
        big_all_right.append(all_right)

        # update tracking time for missing hand side
        assert len(results_dict[img_path]["tracked_ids"]) <= 2
        if len(results_dict[img_path]["tracked_ids"]) == 1:
            if results_dict[img_path]["tracked_ids"][0] == 0:
                # only left present -> right missing
                tracked_time[1] += 1
            else:
                # only right present -> left missing
                tracked_time[0] += 1

        tid = sorted(tid)
        for idx_t, t in enumerate(tracked_time):
            if t > 50 and (idx_t in tid):
                tid.remove(idx_t)

        results_dict[img_path]["shot"] = 0
        results_dict[img_path]["tracked_time"] = []
        for i in tid:
            results_dict[img_path]["tracked_time"].append(tracked_time[i])

        results_dict[img_path]["tid"] = np.array(tid)
        print(
            "tid/tracked_ids/tracked_time",
            results_dict[img_path]["tid"],
            results_dict[img_path]["tracked_ids"],
            results_dict[img_path]["tracked_time"],
        )

        # ------------------------------------------------------------------
        # Full-frame rendering (overlay mesh on original image)
        # ------------------------------------------------------------------
        if args.full_frame and len(all_verts) > 0:
            print("render full-frame view")

            all_pred_2d_np = np.stack(all_pred_2d)  # (M, 21, 3)
            all_bboxes_np = np.stack(all_bboxes)  # (M, 4)

            # convert to original image coordinates
            all_pred_2d_img = model_cfg.MODEL.IMAGE_SIZE * (all_pred_2d_np + 0.5)
            all_pred_2d_img = convert_crop_coords_to_orig_img(
                bbox=all_bboxes_np,
                keypoints=all_pred_2d_img,
                crop_size=model_cfg.MODEL.IMAGE_SIZE,
            )
            all_pred_2d_img[:, :, -1] = 1
            results_dict[img_path]["extra_data"] = [
                kp.tolist() for kp in all_pred_2d_img
            ]

            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )

            cam_view, multi_depth = renderer.render_rgba_multiple(
                all_verts,
                cam_t=all_cam_t,
                render_res=img_size_current,
                is_right=all_right,
                **misc_args,
            )

            # Overlay mesh on original image
            input_img = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
            input_img_overlay = (
                input_img[:, :, :3] * (1 - cam_view[:, :, 3:])
                + cam_view[:, :, :3] * cam_view[:, :, 3:]
            )

            img_fn, _ = os.path.splitext(os.path.basename(img_path))
            overlay_bgr = (255 * input_img_overlay[:, :, ::-1]).astype(np.uint8)



            # Overlay 2d keypoints on original image
            # pred_img = input_img.copy()[:,:,:-1][:,:,::-1] * 255
            # for i in range(len(all_verts)):
            #     body_keypoints_2d = all_pred_2d_img[i, :21].copy()
            #     for op, gt in zip(openpose_indices, gt_indices):
            #         if all_pred_2d_img[i, gt, -1] > body_keypoints_2d[op, -1]:
            #             body_keypoints_2d[op] = all_pred_2d_img[i, gt]
            #             raise ValueError
            #         else:
            #             pass

            #     assert (body_keypoints_2d == all_pred_2d_img[i, :21]).all()
            #     pred_img = render_openpose(pred_img, body_keypoints_2d)

            ###############################################
            # Draw 2D hand keypoints onto original image
            ###############################################
            pred_img = (input_img[:, :, :3] * 255).astype(np.uint8).copy()

            for i in range(len(all_verts)):

                kp = all_pred_2d_img[i, :21]
                valid = kp[:, 2] > 0.1
                keypoints_xy = kp[:, :2]
                color = RIGHT_COLOR if all_right[i] == 1 else LEFT_COLOR

                # draw
                _draw_hand(
                    image=pred_img,
                    keypoints=keypoints_xy,
                    valid=valid,
                    color=color,
                    circle_radius=4,
                    line_thickness=2,
                )
            

            # --------------------------------------------
            # Draw bboxes (left: red, right: blue)
            # --------------------------------------------
            bbox_img = (input_img[:, :, :3] * 255).astype(np.uint8).copy()

            for idx, bbox in enumerate(bboxes):
                is_right = is_right_list[idx]
                if is_right == 1:
                    color = BBOX_RIGHT_COLOR
                    label = "right"
                else:
                    color = BBOX_LEFT_COLOR
                    label = "left"
                bbox_img = draw_bbox_and_label(bbox_img, bbox, label, color)

            h, w = bbox_img.shape[:2]
            target_w, target_h = 512, int(h * (512 / w))
            bbox_img_512 = cv2.resize(bbox_img, (target_w, target_h), interpolation=cv2.INTER_AREA)


            # save main render
            cv2.imwrite(os.path.join(render_save_path, f"{img_fn}.jpg"), overlay_bgr)
            cv2.imwrite(os.path.join(joint2d_save_path, f"{img_fn}.jpg"), pred_img[:, :, ::-1])
            cv2.imwrite(os.path.join(vit_save_path, f"{img_fn}.jpg"), bbox_img[:, :, ::-1])
            print("saved image:", os.path.join(render_save_path, f"{img_fn}.jpg"))

        c = time.time()
        print("one step time:", c - a)

    # ----------------------------------------------------------------------
    # Optional: save meshes (same behavior as原始代码，保留但按需开启)
    # ----------------------------------------------------------------------
    if len(big_all_cam_t) > 0 and len(big_all_cam_t[0]) > 0:
        if len(big_all_joints[0]) == 1:
            init_trans = big_all_cam_t[0][0].copy() + big_all_joints[0][0][9]
        else:
            x = (
                big_all_cam_t[0][0]
                + big_all_joints[0][0][9]
                + big_all_cam_t[0][1]
                + big_all_joints[0][1][9]
            ) / 2
            init_trans = big_all_cam_t[0][0].copy()

        if args.save_mesh:
            N = 0
            for verts_list, joints_list, cam_list, right_list in zip(
                big_all_verts, big_all_joints, big_all_cam_t, big_all_right
            ):
                for verts, joints, cam_t, is_right in zip(
                    verts_list, joints_list, cam_list, right_list
                ):
                    camera_translation = cam_t.copy() - init_trans
                    tmesh = renderer.vertices_to_trimesh(
                        verts, camera_translation, LIGHT_BLUE, is_right=is_right
                    )
                    obj_name = f"{str(N).zfill(6)}_{is_right}.obj"
                    tmesh.export(os.path.join(mesh_dir, obj_name))
                    N += 1

    # ----------------------------------------------------------------------
    # Optional: save videos
    # ----------------------------------------------------------------------
    if args.render:
        if args.res_folder is not None:
            save_video(
                render_save_path,
                base_dir,
                args.out_folder,
            )
            save_video(
                joint2d_save_path,
                base_dir,
                args.out_folder + "_2d",
            )
            save_video(
                vit_save_path,
                base_dir,
                args.out_folder + "_vit",
            )
        else:
            save_video(render_save_path, ".", args.out_folder)
            save_video(joint2d_save_path, ".", args.out_folder + "_2d")
            save_video(vit_save_path, ".", args.out_folder + "_vit")

    # ----------------------------------------------------------------------
    # Post-process results_dict and save
    # ----------------------------------------------------------------------
    for k in results_dict.keys():
        if (
            len(results_dict[k].get("tid", [])) > 1
            and len(results_dict[k]["mano"]) == 1
        ):
            assert (results_dict[k]["tid"] == [0, 1]).all()
            if results_dict[k]["mano"][0]["is_right"] == 0:
                continue
            elif results_dict[k]["mano"][0]["is_right"] == 1:
                results_dict[k]["mano"].insert(0, -100)
                results_dict[k]["cam_trans"].insert(0, -100)
                if len(results_dict[k]["extra_data"]) == 1:
                    d2 = results_dict[k]["extra_data"][0]
                    results_dict[k]["extra_data"] = [-100, d2]

    if args.res_folder is not None:
        print("Saving results_dict to:", args.res_folder)
        with open(args.res_folder, "wb") as f:
            pickle.dump(results_dict, f)
    else:
        print("WARNING: res_folder is None, results_dict not saved to disk.")

    total_time = time.time() - start_time_all
    print("Total time:", total_time, "seconds")


if __name__ == "__main__":
    print("Start HaMeR demo !!!!! simplified pkl-only !!!!!")
    main()