import csv
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from PIL import Image, ImageDraw
from sklearn.metrics import f1_score, accuracy_score
from app.models.job import EvaluationScore


def _normalize_image_key(name: Any) -> str:
    """Normalizes image filenames/paths to lower-case stem for consistent matching."""
    s = str(name).strip()
    return Path(s).stem.lower()


def normalize_polygon(poly_data: Any) -> Optional[np.ndarray]:
    """
    Converts any polygon representation into an (N, 2) numpy array of int32 coordinates.
    Requires at least 3 points.
    """
    if poly_data is None:
        return None

    if isinstance(poly_data, np.ndarray):
        if poly_data.ndim == 2 and poly_data.shape[1] == 2:
            pts = poly_data.astype(np.int32)
            return pts if len(pts) >= 3 else None
        elif poly_data.ndim == 1 and len(poly_data) >= 6 and len(poly_data) % 2 == 0:
            pts = poly_data.reshape(-1, 2).astype(np.int32)
            return pts if len(pts) >= 3 else None

    if isinstance(poly_data, list):
        if not poly_data:
            return None
        first = poly_data[0]
        if isinstance(first, (list, tuple)) and len(first) >= 2:
            try:
                pts = np.array([[int(round(float(p[0]))), int(round(float(p[1])))] for p in poly_data], dtype=np.int32)
                return pts if len(pts) >= 3 else None
            except Exception:
                return None
        elif isinstance(first, (int, float)):
            if len(poly_data) >= 6 and len(poly_data) % 2 == 0:
                try:
                    pts = np.array([[int(round(float(poly_data[i]))), int(round(float(poly_data[i + 1])))] for i in range(0, len(poly_data), 2)], dtype=np.int32)
                    return pts if len(pts) >= 3 else None
                except Exception:
                    return None
    return None


def extract_polygon(obj: Any) -> Optional[np.ndarray]:
    """Extracts a valid polygon coordinate array from various object dictionary or list shapes."""
    if isinstance(obj, dict):
        poly = obj.get("polygon") or obj.get("segmentation") or obj.get("mask") or obj.get("points")
        if poly is not None:
            return normalize_polygon(poly)
    elif isinstance(obj, (list, np.ndarray)):
        return normalize_polygon(obj)
    return None


def compute_polygon_iou(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Computes exact rasterized Intersection-over-Union (IoU) between two polygons
    using an optimized local sub-canvas bounding box.
    """
    min_x1, min_y1 = p1.min(axis=0)
    max_x1, max_y1 = p1.max(axis=0)
    min_x2, min_y2 = p2.min(axis=0)
    max_x2, max_y2 = p2.max(axis=0)

    # Fast rejection via bounding box overlap
    if max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1:
        return 0.0

    # Local canvas on bounding box union
    u_min_x = int(min(min_x1, min_x2))
    u_min_y = int(min(min_y1, min_y2))
    u_max_x = int(max(max_x1, max_x2))
    u_max_y = int(max(max_y1, max_y2))

    w = u_max_x - u_min_x + 1
    h = u_max_y - u_min_y + 1
    if w <= 0 or h <= 0 or w > 8000 or h > 8000:
        return 0.0

    m1 = np.zeros((h, w), dtype=np.uint8)
    m2 = np.zeros((h, w), dtype=np.uint8)

    cv2.fillPoly(m1, [p1 - [u_min_x, u_min_y]], 1)
    cv2.fillPoly(m2, [p2 - [u_min_x, u_min_y]], 1)

    intersection = int(np.logical_and(m1, m2).sum())
    union = int(np.logical_or(m1, m2).sum())
    return float(intersection) / float(union) if union > 0 else 0.0


def _parse_day2_predictions(data: Any) -> Dict[str, List[Any]]:
    """
    Normalizes Day 2 ground truth and prediction inputs into:
    { "parchment_test": [obj1, obj2, ...], ... }
    """
    result = {}
    if not data:
        return result

    if isinstance(data, dict):
        if "images" in data and isinstance(data["images"], list):
            data = data["images"]
        else:
            for k, v in data.items():
                key = _normalize_image_key(k)
                if isinstance(v, list):
                    result[key] = v
                elif isinstance(v, dict):
                    objs = v.get("objects") or v.get("polygons") or v.get("masks") or [v]
                    result[key] = objs

    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            name = (
                item.get("image_name")
                or item.get("filename")
                or item.get("file")
                or item.get("image")
                or item.get("name")
            )
            if not name:
                continue
            key = _normalize_image_key(name)
            objs = item.get("objects") or item.get("polygons") or item.get("masks")
            if objs is None:
                if "polygon" in item or "segmentation" in item:
                    objs = [item]
                else:
                    objs = []
            result[key] = objs

    return result


def _extract_timings(
    timings: Optional[List[Dict[str, Any]]],
    benchmark: Optional[Dict[str, Any]],
    total_wall_time_s: Optional[float]
) -> Tuple[float, float, float, float]:
    model_load_s = 0.0
    mean_proc_s = 0.0
    mean_total_s = 0.0

    if benchmark:
        model_load_s = float(benchmark.get("model_load_seconds", 0.0))
        mean_proc_s = float(benchmark.get("mean_processing_seconds", 0.0))
        mean_total_s = float(benchmark.get("mean_total_seconds", 0.0))
    elif timings and len(timings) > 0:
        proc_times = [float(t.get("processing_seconds", 0.0)) for t in timings if "processing_seconds" in t]
        tot_times = [float(t.get("total_seconds", 0.0)) for t in timings if "total_seconds" in t]
        if proc_times:
            mean_proc_s = float(np.mean(proc_times))
        if tot_times:
            mean_total_s = float(np.mean(tot_times))

    wall_time = float(total_wall_time_s or 0.0)
    return round(mean_proc_s, 4), round(mean_total_s, 4), round(model_load_s, 4), round(wall_time, 4)


def _mask_to_binary_array(mask_obj: Any, canvas_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
    """
    Converts various mask representations (polygon coordinates, bounding box, or 2D array)
    into a boolean 2D numpy mask.
    """
    if isinstance(mask_obj, np.ndarray):
        return mask_obj.astype(bool)

    if isinstance(mask_obj, list):
        if not mask_obj:
            return np.zeros(canvas_size, dtype=bool)

        # Check if polygon: list of [x, y] or list of numbers [x1, y1, x2, y2, ...]
        first_elem = mask_obj[0]
        if isinstance(first_elem, (list, tuple)) and len(first_elem) == 2:
            # Format: [[x1, y1], [x2, y2], ...]
            poly_tuples = [(float(pt[0]), float(pt[1])) for pt in mask_obj]
        elif isinstance(first_elem, (int, float)):
            if len(mask_obj) == 4:
                # Bounding box format: [x1, y1, x2, y2]
                x1, y1, x2, y2 = [int(v) for v in mask_obj]
                arr = np.zeros(canvas_size, dtype=bool)
                arr[max(0, y1):min(canvas_size[0], y2), max(0, x1):min(canvas_size[1], x2)] = True
                return arr
            elif len(mask_obj) >= 6 and len(mask_obj) % 2 == 0:
                # Flat polygon: [x1, y1, x2, y2, ...]
                poly_tuples = [(float(mask_obj[i]), float(mask_obj[i + 1])) for i in range(0, len(mask_obj), 2)]
            else:
                return np.zeros(canvas_size, dtype=bool)
        else:
            return np.zeros(canvas_size, dtype=bool)

        # Rasterize polygon using PIL
        img = Image.new("1", (canvas_size[1], canvas_size[0]), 0)
        draw = ImageDraw.Draw(img)
        draw.polygon(poly_tuples, outline=1, fill=1)
        return np.array(img, dtype=bool)

    elif isinstance(mask_obj, dict):
        if "polygon" in mask_obj:
            return _mask_to_binary_array(mask_obj["polygon"], canvas_size)
        if "segmentation" in mask_obj:
            return _mask_to_binary_array(mask_obj["segmentation"], canvas_size)
        if "bbox" in mask_obj:
            return _mask_to_binary_array(mask_obj["bbox"], canvas_size)
        if "counts" in mask_obj and "size" in mask_obj:
            # RLE format placeholder
            h, w = mask_obj.get("size", canvas_size)
            return np.zeros((h, w), dtype=bool)

    return np.zeros(canvas_size, dtype=bool)


def _compute_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Calculates Intersection-over-Union (IoU) between two binary masks."""
    if mask_a.shape != mask_b.shape:
        # Resize or pad if shapes differ
        min_h = min(mask_a.shape[0], mask_b.shape[0])
        min_w = min(mask_a.shape[1], mask_b.shape[1])
        mask_a = mask_a[:min_h, :min_w]
        mask_b = mask_b[:min_h, :min_w]

    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 1.0 if not mask_a.any() and not mask_b.any() else 0.0
    return float(intersection / union)


class SubmissionEvaluator:
    """
    Dynamic Multi-Day Evaluator:
    - Day 1: Simple Classification (Count correct predictions, Accuracy, CSV)
    - Day 2: Binary Object Masking (IoU >= 0.95, TP, FP, FN, F1-score, JSON)
    - Day 3: Combined Masking + Multi-Class Classification (mAP@95, Class Accuracy, JSON)
    """

    # -------------------------------------------------------------
    # DAY 1: Classification Evaluator
    # -------------------------------------------------------------
    def evaluate_day1(
        self,
        ground_truth_path: Path,
        predictions: Optional[List[Dict[str, Any]]],
        timings: Optional[List[Dict[str, Any]]],
        benchmark: Optional[Dict[str, Any]],
        total_wall_time_s: Optional[float]
    ) -> EvaluationScore:
        mean_p, mean_t, load_s, wall_s = _extract_timings(timings, benchmark, total_wall_time_s)

        if not predictions:
            return EvaluationScore(day=1, total_images=0, mean_processing_seconds=mean_p,
                                   mean_total_seconds=mean_t, model_load_seconds=load_s, total_wall_time_s=wall_s)

        def clean_label(label: str) -> str:
            lbl = str(label).strip().lower()
            if lbl in ("black", "full_black"):
                return "full_black"
            return lbl

        # Load CSV ground truth
        gt = {}
        with open(ground_truth_path, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cleaned = {k.strip().lower(): v.strip() for k, v in row.items() if k and v}
                fname = cleaned.get("filename") or cleaned.get("file") or cleaned.get("name")
                label = cleaned.get("class") or cleaned.get("label") or cleaned.get("target")
                if fname and label:
                    gt[fname] = clean_label(label)

        pred_map = {}
        if isinstance(predictions, list):
            for p in predictions:
                if isinstance(p, dict):
                    fn = p.get("filename", "")
                    cl = p.get("class") or p.get("label", "")
                    if fn:
                        pred_map[fn] = clean_label(cl)

        y_true, y_pred = [], []
        correct_count = 0
        for filename, true_clean in gt.items():
            pred_clean = pred_map.get(filename, "__MISSING__")
            y_true.append(true_clean)
            y_pred.append(pred_clean)
            if pred_clean == true_clean:
                correct_count += 1

        total_images = len(gt)
        acc = float(accuracy_score(y_true, y_pred)) if total_images > 0 else 0.0
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if total_images > 0 else 0.0
        composite = round(acc * 100.0, 4)

        return EvaluationScore(
            day=1,
            total_images=total_images,
            correct_predictions=correct_count,
            accuracy=round(acc, 4),
            macro_f1=round(macro_f1, 4),
            mean_processing_seconds=mean_p,
            mean_total_seconds=mean_t,
            model_load_seconds=load_s,
            total_wall_time_s=wall_s,
            composite_score=composite
        )

    # -------------------------------------------------------------
    # DAY 2: Object Masking & Counting (Hungarian Matching, IoU, Count Acc)
    # -------------------------------------------------------------
    def evaluate_day2(
        self,
        ground_truth_path: Path,
        predictions_data: Any,
        timings: Optional[List[Dict[str, Any]]],
        benchmark: Optional[Dict[str, Any]],
        total_wall_time_s: Optional[float],
        iou_threshold: float = 0.95
    ) -> EvaluationScore:
        mean_p, mean_t, load_s, wall_s = _extract_timings(timings, benchmark, total_wall_time_s)
        latency = wall_s if wall_s > 0 else (mean_t if mean_t > 0 else 0.0)

        # Parse Ground Truth JSON
        with open(ground_truth_path, mode="r", encoding="utf-8") as f:
            gt_data = json.load(f)

        gt_dict = _parse_day2_predictions(gt_data)
        pred_dict = _parse_day2_predictions(predictions_data) if predictions_data else {}

        total_gt_objects = 0
        total_pred_objects = 0
        per_image_count_scores = []
        all_matched_ious = []
        iou95_count = 0
        iou90_count = 0
        image_details = {}

        for img_key, gt_objs in gt_dict.items():
            # Extract valid polygons for ground truth
            gt_polys = []
            for obj in gt_objs:
                p = extract_polygon(obj)
                if p is not None:
                    gt_polys.append(p)

            pred_objs = pred_dict.get(img_key, [])
            pred_polys = []
            for obj in pred_objs:
                p = extract_polygon(obj)
                if p is not None:
                    pred_polys.append(p)

            num_gt = len(gt_polys)
            num_pred = len(pred_polys)
            total_gt_objects += num_gt
            total_pred_objects += num_pred

            # Count accuracy for this image:
            # max(0, (num_gt - abs(num_pred - num_gt)) / num_gt)
            if num_gt > 0:
                pic_score = max(0.0, float(num_gt - abs(num_pred - num_gt)) / float(num_gt))
            else:
                pic_score = 1.0 if num_pred == 0 else 0.0
            per_image_count_scores.append(pic_score)

            # Match polygons using Hungarian algorithm (scipy.optimize.linear_sum_assignment)
            # Cost matrix shape: (num_pred, num_gt)
            if num_pred > 0 and num_gt > 0:
                cost_matrix = np.zeros((num_pred, num_gt), dtype=np.float32)
                for i, p_poly in enumerate(pred_polys):
                    for j, g_poly in enumerate(gt_polys):
                        cost_matrix[i, j] = compute_polygon_iou(p_poly, g_poly)

                # linear_sum_assignment minimizes cost, so minimize -IoU
                row_ind, col_ind = linear_sum_assignment(-cost_matrix)

                gt_matched_iou = {j: 0.0 for j in range(num_gt)}
                for r, c in zip(row_ind, col_ind):
                    gt_matched_iou[c] = float(cost_matrix[r, c])

                img_ious = list(gt_matched_iou.values())
            else:
                img_ious = [0.0] * num_gt

            for iou in img_ious:
                all_matched_ious.append(iou)
                if iou >= 0.95:
                    iou95_count += 1
                if iou >= 0.90:
                    iou90_count += 1

            image_details[img_key] = {
                "gt_count": num_gt,
                "pred_count": num_pred,
                "count_score": round(pic_score * 100.0, 2),
                "matched_ious": [round(x, 4) for x in img_ious],
                "mean_iou": round(float(np.mean(img_ious)), 4) if img_ious else 0.0
            }

        avg_count_acc = float(np.mean(per_image_count_scores)) * 100.0 if per_image_count_scores else 0.0
        avg_iou = float(np.mean(all_matched_ious)) if all_matched_ious else 0.0

        # Composite score: 50% count accuracy + 50% average IoU (scaled to 100)
        composite = round(0.5 * avg_count_acc + 0.5 * (avg_iou * 100.0), 4)

        # Standard binary detection metrics at iou_threshold
        tp = sum(1 for iou in all_matched_ious if iou >= iou_threshold)
        fp = max(0, total_pred_objects - tp)
        fn = total_gt_objects - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return EvaluationScore(
            day=2,
            total_images=len(gt_dict),
            total_gt_objects=total_gt_objects,
            total_pred_objects=total_pred_objects,
            count_accuracy=round(avg_count_acc, 2),
            iou95=iou95_count,
            iou90=iou90_count,
            iou_average=round(avg_iou, 4),
            latency=round(latency, 4),
            tp=tp,
            fp=fp,
            fn=fn,
            precision=round(float(precision), 4),
            recall=round(float(recall), 4),
            f1_score=round(float(f1), 4),
            mean_iou=round(avg_iou, 4),
            iou_threshold=iou_threshold,
            mean_processing_seconds=mean_p,
            mean_total_seconds=mean_t,
            model_load_seconds=load_s,
            total_wall_time_s=wall_s,
            composite_score=composite,
            details={"per_image": image_details}
        )

    # -------------------------------------------------------------
    # DAY 3: Combined Masking + Multi-Class Classification (mAP@95)
    # -------------------------------------------------------------
    def evaluate_day3(
        self,
        ground_truth_path: Path,
        predictions_data: Any,
        timings: Optional[List[Dict[str, Any]]],
        benchmark: Optional[Dict[str, Any]],
        total_wall_time_s: Optional[float],
        iou_threshold: float = 0.95
    ) -> EvaluationScore:
        mean_p, mean_t, load_s, wall_s = _extract_timings(timings, benchmark, total_wall_time_s)

        with open(ground_truth_path, mode="r", encoding="utf-8") as f:
            gt_data = json.load(f)

        gt_dict = self._normalize_detection_json(gt_data)
        pred_dict = self._normalize_detection_json(predictions_data) if predictions_data else {}

        # Collect all unique classes
        all_classes = set()
        for objs in gt_dict.values():
            for o in objs:
                all_classes.add(o.get("class", "object"))

        class_aps = {}
        total_tp = 0
        total_fp = 0
        total_fn = 0
        correct_class_matches = 0
        total_matches = 0
        matched_ious = []

        # Evaluate per class for mAP@95
        for cls in all_classes:
            cls_preds = []
            cls_gt_count = 0

            for fname, gt_objs in gt_dict.items():
                cls_gts = [o for o in gt_objs if o.get("class") == cls]
                cls_gt_count += len(cls_gts)
                img_preds = [p for p in pred_dict.get(fname, []) if p.get("class") == cls]
                for p in img_preds:
                    cls_preds.append((fname, p))

            # Sort predictions by confidence score descending
            cls_preds.sort(key=lambda item: float(item[1].get("confidence", item[1].get("score", 1.0))), reverse=True)

            tp_list = []
            fp_list = []
            matched_gt_per_img: Dict[str, set] = {f: set() for f in gt_dict}

            for fname, pred_obj in cls_preds:
                pred_mask = _mask_to_binary_array(pred_obj.get("mask", pred_obj.get("polygon", pred_obj.get("bbox"))))
                gt_objs = [o for o in gt_dict.get(fname, []) if o.get("class") == cls]

                best_iou = 0.0
                best_gt_idx = -1
                for idx, g_obj in enumerate(gt_objs):
                    if idx in matched_gt_per_img[fname]:
                        continue
                    g_mask = _mask_to_binary_array(g_obj.get("mask", g_obj.get("polygon", g_obj.get("bbox"))))
                    iou = _compute_mask_iou(pred_mask, g_mask)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx

                if best_iou >= iou_threshold and best_gt_idx != -1:
                    tp_list.append(1)
                    fp_list.append(0)
                    matched_gt_per_img[fname].add(best_gt_idx)
                    total_tp += 1
                    correct_class_matches += 1
                    total_matches += 1
                    matched_ious.append(best_iou)
                else:
                    tp_list.append(0)
                    fp_list.append(1)
                    total_fp += 1

            total_fn += (cls_gt_count - sum(tp_list))

            # Compute AP using 11-point interpolation or AUC
            if cls_gt_count == 0:
                continue

            cum_tp = np.cumsum(tp_list)
            cum_fp = np.cumsum(fp_list)
            recalls = cum_tp / cls_gt_count
            precisions = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float64).eps)

            # 11-point interpolation
            ap = 0.0
            for t in np.arange(0.0, 1.1, 0.1):
                p_interp = np.max(precisions[recalls >= t]) if np.sum(recalls >= t) > 0 else 0.0
                ap += p_interp / 11.0
            class_aps[cls] = float(ap)

        map_95 = float(np.mean(list(class_aps.values()))) if class_aps else 0.0
        class_acc = (correct_class_matches / total_matches) if total_matches > 0 else 0.0
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0

        return EvaluationScore(
            day=3,
            total_images=len(gt_dict),
            tp=total_tp,
            fp=total_fp,
            fn=total_fn,
            precision=round(float(precision), 4),
            recall=round(float(recall), 4),
            f1_score=round(float(f1), 4),
            mean_iou=round(mean_iou, 4),
            iou_threshold=iou_threshold,
            map_95=round(map_95, 4),
            class_accuracy=round(float(class_acc), 4),
            mean_processing_seconds=mean_p,
            mean_total_seconds=mean_t,
            model_load_seconds=load_s,
            total_wall_time_s=wall_s,
            composite_score=round(float(map_95 * 100.0), 4),
            details={"per_class_ap95": class_aps}
        )

    # -------------------------------------------------------------
    # Shared Helper Parsers
    # -------------------------------------------------------------
    @staticmethod
    def _normalize_mask_json(data: Any) -> Dict[str, List[Any]]:
        """Normalizes Day 2 JSON input into {filename: [mask, mask, ...]}."""
        result = {}
        if isinstance(data, dict):
            for k, v in data.items():
                result[k] = v if isinstance(v, list) else [v]
        elif isinstance(data, list):
            for item in data:
                fn = item.get("filename") or item.get("file")
                masks = item.get("masks") or item.get("objects") or [item.get("mask")] or [item]
                if fn:
                    result[fn] = masks
        return result

    @staticmethod
    def _normalize_detection_json(data: Any) -> Dict[str, List[Dict[str, Any]]]:
        """Normalizes Day 3 JSON input into {filename: [{class: '...', mask: ...}, ...]}."""
        result = {}
        if isinstance(data, dict):
            for k, v in data.items():
                result[k] = v if isinstance(v, list) else [v]
        elif isinstance(data, list):
            for item in data:
                fn = item.get("filename") or item.get("file")
                objs = item.get("objects") or item.get("predictions") or [item]
                if fn:
                    result[fn] = objs
        return result

    # -------------------------------------------------------------
    # Main Dispatcher
    # -------------------------------------------------------------
    def evaluate(
        self,
        day: int,
        ground_truth_path: Path,
        predictions: Any,
        timings: Optional[List[Dict[str, Any]]],
        benchmark: Optional[Dict[str, Any]],
        total_wall_time_s: Optional[float]
    ) -> EvaluationScore:
        if day == 1:
            return self.evaluate_day1(ground_truth_path, predictions, timings, benchmark, total_wall_time_s)
        elif day == 2:
            return self.evaluate_day2(ground_truth_path, predictions, timings, benchmark, total_wall_time_s)
        elif day == 3:
            return self.evaluate_day3(ground_truth_path, predictions, timings, benchmark, total_wall_time_s)
        else:
            raise ValueError(f"Unsupported day {day}. Must be 1, 2, or 3.")
