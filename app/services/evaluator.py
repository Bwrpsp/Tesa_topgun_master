import csv
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageDraw
from sklearn.metrics import f1_score, accuracy_score
from app.models.job import EvaluationScore


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
    # DAY 2: Binary Object Masking (IoU >= 0.95, TP / FP / FN / F1)
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

        # Parse Ground Truth JSON
        with open(ground_truth_path, mode="r", encoding="utf-8") as f:
            gt_data = json.load(f)

        # Normalize ground truth to: {filename: [mask1, mask2, ...]}
        gt_dict = self._normalize_mask_json(gt_data)

        # Normalize predictions to: {filename: [mask1, mask2, ...]}
        pred_dict = self._normalize_mask_json(predictions_data) if predictions_data else {}

        total_tp = 0
        total_fp = 0
        total_fn = 0
        matched_ious = []

        for filename, gt_masks in gt_dict.items():
            preds = pred_dict.get(filename, [])
            # Convert to binary arrays
            gt_binaries = [_mask_to_binary_array(m) for m in gt_masks]
            pred_binaries = [_mask_to_binary_array(m) for m in preds]

            # Compute pairwise IoU
            matched_gt = set()
            for p_idx, p_mask in enumerate(pred_binaries):
                best_iou = 0.0
                best_gt_idx = -1
                for g_idx, g_mask in enumerate(gt_binaries):
                    if g_idx in matched_gt:
                        continue
                    iou = _compute_mask_iou(p_mask, g_mask)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_idx

                if best_iou >= iou_threshold and best_gt_idx != -1:
                    total_tp += 1
                    matched_gt.add(best_gt_idx)
                    matched_ious.append(best_iou)
                else:
                    total_fp += 1

            # Unmatched ground truth masks are False Negatives
            total_fn += (len(gt_binaries) - len(matched_gt))

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0

        return EvaluationScore(
            day=2,
            total_images=len(gt_dict),
            tp=total_tp,
            fp=total_fp,
            fn=total_fn,
            precision=round(float(precision), 4),
            recall=round(float(recall), 4),
            f1_score=round(float(f1), 4),
            mean_iou=round(mean_iou, 4),
            iou_threshold=iou_threshold,
            mean_processing_seconds=mean_p,
            mean_total_seconds=mean_t,
            model_load_seconds=load_s,
            total_wall_time_s=wall_s,
            composite_score=round(float(f1 * 100.0), 4)
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
