import csv
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from app.models.job import EvaluationScore


class SubmissionEvaluator:
    """
    Evaluates predictions returned by slaves against day-specific ground truth labels.
    Calculates accuracy, macro F1-score, and latency metrics.
    """

    def load_ground_truth(self, ground_truth_path: Path) -> Dict[str, str]:
        """
        Loads ground truth CSV mapping filename -> true class.
        Supports headers: (filename/file/name, class/label/target)
        """
        if not ground_truth_path.exists():
            raise FileNotFoundError(f"Ground truth file not found at: {ground_truth_path}")

        gt = {}
        with open(ground_truth_path, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cleaned = {k.strip().lower(): v.strip() for k, v in row.items() if k and v}
                fname = cleaned.get("filename") or cleaned.get("file") or cleaned.get("name")
                label = cleaned.get("class") or cleaned.get("label") or cleaned.get("target")
                if fname and label:
                    gt[fname] = label
        return gt

    def evaluate(
        self,
        ground_truth_path: Path,
        predictions: Optional[List[Dict[str, Any]]],
        timings: Optional[List[Dict[str, Any]]],
        benchmark: Optional[Dict[str, Any]],
        total_wall_time_s: Optional[float]
    ) -> EvaluationScore:
        if not predictions:
            return EvaluationScore(total_images=0, error="No predictions provided")

        gt = self.load_ground_truth(ground_truth_path)
        if not gt:
            raise ValueError(f"Ground truth file {ground_truth_path} contains no valid rows")

        y_true = []
        y_pred = []
        correct_count = 0

        # Match predictions against ground truth
        pred_map = {}
        for p in predictions:
            fn = p.get("filename", "")
            cl = p.get("class") or p.get("label", "")
            if fn:
                pred_map[fn] = str(cl)

        for filename, true_label in gt.items():
            predicted_label = pred_map.get(filename, "__MISSING__")
            y_true.append(true_label)
            y_pred.append(predicted_label)
            if predicted_label == true_label:
                correct_count += 1

        total_samples = len(gt)
        acc = accuracy_score(y_true, y_pred) if total_samples > 0 else 0.0
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) if total_samples > 0 else 0.0

        # Latency metrics
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

        # Composite score formula:
        # Example: 80% weight on Macro F1 (scaled to 100), with a latency bonus/penalty
        # Organizers can adjust this in the leaderboard
        composite = float((macro_f1 * 100.0) - min(mean_proc_s * 5.0, 20.0))

        return EvaluationScore(
            total_images=total_samples,
            correct_predictions=correct_count,
            accuracy=round(float(acc), 4),
            macro_f1=round(float(macro_f1), 4),
            mean_processing_seconds=round(mean_proc_s, 4),
            mean_total_seconds=round(mean_total_s, 4),
            model_load_seconds=round(model_load_s, 4),
            total_wall_time_s=round(wall_time, 4),
            composite_score=round(composite, 4)
        )
