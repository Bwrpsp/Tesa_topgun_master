from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    ASSIGNED = "ASSIGNED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"


class HardwareReport(BaseModel):
    start_temp: Optional[float] = None
    end_temp: Optional[float] = None
    throttled_flag: Optional[str] = "0x0"
    load_avg: Optional[float] = None
    memory_used_mb: Optional[float] = None


class JobPollRequest(BaseModel):
    slave_id: str
    current_temp: Optional[float] = None
    throttled_flag: Optional[str] = "0x0"
    is_idle: bool = True


class JobPollResponse(BaseModel):
    job_id: str
    day: int
    team_name: str
    bundle_url: str
    test_data_dir_name: str
    timeout_seconds: int = 300


class JobReturnRequest(BaseModel):
    slave_id: str
    status: str  # "SUCCESS" or "FAILED"
    predictions: Optional[Any] = None  # List of CSV rows for Day 1, or JSON (list/dict) for Day 2/3
    timings: Optional[List[Dict[str, Any]]] = None
    benchmark: Optional[Dict[str, Any]] = None
    total_wall_time_s: Optional[float] = None
    logs: Optional[str] = ""
    hardware_report: Optional[HardwareReport] = None
    error_message: Optional[str] = None


class EvaluationScore(BaseModel):
    day: int = 1
    total_images: int = 0

    # Day 1: Simple Classification (count correct pics)
    correct_predictions: int = 0
    accuracy: float = 0.0
    macro_f1: float = 0.0

    # Day 2: Binary Object Masking (IoU >= 0.95, TP, FP, FN, F1)
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    mean_iou: float = 0.0
    iou_threshold: float = 0.95

    # Day 3: Masking + Classification (mAP@95)
    map_95: float = 0.0
    class_accuracy: float = 0.0

    # Latency / Timing metrics
    mean_processing_seconds: float = 0.0
    mean_total_seconds: float = 0.0
    model_load_seconds: float = 0.0
    total_wall_time_s: float = 0.0

    # Final combined / weighted score
    composite_score: float = 0.0
    details: Optional[Dict[str, Any]] = None


class Job(BaseModel):
    job_id: str
    day: int
    team_name: str
    bundle_path: str
    status: JobStatus = JobStatus.QUEUED
    assigned_slave: Optional[str] = None
    assigned_at: Optional[float] = None
    completed_at: Optional[float] = None
    error_message: Optional[str] = None
    evaluation: Optional[EvaluationScore] = None
    hardware_report: Optional[HardwareReport] = None
