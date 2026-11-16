# TESA TopGun 2026 - Master Evaluation Server

Central orchestrator for the distributed evaluation system. Manages high-speed parallel downloads from Google Drive, validates contestant submissions, schedules evaluation jobs across Raspberry Pi slaves via FastAPI, and aggregates final leaderboards.

---

## Features

- **Dynamic 3-Day Competition Pipeline:** Separate data, ground truth, and scoring rules for each day:
  - **Day 1 (Classification - CSV):** Simple image classification. Counts correct images (`correct_pics`), evaluates exact-match Accuracy (`correct / total`), Macro F1, and Latency.
  - **Day 2 (1-Class Masking - JSON):** Binary object segmentation. Calculates pixel/polygon IoU with a $\ge 0.95$ threshold, evaluates True Positives (TP), False Positives (FP), False Negatives (FN), Precision, Recall, and F1-score.
  - **Day 3 (Combined Masking + Multi-Class Classification - JSON):** Evaluates object segmentation matched with correct class labels using $\text{mAP@95}$, Class Accuracy, and composite score.
- **High-Speed Two-Stage Downloader:**
  - *Stage 1:* Concurrent parallel downloads of all raw `.zip` archives directly from Google Drive / links.
  - *Stage 2:* Batch extraction, validation (`infer.py`, `requirements.txt`, `weights/`), and bundling.
- **FastAPI Core:** Interactive Swagger UI at `/docs`, REST endpoints for Pi slaves, and dynamic leaderboard generation.
- **Leaderboard Aggregation:** Instant scoring tailored per day and automatic export to `final_leaderboard.csv`.

---

## Directory Structure

```text
master/
├── .env                     # Server host, port, active day config
├── requirements.txt         # FastAPI, uvicorn, gdown, pandas, etc.
├── run_master.py            # CLI entry point (serve, download, prepare)
├── app/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Pydantic Settings
│   ├── models/job.py        # Pydantic data schemas
│   ├── services/
│   │   ├── downloader.py    # High-speed parallel downloader
│   │   ├── packager.py      # Batch zip unpacker & bundle creator
│   │   ├── queue_manager.py # Thread-safe multi-day job queue
│   │   ├── evaluator.py     # Scores predictions vs ground_truth.csv
│   │   └── aggregator.py    # Generates final_leaderboard.csv
│   └── routers/
│       ├── admin.py         # Pipeline control endpoints
│       ├── jobs.py          # Slave communication (/poll, /bundle, /return)
│       └── leaderboard.py   # Leaderboard queries & CSV export
├── data/
│   ├── day_1/               # contestants.csv, ground_truth.csv, raw_zips, bundles
│   ├── day_2/
│   └── day_3/
└── results/
    ├── day_1/final_leaderboard.csv
    ├── day_2/final_leaderboard.csv
    └── day_3/final_leaderboard.csv
```

---

## Quick Start

### 1. Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Day Data
Fill `data/day_{N}/contestants.csv` with contestant IDs and Drive links:
```csv
team_id,team_name,drive_url
team_01,Alpha,https://drive.google.com/file/d/YOUR_FILE_ID/view
```
Place your test dataset labels into `data/day_{N}/ground_truth.csv`:
```csv
filename,class
image_001.jpg,good
image_002.jpg,full_black
```

### 3. Execution Options

#### Option A: Running via CLI
```bash
# 1. Bulk download all contestant zips for Day 1
python run_master.py download --day 1

# 2. Batch unpack, validate files, and build bundles
python run_master.py prepare --day 1

# 3. Start the FastAPI server to serve slaves
python run_master.py serve
```

#### Option B: Running via FastAPI Swagger UI
1. Run `python run_master.py serve`
2. Open `http://localhost:8000/docs` in your browser.
3. Trigger `/api/v1/admin/day/1/download-all`
4. Trigger `/api/v1/admin/day/1/prepare-queue`
5. Monitor connected slaves and download `final_leaderboard_day_1.csv` from `/api/v1/leaderboard/day/1/csv`.