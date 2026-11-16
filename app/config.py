from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    MASTER_HOST: str = "0.0.0.0"
    MASTER_PORT: int = 8000
    SECRET_TOKEN: str = "tesa_topgun_secret_key_2026"
    
    # Active competition day (1, 2, or 3)
    ACTIVE_DAY: int = Field(default=1, ge=1, le=3)
    
    # Base storage directories relative to master root
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: str = "data"
    RESULTS_DIR: str = "results"
    
    # Downloader settings
    DOWNLOAD_MAX_CONCURRENT: int = 6
    DOWNLOAD_TIMEOUT_SECONDS: int = 180
    
    # Job timeout
    JOB_TIMEOUT_SECONDS: int = 300

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    # Helper methods for day-specific directories and files
    def get_data_dir(self) -> Path:
        path = self.BASE_DIR / self.DATA_DIR
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_results_dir(self) -> Path:
        path = self.BASE_DIR / self.RESULTS_DIR
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_day_data_dir(self, day: int) -> Path:
        path = self.get_data_dir() / f"day_{day}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_day_results_dir(self, day: int) -> Path:
        path = self.get_results_dir() / f"day_{day}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_contestants_csv(self, day: int) -> Path:
        return self.get_day_data_dir(day) / "contestants.csv"

    def get_ground_truth_csv(self, day: int) -> Path:
        return self.get_day_data_dir(day) / "ground_truth.csv"

    def get_raw_zips_dir(self, day: int) -> Path:
        path = self.get_day_data_dir(day) / "raw_zips"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_extracted_dir(self, day: int) -> Path:
        path = self.get_day_data_dir(day) / "extracted"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_bundles_dir(self, day: int) -> Path:
        path = self.get_day_data_dir(day) / "bundles"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_leaderboard_csv(self, day: int) -> Path:
        return self.get_day_results_dir(day) / "final_leaderboard.csv"


settings = Settings()
