import os
import zipfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def safe_extract_zip(zip_path: Path, extract_dir: Path):
    """
    Safely extracts all files from a zip archive.
    Normalizes Windows backslashes ('\\') in entry names to POSIX forward slashes ('/'),
    ensuring correct directory hierarchies are created on all operating systems (Linux, macOS, Windows).
    Guards against Zip Slip (directory traversal).
    """
    extract_dir = extract_dir.resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            # Replace backslashes with forward slashes
            clean_name = member.filename.replace("\\", "/").strip()
            # Strip leading slashes to prevent root-relative paths
            clean_name = clean_name.lstrip("/")
            if not clean_name:
                continue

            target_path = (extract_dir / clean_name).resolve()
            # Security: Prevent Zip Slip (escaping extract_dir)
            try:
                target_path.relative_to(extract_dir)
            except ValueError:
                # Path escapes extract_dir
                continue

            if member.is_dir() or clean_name.endswith("/"):
                target_path.mkdir(parents=True, exist_ok=True)
            else:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member, "r") as src, open(target_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)


def normalize_extracted_paths(base_dir: Path):
    """
    On POSIX filesystems where filenames may contain literal backslashes
    (e.g., 'TGR-46_day2\\infer.py' or 'TGR-46_day2\\weights\\model.bin'),
    recursively moves and restructures them into their proper directory hierarchy.
    """
    if not base_dir.exists():
        return
    for root, dirs, files in os.walk(base_dir, topdown=False):
        for name in files:
            if "\\" in name:
                full_path = Path(root) / name
                parts = [p for p in name.replace("\\", "/").split("/") if p]
                new_target = Path(root).joinpath(*parts)
                if new_target != full_path:
                    new_target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(full_path), str(new_target))
        for name in dirs:
            if "\\" in name:
                full_path = Path(root) / name
                parts = [p for p in name.replace("\\", "/").split("/") if p]
                new_target = Path(root).joinpath(*parts)
                if new_target != full_path:
                    new_target.parent.mkdir(parents=True, exist_ok=True)
                    if new_target.exists():
                        for item in full_path.iterdir():
                            shutil.move(str(item), str(new_target / item.name))
                        full_path.rmdir()
                    else:
                        shutil.move(str(full_path), str(new_target))


class SubmissionPackager:
    """
    Stage 2: Batch unpacks raw contestant zips, validates required files
    (infer.py, requirements.txt, weights/*, and auxiliary .py scripts),
    and creates standardized sanitized bundles ready for Raspberry Pi evaluation.
    """

    EXCLUDED_PATTERNS = {"__pycache__", ".git", ".venv", "venv", ".idea", ".vscode", ".DS_Store", "outputs"}

    def _find_submission_root(self, extract_dir: Path) -> Optional[Path]:
        """
        Locates the directory containing infer.py. Contestants often zip either
        the contents directly or the parent folder containing them.
        """
        normalize_extracted_paths(extract_dir)
        for root, dirs, files in os.walk(extract_dir):
            dirs[:] = [d for d in dirs if d not in self.EXCLUDED_PATTERNS]
            if "infer.py" in files:
                return Path(root)
        return None

    def _validate_submission(self, sub_root: Path) -> Tuple[bool, Optional[str], Optional[Path], List[str]]:
        """
        Checks that infer.py, requirements.txt, and weights/ folder exist and are valid.
        Accepts ANY file type/format inside weights/ as long as the folder is present and non-empty.
        Returns: (is_valid, error_message, weights_dir, list_of_weight_filenames)
        """
        infer_path = sub_root / "infer.py"
        req_path = sub_root / "requirements.txt"
        weights_dir = sub_root / "weights"

        if not infer_path.exists():
            return False, "Missing infer.py", None, []

        if not req_path.exists():
            return False, "Missing requirements.txt", None, []

        # Check weights folder exists
        if not weights_dir.exists() or not weights_dir.is_dir():
            return False, "Missing required 'weights' directory", None, []

        # Check that weights folder is not empty (accepts any file type/extension)
        weight_files = [f for f in weights_dir.rglob("*") if f.is_file()]
        if not weight_files:
            return False, "'weights' directory exists but is empty (no files found)", None, []

        weight_names = [f.name for f in weight_files]
        return True, None, weights_dir, weight_names

    def _create_clean_bundle(self, sub_root: Path, bundle_zip_path: Path):
        """
        Creates a clean, normalized zip file with infer.py, requirements.txt,
        weights/, and all Python helper modules at root.
        All entries are written with standard POSIX forward slashes.
        """
        bundle_zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(bundle_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(sub_root):
                # Filter out excluded directories
                dirs[:] = [d for d in dirs if d not in self.EXCLUDED_PATTERNS]
                
                rel_root = Path(root).relative_to(sub_root)
                for file in files:
                    if file in self.EXCLUDED_PATTERNS:
                        continue
                    file_path = Path(root) / file
                    rel_path = (rel_root / file) if str(rel_root) != "." else Path(file)
                    arcname = rel_path.as_posix()
                    zf.write(file_path, arcname=arcname)

    def process_team(self, team_id: str, raw_zip: Path, extract_base_dir: Path, bundles_dir: Path) -> Dict[str, any]:
        extract_dir = extract_base_dir / team_id
        bundle_file = bundles_dir / f"{team_id}.zip"

        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)

        try:
            safe_extract_zip(raw_zip, extract_dir)
        except Exception as e:
            return {"team_id": team_id, "is_valid": False, "error": f"Failed to unzip raw archive: {e}"}

        sub_root = self._find_submission_root(extract_dir)
        if not sub_root:
            return {"team_id": team_id, "is_valid": False, "error": "infer.py not found in the archive"}

        is_valid, error, weights_dir, weight_names = self._validate_submission(sub_root)
        if not is_valid:
            return {"team_id": team_id, "is_valid": False, "error": error}

        try:
            self._create_clean_bundle(sub_root, bundle_file)
            return {
                "team_id": team_id,
                "is_valid": True,
                "bundle_path": str(bundle_file),
                "weights_dir": "weights",
                "weights_count": len(weight_names),
                "weight_files": weight_names
            }
        except Exception as e:
            return {"team_id": team_id, "is_valid": False, "error": f"Failed to create bundle zip: {e}"}

    def process_all_for_day(self, raw_zips_dir: Path, extracted_dir: Path, bundles_dir: Path) -> Dict[str, any]:
        """Processes all downloaded zips for the given day."""
        bundles_dir.mkdir(parents=True, exist_ok=True)
        extracted_dir.mkdir(parents=True, exist_ok=True)

        results = []
        valid_count = 0
        invalid_count = 0

        for raw_zip in raw_zips_dir.glob("*.zip"):
            team_id = raw_zip.stem
            res = self.process_team(team_id, raw_zip, extracted_dir, bundles_dir)
            if res["is_valid"]:
                valid_count += 1
            else:
                invalid_count += 1
            results.append(res)

        return {
            "total_processed": len(results),
            "valid": valid_count,
            "invalid": invalid_count,
            "details": results
        }
