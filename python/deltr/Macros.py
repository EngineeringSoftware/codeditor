import os
from pathlib import Path


class Macros:
    this_dir: Path = Path(os.path.dirname(os.path.realpath(__file__)))
    python_dir: Path = this_dir.parent
    project_dir: Path = python_dir.parent
    model_dir: Path = project_dir / "models"
    data_dir: Path = project_dir / "data"
    raw_data_dir: Path = project_dir / "raw_data"
    script_dir: Path = project_dir / "scripts"
    tacc_log_dir: Path = project_dir / "tacc-logs"
    results_dir: Path = project_dir / "results"
    model_results_dir: Path = results_dir / "model-results"
    log_file: Path = python_dir / "experiments.log"
    paper_dir: Path = project_dir / "papers" / "paper"
    config_dir: Path = python_dir / "configs"
    doc_dir: Path = project_dir / "docs"
    gleu_dir: Path = this_dir / "gleu"

    downloads_dir: Path = project_dir / "_downloads"
    repos_downloads_dir: Path = downloads_dir / "repos"
    repos_results_dir: Path = downloads_dir / "repos_results"
    collector_dir: Path = project_dir / "collector"
    collector_version = "1.0-SNAPSHOT"

    train: str = "train"
    valid: str = "valid"
    test: str = "test"
