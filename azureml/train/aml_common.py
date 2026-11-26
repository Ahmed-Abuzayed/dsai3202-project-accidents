"""Shared by leaderboard.py, show_run.py, model_yml.py, load_model.py and evaluate_holdout.py.

Copied from Lab-9 aml_common.py. One job: point MLflow at the workspace. The tracking URI
comes from MLFLOW_TRACKING_URI, then azureml/.tracking_uri (setup.sh writes it), then
`az ml workspace show`. Only the folder names changed.
"""
import os
import pathlib
import subprocess
import sys

WEEK = pathlib.Path(__file__).resolve().parent.parent   # azureml/


def tracking_uri() -> str:
    uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if uri:
        return uri
    f = WEEK / ".tracking_uri"
    if f.is_file() and f.read_text().strip():
        return f.read_text().strip()
    try:
        uri = subprocess.run(
            ["az", "ml", "workspace", "show", "--query", "mlflow_tracking_uri", "-o", "tsv"],
            capture_output=True, text=True, timeout=60, check=True,
        ).stdout.strip()
    except Exception as e:  # noqa: BLE001
        sys.exit(f"cannot find the workspace tracking URI ({e}). Run: bash scripts/setup.sh")
    if not uri:
        sys.exit("az ml workspace show returned no mlflow_tracking_uri. Run: bash scripts/setup.sh")
    f.write_text(uri + "\n")
    return uri


def connect():
    """Import mlflow, set the tracking URI, return (mlflow, MlflowClient())."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError:
        sys.exit("mlflow is not installed in this shell. Run: bash scripts/setup.sh")
    try:
        import azureml.mlflow  # noqa: F401  - registers the azureml:// scheme
    except ImportError:
        sys.exit("azureml-mlflow is not installed in this shell. Run: bash scripts/setup.sh")
    mlflow.set_tracking_uri(tracking_uri())
    return mlflow, MlflowClient()


def read_file(name: str) -> str:
    f = WEEK / name
    return f.read_text().strip() if f.is_file() else ""


def evidence(name: str) -> pathlib.Path:
    """azureml/evidence/<name>: the files that go into the repo."""
    d = WEEK / "evidence"
    d.mkdir(exist_ok=True)
    return d / name
