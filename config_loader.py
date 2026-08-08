# Source: created in repo, no external source
"""Central config loader for the repository.

All scripts resolve config.yaml relative to the repository root so they can be
run from any working directory.
"""
from pathlib import Path
import yaml

_REPO_ROOT = Path(__file__).resolve().parent


def load_config() -> dict:
    """Load the main config.yaml from the repository root."""
    config_path = _REPO_ROOT / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def repo_root() -> Path:
    """Return the repository root path."""
    return _REPO_ROOT


def resolve_path(relative_path: str) -> Path:
    """Resolve a path string relative to the repository root."""
    return _REPO_ROOT / relative_path
