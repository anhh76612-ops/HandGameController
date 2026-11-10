"""Đọc config.yaml và biến môi trường (.env)."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    return cfg


def resolve(path: str) -> Path:
    """Đường dẫn tương đối trong config được tính từ thư mục gốc dự án."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
