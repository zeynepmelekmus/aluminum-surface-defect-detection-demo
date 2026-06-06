from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import pandas as pd


REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)


def build_report_payload(product_name: str, product_id: str, threshold: float, detections: list[dict]) -> dict:
    return {
        "product_name": product_name,
        "product_id": product_id,
        "threshold": threshold,
        "total_defect_count": len(detections),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "defects": detections,
    }


def save_json_report(payload: dict) -> Path:
    safe_id = payload.get("product_id", "product").replace("/", "-").replace(" ", "_")
    path = REPORT_DIR / f"report_{safe_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def save_csv_report(payload: dict) -> Path:
    safe_id = payload.get("product_id", "product").replace("/", "-").replace(" ", "_")
    path = REPORT_DIR / f"report_{safe_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df = pd.DataFrame(payload.get("defects", []))
    if df.empty:
        df = pd.DataFrame(columns=["defect_type", "meter", "confidence", "timestamp", "bbox", "source"])
    df.insert(0, "product_name", payload.get("product_name", ""))
    df.insert(1, "product_id", payload.get("product_id", ""))
    df.insert(2, "threshold", payload.get("threshold", ""))
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path
