from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import hashlib
import random
from typing import Any, List, Optional

DEFECT_TYPES = [
    "Cizik",
    "Leke",
    "Catlak",
    "Delik",
    "Yuzey Dalgalanmasi",
    "Boya/Kaplama Hatası",
]


@dataclass
class DetectionResult:
    defect_type: str
    meter: float
    confidence: float
    timestamp: str
    bbox: Optional[list[float]] = None
    source: str = "mock"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaseDetectionService(ABC):
    @abstractmethod
    def detect(self, image_path: str | Path | None, threshold: float) -> List[DetectionResult]:
        """Abstract method to run defect detection."""
        pass


class YoloDetectionService(BaseDetectionService):
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model = None
        self.model_loaded = False
        self.load_error: Optional[str] = None
        self._load_model()

    def _load_model(self) -> None:
        if not self.model_path.exists():
            self.load_error = f"Model file not found: {self.model_path}"
            return
        try:
            from ultralytics import YOLO  # type: ignore
            self.model = YOLO(str(self.model_path))
            self.model_loaded = True
        except Exception as exc:
            self.load_error = str(exc)
            self.model_loaded = False

    def detect(self, image_path: str | Path | None, threshold: float) -> List[DetectionResult]:
        if not self.model_loaded or not image_path:
            # Fallback to mock if model is requested but failed to load
            mock_fallback = MockDetectionService()
            return mock_fallback.detect(image_path, threshold)
            
        try:
            results = self.model.predict(str(image_path), conf=threshold / 100.0, verbose=False)
            normalized: List[DetectionResult] = []
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for result in results:
                names = getattr(result, "names", {})
                boxes = getattr(result, "boxes", None)
                if boxes is None:
                    continue
                for box in boxes:
                    cls_id = int(box.cls[0].item()) if hasattr(box.cls[0], "item") else int(box.cls[0])
                    conf = float(box.conf[0].item()) if hasattr(box.conf[0], "item") else float(box.conf[0])
                    xyxy = box.xyxy[0].tolist() if hasattr(box.xyxy[0], "tolist") else list(box.xyxy[0])
                    normalized.append(
                        DetectionResult(
                            defect_type=str(names.get(cls_id, f"Class {cls_id}")),
                            meter=round(random.uniform(0.5, 120.0), 2),
                            confidence=round(conf * 100, 2),
                            timestamp=now,
                            bbox=[round(float(v), 2) for v in xyxy],
                            source="yolo-pytorch",
                        )
                    )
            return normalized
        except Exception as exc:
            self.load_error = f"Model inference failed, mock mode fallback: {exc}"
            mock_fallback = MockDetectionService()
            return mock_fallback.detect(image_path, threshold)


class MockDetectionService(BaseDetectionService):
    def __init__(self):
        self.model_loaded = False
        self.load_error = None

    def detect(self, image_path: str | Path | None, threshold: float) -> List[DetectionResult]:
        seed_text = str(image_path) if image_path else datetime.now().strftime("%Y%m%d%H%M")
        seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest(), 16) % (10**8)
        rng = random.Random(seed)
        count = rng.randint(3, 8)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        results: List[DetectionResult] = []
        for _ in range(count):
            confidence = round(rng.uniform(max(35, threshold), 99), 2)
            if confidence < threshold:
                continue
            x1, y1 = rng.randint(5, 180), rng.randint(5, 180)
            w, h = rng.randint(20, 120), rng.randint(20, 120)
            results.append(
                DetectionResult(
                    defect_type=rng.choice(DEFECT_TYPES),
                    meter=round(rng.uniform(0.5, 120.0), 2),
                    confidence=confidence,
                    timestamp=now,
                    bbox=[x1, y1, x1 + w, y1 + h],
                    source="mock",
                )
            )
        return sorted(results, key=lambda x: x.meter)


class DetectionService(BaseDetectionService):
    """Facade class to retain full compatibility with the existing UI/tests."""
    def __init__(self, model_path: str | Path = "models/best.pt", use_real_model: bool = True):
        self.model_path = Path(model_path)
        if use_real_model:
            self.delegate = YoloDetectionService(self.model_path)
        else:
            self.delegate = MockDetectionService()

    def detect(self, image_path: str | Path | None, threshold: float) -> List[DetectionResult]:
        return self.delegate.detect(image_path, threshold)

    @property
    def model_loaded(self) -> bool:
        return getattr(self.delegate, "model_loaded", False)

    @property
    def load_error(self) -> str | None:
        return getattr(self.delegate, "load_error", None)
