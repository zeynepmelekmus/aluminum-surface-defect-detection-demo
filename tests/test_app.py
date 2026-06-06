import unittest
from pathlib import Path
from app.detection_service import DetectionService, DetectionResult
from app.report_service import build_report_payload

class TestSurfaceDefectDetection(unittest.TestCase):
    def setUp(self):
        self.service = DetectionService(use_real_model=False)

    def test_mock_detection_structure(self):
        results = self.service.detect(None, 50.0)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        
        for item in results:
            self.assertIsInstance(item, DetectionResult)
            self.assertIsNotNone(item.defect_type)
            self.assertGreaterEqual(item.confidence, 50.0)
            self.assertGreaterEqual(item.meter, 0.0)
            self.assertIsNotNone(item.timestamp)
            self.assertEqual(len(item.bbox), 4)
            
            # Verify structure converted to dictionary
            d = item.to_dict()
            self.assertIn("defect_type", d)
            self.assertIn("meter", d)
            self.assertIn("confidence", d)
            self.assertIn("timestamp", d)
            self.assertIn("bbox", d)
            self.assertIn("source", d)

    def test_report_builder(self):
        detections = [
            {"defect_type": "Cizik", "meter": 12.5, "confidence": 85.0, "timestamp": "2026-06-06 10:00:00", "bbox": [10, 20, 30, 40], "source": "mock"}
        ]
        payload = build_report_payload("Profil-X", "ID-99", 50.0, detections)
        self.assertEqual(payload["product_name"], "Profil-X")
        self.assertEqual(payload["product_id"], "ID-99")
        self.assertEqual(payload["threshold"], 50.0)
        self.assertEqual(payload["total_defect_count"], 1)
        self.assertEqual(len(payload["defects"]), 1)

if __name__ == "__main__":
    unittest.main()
