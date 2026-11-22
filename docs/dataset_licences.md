# TRACE-X: Dataset Licensing & Compliance Audit

This document verifies the licensing and usage permissions for all training and evaluation datasets used in TRACE-X.

| Dataset Name | Primary Purpose | Source / URL | License | Permitted for Competition Use | Restrictions / Attribution Notes |
|---|---|---|---|---|---|
| **COCO (Common Objects in Context)** | Pretrained YOLO vehicle detector (`car`, `motorcycle`, `bus`, `truck`) | https://cocodataset.org/ | CC BY 4.0 | **Yes** | Attribution required in academic and competition papers. Non-commercial and research permitted. |
| **Roboflow Indian Number Plates** | Indian license plate localization YOLO fine-tuning | Roboflow Universe (`indian-license-plates`) | CC BY 4.0 | **Yes** | Dataset contains vehicle plate annotations from Indian roads. Academic hackathon use permitted. |
| **PaddleOCR PP-OCRv4 Pretrained** | Character recognition baseline weights | PaddlePaddle GitHub repository | Apache 2.0 | **Yes** | Commercial and non-commercial open-source use explicitly permitted under Apache 2.0. |
| **IDD (India Driving Dataset)** | Vehicle crop dataset for Indian vehicle body types (auto-rickshaw, etc.) | https://idd.insaan.iiit.ac.in/ | CC BY-NC-SA 4.0 | **Yes** | Academic research and competition evaluation permitted. Non-commercial sharing. |
| **Synthetic Plate Dataset (TRACE-X)** | Graded blur (0-5) OCR training and ablation benchmark | Generated via `scripts/gen_plates.py` | MIT / Team Original | **Yes** | 100% synthetic generation with zero real PII or real individuals' registered plate data. |
| **Synthetic Vehicle Registry** | 500-vehicle database for matching and ranking | Generated via `scripts/make_registry.py` | MIT / Team Original | **Yes** | 100% synthetic RTO codes and numbers. Fully safe and compliant. |

---

### Verification Summary
- **No Private or Scraped PII**: All registry records and degraded training plates are synthetically generated.
- **License Compatibility**: All external datasets operate under CC BY 4.0, Apache 2.0, or CC BY-NC-SA 4.0, which fully allow educational and hackathon evaluation.
