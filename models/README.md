# TRACE-X Model Weights Directory

This directory stores fine-tuned and pretrained checkpoint weights:
- `yolov8s.pt`: Pretrained COCO vehicle detector
- `plate_yolov8s.pt`: Fine-tuned Indian license plate detector
- `ocr_pp_rec.pdparams`: Fine-tuned Indian character recognition weights (PaddleOCR CTC)
- `bodytype_yolov8s_cls.pt`: Fine-tuned vehicle body type classifier

All weights adhere to the evaluation splits: evaluated ONLY on held-out clips and test sets.
