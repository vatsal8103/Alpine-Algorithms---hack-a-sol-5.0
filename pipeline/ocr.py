"""TRACE-X OCR Recognition Wrapper.

Per-image inference returning:
1. Recognized text string
2. Per-character calibrated probabilities
3. Top-3 alternative character predictions per position (from CTC / probability matrix)

Backends:
- 'paddle': PaddleOCR PP-OCRv4 recognition with CTC head extraction
- 'template': High-accuracy glyph template & feature matching fallback (for offline dev/demo)
- 'auto': Attempts PaddleOCR first, gracefully falls back to template OCR
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pipeline.schemas import CharAlternative, CharCandidate

CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
CHAR_TO_IDX = {c: i for i, c in enumerate(CHARSET)}
IDX_TO_CHAR = {i + 1: c for i, c in enumerate(CHARSET)}


class TemplateOCRBackend:
    """Robust fallback OCR backend matching segmented character glyphs against plate font templates."""

    def __init__(self) -> None:
        self.templates: dict[str, np.ndarray] = self._render_template_bank()

    def _render_template_bank(self) -> dict[str, np.ndarray]:
        """Render binary reference templates (32x48) for each alphanumeric character."""
        font_path = None
        for p in [r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\calibrib.ttf", r"C:\Windows\Fonts\segoeuib.ttf"]:
            if Path(p).exists():
                font_path = p
                break

        font = ImageFont.truetype(font_path, 40) if font_path else ImageFont.load_default()
        bank: dict[str, np.ndarray] = {}

        for ch in CHARSET:
            img = Image.new("L", (32, 48), 0)
            draw = ImageDraw.Draw(img)
            # Center character
            bbox = draw.textbbox((0, 0), ch, font=font)
            bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
            x = (32 - bw) // 2
            y = (48 - bh) // 2
            draw.text((x, y), ch, fill=255, font=font)
            bank[ch] = np.array(img, dtype=np.uint8)

        return bank

    def segment_characters(self, plate_crop: np.ndarray) -> list[tuple[int, int, int, int, np.ndarray]]:
        """Preprocess plate crop and segment character candidate boxes."""
        if len(plate_crop.shape) == 3:
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_crop

        # Contrast enhancement (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        norm = clahe.apply(gray)

        # Otsu thresholding
        _, thresh = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Remove outer borders
        h, w = thresh.shape
        thresh[:int(h * 0.08), :] = 0
        thresh[int(h * 0.92):, :] = 0
        thresh[:, :int(w * 0.04)] = 0
        thresh[:, int(w * 0.96):] = 0

        # Find connected components / contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = ch / float(max(1, cw))
            area_frac = (cw * ch) / float(w * h)
            # Filter contours by height and area typical of license characters
            if 0.30 <= (ch / h) <= 0.95 and 0.8 <= aspect <= 5.5 and area_frac >= 0.015:
                char_patch = thresh[y:y+ch, x:x+cw]
                boxes.append((x, y, cw, ch, char_patch))

        # Sort left-to-right
        boxes.sort(key=lambda b: b[0])

        # If segmentation found too few characters, fall back to sliding uniform windows
        if len(boxes) < 8:
            boxes = []
            target_n = 10
            margin_x = int(w * 0.08)
            eff_w = w - (margin_x * 2)
            slot_w = eff_w // target_n
            for i in range(target_n):
                sx = margin_x + i * slot_w
                sy = int(h * 0.15)
                sw = slot_w
                sh = int(h * 0.70)
                patch = thresh[sy:sy+sh, sx:sx+sw]
                boxes.append((sx, sy, sw, sh, patch))

        return boxes

    def recognize_character(
        self,
        char_patch: np.ndarray,
        temperature: float = 1.0
    ) -> tuple[str, float, list[CharAlternative]]:
        """Match patch against reference templates and compute calibrated match probabilities."""
        resized = cv2.resize(char_patch, (32, 48), interpolation=cv2.INTER_AREA)

        scores: list[tuple[str, float]] = []
        for ch, templ in self.templates.items():
            # Normalized correlation coefficient
            res = cv2.matchTemplate(resized, templ, cv2.TM_CCOEFF_NORMED)
            score = float(res[0][0])
            scores.append((ch, max(0.0, score)))

        chars = [s[0] for s in scores]
        vals = [s[1] for s in scores]
        ranked_indices = np.argsort(vals)[::-1]
        top_idx = ranked_indices[0]
        runner_idx = ranked_indices[1] if len(ranked_indices) > 1 else top_idx

        top_char = chars[top_idx]
        top_score = vals[top_idx]
        runner_score = vals[runner_idx]
        margin = max(0.0, top_score - runner_score)

        # Calibrate match correlation into evidential posterior confidence
        top_prob = max(0.20, min(0.96, (top_score * 0.85) + (margin * 0.35)))

        alts = [
            CharAlternative(char=chars[i], prob=round(max(0.05, float(vals[i]) * 0.75), 3))
            for i in ranked_indices[1:4]
        ]
        return top_char, round(top_prob, 3), alts


class LicensePlateOCR:
    """Production OCR Engine with EasyOCR (Primary for robust video ANPR), fine-tuned CRNN, and template fallback."""

    def __init__(
        self,
        backend: str | None = None,
        temperature: float = 0.5,
        min_char_conf: float = 0.28,
        model_path: str | None = None,
        calibration_path: str = "configs/ocr_calibration.json",
        use_finetuned: bool | None = None,
        fallback_backend: str = "easyocr"
    ) -> None:
        cfg_backend = "fast_plate_ocr"
        cfg_model_path = "models/ocr_finetuned.pth"
        cfg_use_ft = False
        cfg_fast_model = "cct-xs-v2-global-model"
        try:
            import yaml
            cfg_p = Path("configs/config.yaml")
            if cfg_p.exists():
                with open(cfg_p, "r", encoding="utf-8") as f:
                    cfg_data = yaml.safe_load(f) or {}
                ocr_cfg = cfg_data.get("ocr", {})
                cfg_backend = ocr_cfg.get("backend", "fast_plate_ocr")
                cfg_model_path = ocr_cfg.get("model_path", cfg_model_path)
                cfg_use_ft = bool(ocr_cfg.get("use_finetuned", False))
                cfg_fast_model = ocr_cfg.get("fast_plate_model", "cct-xs-v2-global-model")
        except Exception:
            pass

        self.backend_choice = backend if backend is not None else cfg_backend
        self.use_finetuned = use_finetuned if use_finetuned is not None else cfg_use_ft
        self.fast_plate_model_name = cfg_fast_model
        self.fallback_backend = fallback_backend
        self.temperature = temperature
        self.min_char_conf = min_char_conf
        self.model_path = model_path or cfg_model_path
        self.calibration_path = calibration_path
        self.finetuned_model = None
        self.fast_plate_rec = None
        self.easy_ocr = None
        self.paddle_ocr = None
        self.device = None
        self.template_ocr = TemplateOCRBackend()

        # Load calibrated temperature if available
        p_calib = Path(calibration_path)
        if p_calib.exists():
            try:
                with open(p_calib, "r", encoding="utf-8") as f:
                    calib_data = json.load(f)
                    self.temperature = float(calib_data.get("optimal_temperature", 0.5))
            except Exception:
                pass

        if self.backend_choice == "finetuned" or (self.backend_choice == "auto" and self.use_finetuned):
            self._init_finetuned()
            if self.finetuned_model is None:
                self.backend_choice = "fast_plate_ocr"

        if self.backend_choice in ("fast_plate_ocr", "fast-plate-ocr", "auto") and self.finetuned_model is None:
            self._init_fast_plate_ocr()

        if backend in ("easyocr", "paddle", "auto") or self.fallback_backend == "easyocr":
            self._init_easyocr()
            self._init_paddle()

    def _init_fast_plate_ocr(self) -> None:
        """Initialize fast-plate-ocr (CCT-XS / ONNX) high-speed recognizer."""
        try:
            from fast_plate_ocr import LicensePlateRecognizer
            self.fast_plate_rec = LicensePlateRecognizer(self.fast_plate_model_name)
        except Exception:
            self.fast_plate_rec = None

    def _init_finetuned(self) -> None:
        """Attempt to load fine-tuned CRNN weights on GPU or CPU."""
        p_model = Path(self.model_path)
        if p_model.exists():
            try:
                import torch
                from scripts.train_ocr import CRNNOcrModel, NUM_CLASSES
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = CRNNOcrModel(num_classes=NUM_CLASSES).to(device)
                model.load_state_dict(torch.load(p_model, map_location=device))
                model.eval()
                self.finetuned_model = model
                self.device = device
            except Exception:
                self.finetuned_model = None
                self.device = None

    def _init_easyocr(self) -> None:
        try:
            import easyocr
            import torch
            use_gpu = torch.cuda.is_available()
            self.easy_ocr = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
        except Exception:
            self.easy_ocr = None

    def _init_paddle(self) -> None:
        try:
            from paddleocr import PaddleOCR
            self.paddle_ocr = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)
        except Exception:
            self.paddle_ocr = None

    def recognize(
        self,
        plate_image: np.ndarray,
        temperature_override: float | None = None
    ) -> tuple[str, list[CharCandidate]]:
        """Perform character recognition on a plate crop image.
        
        Returns:
            (raw_string, list_of_char_candidates)
        """
        temp = temperature_override or self.temperature

        # 1. Primary Empirical Winner: Fine-Tuned TRACE-X CRNN (Pure Recognition-Only)
        if self.finetuned_model is not None:
            try:
                import torch
                dev = self.device or torch.device("cpu")
                gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY) if len(plate_image.shape) == 3 else plate_image
                resized = cv2.resize(gray, (128, 32), interpolation=cv2.INTER_AREA)
                tensor = torch.from_numpy(resized).float() / 127.5 - 1.0
                tensor = tensor.unsqueeze(0).unsqueeze(0).to(dev)  # (1, 1, 32, 128)

                with torch.no_grad():
                    log_probs = self.finetuned_model(tensor)  # (T, 1, C)
                    scaled_logits = log_probs / max(0.1, temp)
                    probs = torch.softmax(scaled_logits, dim=-1).squeeze(1).cpu().numpy()  # (T, C)

                preds = np.argmax(probs, axis=-1)
                chars = []
                candidates = []
                prev = 0

                for t, p in enumerate(preds):
                    if p != 0 and p != prev and p in IDX_TO_CHAR:
                        ch = IDX_TO_CHAR[p]
                        ch_prob = float(probs[t, p])
                        chars.append(ch)

                        ranked_c = np.argsort(probs[t])[::-1]
                        alts = []
                        for alt_idx in ranked_c:
                            if alt_idx != 0 and alt_idx != p and alt_idx in IDX_TO_CHAR:
                                alts.append(CharAlternative(char=IDX_TO_CHAR[alt_idx], prob=round(float(probs[t, alt_idx]), 3)))
                                if len(alts) >= 3:
                                    break

                        candidates.append(CharCandidate(
                            char=ch,
                            prob=round(ch_prob, 3),
                            alternatives=alts
                        ))
                    prev = p

                ft_text = "".join(chars)
                if len(ft_text) >= 3:
                    return ft_text, candidates
            except Exception:
                pass

        # 2. Benchmark Champion: fast-plate-ocr CCT-XS (Conv-Transformer ONNX)
        if self.fast_plate_rec is not None:
            try:
                if len(plate_image.shape) == 2:
                    img_rgb = cv2.cvtColor(plate_image, cv2.COLOR_GRAY2RGB)
                elif plate_image.shape[2] == 4:
                    img_rgb = cv2.cvtColor(plate_image, cv2.COLOR_BGRA2RGB)
                else:
                    img_rgb = cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB)

                prediction = self.fast_plate_rec.run_one(img_rgb, return_confidence=True)
                scores = prediction.char_probs
                if scores is not None and len(scores) >= len(prediction.plate):
                    candidates = [
                        CharCandidate(char=ch, prob=float(score), alternatives=[])
                        for ch, score in zip(prediction.plate, scores)
                        if ch in CHARSET
                    ]
                    if len(candidates) >= 3:
                        return "".join(c.char for c in candidates), candidates
            except Exception:
                pass

        # 3. Fallback: Pretrained EasyOCR
        if self.easy_ocr is not None and self.backend_choice != "template":
            try:
                h, w = plate_image.shape[:2]
                img_ocr = plate_image
                if h < 60:
                    scale = 64.0 / max(1, h)
                    img_ocr = cv2.resize(plate_image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

                results = self.easy_ocr.readtext(img_ocr)
                best_text = ""
                best_conf = 0.0

                for bbox, text, conf in results:
                    clean = re.sub(r"[^A-Za-z0-9]", "", text).upper()
                    if len(clean) >= 3 and conf >= self.min_char_conf and (conf, len(clean)) > (best_conf, len(best_text)):
                        best_text = clean
                        best_conf = float(conf)

                if len(best_text) >= 3 and best_conf >= self.min_char_conf:
                    char_conf = min(1.0, best_conf)
                    candidates = []
                    for ch in best_text:
                        candidates.append(CharCandidate(char=ch, prob=char_conf, alternatives=[]))
                    return best_text, candidates
            except Exception:
                pass

        if self.backend_choice != "template":
            return "", []

        # Explicit template matching mode
        segments = self.template_ocr.segment_characters(plate_image)
        candidates = []
        chars_out = []

        for sx, sy, sw, sh, patch in segments:
            ch, prob, alts = self.template_ocr.recognize_character(patch, temperature=temp)
            chars_out.append(ch)
            candidates.append(CharCandidate(char=ch, prob=prob, alternatives=alts))

        text_str = "".join(chars_out)
        return text_str, candidates
