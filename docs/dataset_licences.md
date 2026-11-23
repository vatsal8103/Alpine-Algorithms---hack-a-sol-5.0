# TRACE-X: Dataset Licensing & Compliance Audit (100% Real Indian Road Data)

This document verifies the licensing and usage permissions for all training and evaluation datasets used in TRACE-X.
**Policy:** 100% Real Indian Road Data Only (Zero Synthetic Plates). Permissive open licenses only. Non-derivative (CC BY-NC-ND) sources are strictly excluded from model training to prevent copyright and licensing violations.

| Dataset Name | Primary Purpose | Source / URL | License | Permitted for Competition Use | Restrictions / Attribution Notes |
|---|---|---|---|---|---|
| **Indian License Plates with Labels (Kedar Sai)** | Plate detector (YOLOv8) & Real Indian Plate Bounding Boxes | https://www.kaggle.com/datasets/kedarsai/indian-license-plates-with-labels | **CC0: Public Domain** | **Yes** | Dedicated open-source collection of Indian vehicles with normalized YOLO bounding boxes. Recommended by Tanwar et al. |
| **Indian License Plates HR/LR Mirror** | Real Indian plate crop pool | https://huggingface.co/datasets/adityamangal98/HR-LR-images-of-the-Indian-License-plates | Open Academic Mirror | **Yes** | 10,273 high-resolution and low-resolution crops of real Indian vehicle number plates. |
| **COCO (Common Objects in Context)** | Pretrained YOLO vehicle detector (`car`, `motorcycle`, `bus`, `truck`) | https://cocodataset.org/ | CC BY 4.0 | **Yes** | Base detection weights. Attribution required. |
| **VisionPlate ANPR (IEEE DataPort)** | Multimodal Indian Vehicle, Plate Detection & OCR | https://ieee-dataport.org/documents/visionplateanpr-indian-vehicle-classification-license-plate-detection-and-ocr-dataset | CC BY 4.0 | **Yes** | Clean and noisy real Indian commercial vehicle plate sequences. Manual drop via free IEEE account. |
| **IDD (India Driving Dataset)** | Vehicle crop dataset for Indian vehicle body types (auto-rickshaw, etc.) | https://idd.insaan.iiit.ac.in/ | CC BY-NC-SA 4.0 | **Yes** | Academic research and competition evaluation permitted. Non-commercial sharing. |
| **User Real CCTV Video Clips** | Ground-truth held-out test evaluation set | `data/clips` | Competition Hackathon Dataset | **Yes** | 10 authentic surveillance clips across multiple states (DL, MH, JH, KA, CG, UP, WB). Held out strictly for testing; never trained on. |
| **fast-plate-ocr Models (ankandrew)** | Pretrained Compact Conv-Transformer (CCT-XS) ONNX OCR | https://github.com/ankandrew/fast-plate-ocr | **MIT License** | **Yes** | Open-source global plate OCR weights with native per-character confidences. |
| **fast-alpr Pipeline (ankandrew)** | License plate detector & OCR pipeline | https://github.com/ankandrew/fast-alpr | **MIT License** | **Yes** | Permissive open-source license. |
| **Synthetic Vehicle Registry** | 500-vehicle database for matching and ranking | `data/registry/vehicles.csv` | MIT / Team Original | **Yes** | 100% synthetic RTO codes and numbers to avoid real individual PII. |

---

### Authoritative Standards & Legal Sources for Indian Plate Grammar

1. **Central Motor Vehicles Rules (CMVR) 1989 (Rules 50, 51, 80, 81):**
   - *Authority:* Ministry of Road Transport and Highways (MoRTH), Government of India.
   - *URL:* https://parivahan.gov.in/ / https://morth.nic.in/central-motor-vehicles-rules-1989
   - *Scope:* Standard civilian plate structure, omission of letters 'I' and 'O' in alphabetical series blocks (Rule 51(8)), and HSRP font/dimension standards.
2. **MoRTH Gazette Notification S.O. 1332(E) (12 March 2024):**
   - *Authority:* Ministry of Road Transport and Highways / The Gazette of India.
   - *URL:* https://egazette.gov.in/
   - *Scope:* Formal amendment of State registration code for Telangana from `TS` to `TG`.
3. **MoRTH Gazette Notification G.S.R. 594(E) (26 August 2021) - Bharat (BH) Series:**
   - *Authority:* Ministry of Road Transport and Highways (MoRTH).
   - *URL:* https://morth.nic.in/sites/default/files/notifications_document/GSR%20594(E)%20dated%2026th%20August%202021.pdf
   - *Scope:* Introduction of CMVR Rule 50(8) governing the format `YY BH #### XX` for national mobility without inter-state re-registration.
4. **Delhi Motor Vehicles Rules & Transport Department GNCTD:**
   - *Authority:* Transport Department, Government of National Capital Territory of Delhi.
   - *URL:* https://transport.delhi.gov.in/
   - *Scope:* Vehicle category identification codes in RTO block (`C`=Car, `S`=Two-wheeler, `P`=Bus, `R`=Three-wheeler, `T`=Tourist, `V`=Van, `Y`=Hire, `E`=EV).
5. **Ministry of Defence, Government of India - Military Registration Conventions:**
   - *Authority:* Directorate General of Ordnance Services / Indian Armed Forces.
   - *Scope:* Broad upward arrow (`↑`) prefix and defense procurement year classification for military vehicles.

---

### Excluded / Blocked Datasets Audit

1. **Indian License Plate Dataset in the wild (Tanwar et al. - arXiv:2111.06054):**
   - *Status:* **Blocked / Not Public**. Authors withheld the dataset due to legal restrictions regarding Indian road data privacy.
2. **Synthetic Indian ANPR OCR (thundarstrom - Hugging Face):**
   - *Status:* **Blocked / Empty**. Hugging Face repository contains only README.md and zero dataset files.
3. **Kaggle Indian Vehicle Dataset (Sai Sirisha N et al.):**
   - *Status:* **Excluded from Model Training**. Licensed under **CC BY-NC-ND 4.0** (NoDerivatives). Training neural networks creates derivative model weights which cannot be distributed under CC BY-NC-ND terms.
4. **RodoSol-ALPR & UFPR-ALPR (Brazil):**
   - *Status:* **Excluded from Training**. Features Brazilian plate formats and layouts that do not match Indian vehicle registration specifications.
