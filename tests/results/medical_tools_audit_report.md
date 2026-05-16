# VitalAI Medical Tools Audit Report

- Generated at: `2026-05-16T04:52:40+00:00`
- Total cases: `19`
- Passed: `19`
- Failed: `0`

## [PASS] formula_ckd_epi_2021 - CKD-EPI 2021 computes race-free eGFR

- Query: `Nữ 60 tuổi, creatinine 1.4 mg/dL. Hãy ước tính eGFR.`
- disease_name: `None`
- formula_ids: `['ckd_epi_2021_creatinine']`
- Checks:
  - `OK` formula=ckd_epi_2021_creatinine, actual=computed, expected=computed
  - `OK` formula=ckd_epi_2021_creatinine, actual=43.0698, expected=43.0698±0.0001
  - `OK` actual=['CKD stage III', 'G3b'], expected_contains=G3b

## [PASS] formula_mdrd_default_race - MDRD computes with default race assumption

- Query: `Nữ 60 tuổi, creatinine 1.4 mg/dL. Hãy ước tính eGFR.`
- disease_name: `None`
- formula_ids: `['mdrd_gfr']`
- Checks:
  - `OK` formula=mdrd_gfr, actual=computed, expected=computed
  - `OK` formula=mdrd_gfr, actual=40.7681, expected=40.7681±0.0001
  - `OK` actual=['CKD stage III', 'G3b'], expected_contains=G3b

## [PASS] formula_cockcroft_gault - Cockcroft-Gault computes expected creatinine clearance

- Query: `Nam 65 tuổi, nặng 70 kg, creatinine 1.6 mg/dL. Tính Cockcroft-Gault.`
- disease_name: `None`
- formula_ids: `['cockcroft_gault']`
- Checks:
  - `OK` formula=cockcroft_gault, actual=computed, expected=computed
  - `OK` formula=cockcroft_gault, actual=45.5729, expected=45.5729±0.0001

## [PASS] formula_bsa - Body surface area computes expected result

- Query: `Cân nặng 55 kg, chiều cao 160 cm. Tính BSA.`
- disease_name: `None`
- formula_ids: `['body_surface_area']`
- Checks:
  - `OK` formula=body_surface_area, actual=computed, expected=computed
  - `OK` formula=body_surface_area, actual=1.5635, expected=1.5635±0.0001

## [PASS] formula_fena_vietnamese - FENa computes from Vietnamese aliases

- Query: `Natri niệu 20 mmol/L, natri máu 140 mmol/L, creatinine niệu 100 mg/dL, creatinine máu 1 mg/dL. Tính FENa.`
- disease_name: `acute_kidney_injury`
- formula_ids: `['fena_formula']`
- Checks:
  - `OK` formula=fena_formula, actual=computed, expected=computed
  - `OK` formula=fena_formula, actual=0.1429, expected=0.1429±0.0001
  - `OK` actual=['prerenal_aki_suggestive'], expected_contains=prerenal_aki_suggestive

## [PASS] formula_fena_english_words - FENa computes from natural English wording

- Query: `Urine Na 20 mmol/L, plasma Na 140 mmol/L, urine creatinine 100 mg/dL, plasma creatinine 1 mg/dL. Tính FENa.`
- disease_name: `acute_kidney_injury`
- formula_ids: `['fena_formula']`
- Checks:
  - `OK` formula=fena_formula, actual=computed, expected=computed
  - `OK` formula=fena_formula, actual=0.1429, expected=0.1429±0.0001
  - `OK` actual=['creatinine'], expected_missing=sodium

## [PASS] acr_a1 - ACR below 30 maps to A1

- Query: `ACR 29 mg/g`
- disease_name: `None`
- formula_ids: `[]`
- Checks:
  - `OK` actual=['A1'], expected_contains=A1

## [PASS] acr_a2 - ACR from 30 to 299 maps to A2

- Query: `ACR 299 mg/g`
- disease_name: `None`
- formula_ids: `[]`
- Checks:
  - `OK` actual=['A2'], expected_contains=A2

## [PASS] acr_300_boundary - ACR 300 mg/g maps to A3

- Query: `ACR 300 mg/g`
- disease_name: `None`
- formula_ids: `[]`
- Checks:
  - `OK` actual=['A3'], expected_contains=A3

## [PASS] acr_ckd_context - ACR classification survives CKD disease filter

- Query: `ACR 350 mg/g`
- disease_name: `benh_than_man`
- formula_ids: `[]`
- Checks:
  - `OK` actual=['A3'], expected_contains=A3

## [PASS] gfr_g2 - GFR 75 maps to G2

- Query: `GFR 75 ml/ph/1.73m2`
- disease_name: `benh_than_man`
- formula_ids: `[]`
- Checks:
  - `OK` actual=['CKD stage II', 'G2'], expected_contains=G2

## [PASS] gfr_g3a - GFR 55 maps to G3a

- Query: `GFR 55 ml/ph/1.73m2`
- disease_name: `benh_than_man`
- formula_ids: `[]`
- Checks:
  - `OK` actual=['CKD stage III', 'G3a'], expected_contains=G3a

## [PASS] gfr_g5 - GFR 10 maps to G5

- Query: `GFR 10 ml/ph/1.73m2`
- disease_name: `benh_than_man`
- formula_ids: `[]`
- Checks:
  - `OK` actual=['CKD stage V', 'G5'], expected_contains=G5

## [PASS] proteinuria_threshold - Nephrotic-range proteinuria threshold matches

- Query: `Protein niệu 24h 4 g/24h`
- disease_name: `hoi_chung_than_hu`
- formula_ids: `[]`
- Checks:
  - `OK` biomarker=protein_niệu_24h, label=None, matched_count=1

## [PASS] albumin_unit_conversion - Albumin converts g/dL to g/L before threshold compare

- Query: `Albumin máu 2.8 g/dL`
- disease_name: `hoi_chung_than_hu`
- formula_ids: `[]`
- Checks:
  - `OK` biomarker=albumin_máu, label=None, matched_count=1

## [PASS] blood_pressure_parser - Blood pressure parser extracts systolic and diastolic values

- Query: `Huyết áp 128/78 mmHg`
- disease_name: `benh_than_man`
- formula_ids: `[]`
- Checks:
  - `OK` actual=['systolic_bp', 'diastolic_bp'], expected_contains=systolic_bp
  - `OK` actual=['systolic_bp', 'diastolic_bp'], expected_contains=diastolic_bp
  - `OK` biomarker=systolic_bp, label=blood_pressure_target, matched_count=1
  - `OK` biomarker=diastolic_bp, label=blood_pressure_target, matched_count=1

## [PASS] hyperkalemia_boundary - Potassium 6.5 reaches severe hyperkalemia threshold

- Query: `Kali 6.5 mmol/L`
- disease_name: `acute_kidney_injury`
- formula_ids: `[]`
- Checks:
  - `OK` actual=['severe_hyperkalemia'], expected_contains=severe_hyperkalemia

## [PASS] hemoglobin_male_who - Male Hb 12 g/dL is recognized as below WHO threshold

- Query: `Nam, Hb 12 g/dL`
- disease_name: `benh_than_man`
- formula_ids: `[]`
- Checks:
  - `OK` actual=['anemia_threshold'], expected_contains=anemia_threshold

## [PASS] hemoglobin_female_boundary - Female Hb 12 g/dL does not cross female anemia threshold

- Query: `Nữ, Hb 12 g/dL`
- disease_name: `benh_than_man`
- formula_ids: `[]`
- Checks:
  - `OK` actual=[], expected_missing=anemia_threshold_female
