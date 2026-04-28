# TKPI–URT Data Preprocessing for Menu Recommendation System

## 1. Overview

This preprocessing pipeline prepares food composition and portion-size data for a dietary menu recommendation system. The system integrates the Indonesian Food Composition Table (TKPI) with household portion data (URT / bahan penukar) to generate a clean food candidate dataset that can be used for nutrient calculation, rule-based filtering, and menu optimization.

The final dataset is intended to support menu recommendation for dietary scenarios such as diabetes mellitus, hypertension, obesity, and combined cardiometabolic conditions.

---

## 2. Data Sources

### 2.1 TKPI Dataset

The TKPI dataset contains nutrient composition values for food items, generally expressed per 100 grams of edible portion.

Important columns include:

- `KODE`
- `NAMA BAHAN`
- `ENERGI (Kal)`
- `PROTEIN (g)`
- `LEMAK (g)`
- `KH (g)`
- `NATRIUM (mg)`
- `KALIUM (mg)`
- `KALSIUM (mg)`
- `BDD (%)`

### 2.2 URT Dataset

The URT dataset contains household measurement information and gram conversion for food portions.

Important columns include:

- `Nama Pangan`
- `Berat dalam Gram`
- `Ukuran Rumah Tangga (URT)`

Example:

| Nama Pangan | Berat dalam Gram | Ukuran Rumah Tangga (URT) |
|---|---:|---|
| Bihun | 50 | 1/2 Gelas |
| Kentang | 210 | 2 Buah Sedang |
| Telur Ayam | 55 | 1 butir |

---

## 3. Preprocessing Objectives

The preprocessing process aims to:

1. Preserve the original raw data.
2. Standardize column names.
3. Normalize food names for matching.
4. Add food category labels.
5. Merge TKPI with URT data.
6. Assign standard vegetable portion size.
7. Identify usable and unusable food items.
8. Handle missing and non-numeric values.
9. Calculate nutrient values per portion.
10. Save a final food candidate dataset for menu recommendation.

---

## 4. Folder Structure

```text
project/
├── data/
│   ├── raw/
│   │   ├── tkpi_raw.csv
│   │   └── urt_raw.csv
│   ├── processed/
│   │   ├── cleaned_tkpi_urt_dataset.csv
│   │   ├── final_food_candidates.csv
│   │   └── final_food_candidates_clinical.csv
├── outputs/
│   ├── figures/
│   ├── preprocessing_summary.csv
│   ├── missing_value_summary.csv
│   ├── unmatched_foods.csv
│   └── incomplete_macro_candidates_review.csv
├── scripts/
│   └── preprocess_tkpi_urt.py
└── README.md
