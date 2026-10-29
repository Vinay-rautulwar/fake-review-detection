# Data Directory: Fake Review Detection v2

This directory contains data specifications, validation guidelines, and documentation for the Fake Review Detection dataset.

> **Note**: Actual dataset files (`*.csv`, `*.parquet`, `*.jsonl`) are ignored by git in `.gitignore` to prevent committing sensitive or large data files to version control.

---

## 1. Expected Dataset Schema

The training pipeline expects a CSV or tabular file containing exactly the following **17 columns**:

| Column Name | Data Type | Description | Constraints / Valid Range |
| :--- | :--- | :--- | :--- |
| `review_id` | `str` / `int` | Unique primary identifier for each review | Must be unique across all rows |
| `reviewer_id` | `str` / `int` | Unique identifier for the author/reviewer | Non-null |
| `product_id` | `str` / `int` | Unique identifier for the reviewed product | Non-null |
| `review_text` | `str` | Full textual content of the customer review | Non-empty string |
| `rating` | `float` | Star rating assigned by the reviewer | Float in range `[1.0, 5.0]` |
| `verified_purchase` | `bool` / `int` | Indicates if reviewer purchased the item | `1`/`True` or `0`/`False` |
| `review_date` | `str` | Timestamp or date of review submission | Standard date format (e.g. `YYYY-MM-DD`) |
| `product_category` | `str` | Category or department of the product | Non-null string |
| `product_avg_rating` | `float` | Cumulative average rating of the product | Float in range `[1.0, 5.0]` |
| `product_review_count`| `int` | Total lifetime reviews for the product | Integer `≥ 0` |
| `product_price` | `float` | Listed price of the product | Numeric `≥ 0.0` |
| `reviewer_review_count`| `int` | Lifetime review count by this reviewer | Integer `≥ 0` |
| `reviewer_avg_rating` | `float` | Historical average rating given by reviewer | Float in range `[1.0, 5.0]` |
| `reviewer_verified_review_count` | `int` | Number of verified purchase reviews by reviewer | Integer `0 ≤ n ≤ reviewer_review_count` |
| `reviewer_account_age_days` | `float` / `int` | Age of reviewer account in days at review date | Numeric `≥ 0` |
| `days_since_previous_review`| `float` / `int` | Days elapsed since reviewer's previous review | Numeric `≥ 0` |
| `label` | `int` | Ground truth target classification label | Binary: `0` = Genuine, `1` = Fake |

---

## 2. Directory Structure

```text
data/
├── raw/
│   └── fake_review_dataset.csv         # Original untouched dataset (30,000 rows)
├── processed/
│   └── fake_review_dataset_clean.csv   # Cleaned, standardized dataset (29,821 rows)
├── quarantine/
│   ├── invalid_rows.csv                # Quarantined rows with logical violations (179 rows)
│   ├── duplicate_reviews.csv           # Quarantined exact/normalized duplicate reviews (0 rows)
│   └── near_duplicate_reviews.csv      # Flagged template / near-duplicate reviews (3,536 rows)
├── final/
│   └── fake_review_dataset_v1.csv      # Approved quality-gated dataset for splitting (29,821 rows)
├── splits/
│   ├── train.csv                       # Training split (20,875 rows, 70.0%)
│   ├── validation.csv                  # Validation split (4,472 rows, 15.0%)
│   └── test.csv                        # Test split (4,474 rows, 15.0%)
└── README.md                           # Data specifications and dictionary
```

---

## 3. Data Cleaning Pipeline

The reproducible data cleaning pipeline can be executed via:

```bash
python training/scripts/clean_dataset.py \
    --raw_path data/raw/fake_review_dataset.csv \
    --similarity_threshold 0.90
```

### Pipeline Actions
1. **Raw Preservation**: Never modifies or overwrites the original CSV.
2. **Standardization**:
   - Standardizes target labels to `fake` and `genuine`.
   - Standardizes `verified_purchase` to binary `0` and `1`.
   - Parses dates into standard ISO `YYYY-MM-DD` strings.
   - Normalizes whitespace without rewriting review content.
3. **Quarantine Logic**:
   - Identifies and isolates invalid rows (e.g., `reviewer_verified_review_count > reviewer_review_count`).
   - Identifies exact duplicate review texts.
4. **Near-Duplicate Detection**:
   - Uses character n-gram TF-IDF similarity (n=3..4) with configurable threshold (default `0.90`).
   - Clusters template-based reviews and saves them with cluster IDs, sizes, and purity into `data/quarantine/near_duplicate_reviews.csv`.
5. **Auditing & Reporting**:
   - Generates detailed human-readable audit report at `reports/dataset_audit.txt`.
   - Generates traceable action log for all excluded rows at `reports/cleaning_log.csv`.

---

## 4. Dataset Validation

Before model training, datasets can be checked using the validation suite:

```bash
python training/scripts/validate_dataset.py --data_path data/processed/fake_review_dataset_clean.csv
```

### Checks Performed
- **Schema Presence**: Verifies presence of all 17 required columns.
- **Null / Missing Value Analysis**: Confirms 0 null cells across all columns.
- **Primary Key Uniqueness**: Checks for duplicate `review_id` values.
- **Duplicate Content**: Identifies exact duplicate review texts and normalized duplicates.
- **Rating Integrity**: Ensures ratings are numeric and within `[1.0, 5.0]`.
- **Target Label Integrity**: Confirms valid labels (`fake` and `genuine`) and calculates class balance.
- **Domain Consistency**: Validates that verified count does not exceed total count, and prices/counts/days are valid.
