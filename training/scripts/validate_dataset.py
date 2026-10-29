"""
Dataset Validation Script for Fake Review Detection v2
======================================================
This script validates tabular and text datasets against the required schema,
data integrity constraints, and domain consistency rules.

It does NOT modify, clean, or delete problematic rows. All findings are
reported in a structured diagnostic summary.

Usage (CLI):
    python validate_dataset.py --data_path path/to/dataset.csv
    python validate_dataset.py --data_path path/to/dataset.csv --config ../configs/training_config.yaml
"""

import argparse
import os
import re
import string
import sys
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd


EXPECTED_COLUMNS: List[str] = [
    "review_id",
    "reviewer_id",
    "product_id",
    "review_text",
    "rating",
    "verified_purchase",
    "review_date",
    "product_category",
    "product_avg_rating",
    "product_review_count",
    "product_price",
    "reviewer_review_count",
    "reviewer_avg_rating",
    "reviewer_verified_review_count",
    "reviewer_account_age_days",
    "days_since_previous_review",
    "label",
]

NUMERICAL_COLUMNS: List[str] = [
    "rating",
    "product_avg_rating",
    "product_review_count",
    "product_price",
    "reviewer_review_count",
    "reviewer_avg_rating",
    "reviewer_verified_review_count",
    "reviewer_account_age_days",
    "days_since_previous_review",
]


class DatasetValidator:
    """Validates fake review dataset schema, integrity, and domain rules."""

    def __init__(self, data_path: str, expected_columns: Optional[List[str]] = None):
        self.data_path = data_path
        self.expected_columns = expected_columns or EXPECTED_COLUMNS
        self.df: Optional[pd.DataFrame] = None
        self.report: Dict[str, Any] = {
            "status": "NOT_RUN",
            "file_info": {},
            "schema_check": {},
            "missing_values": {},
            "duplicate_ids": {},
            "duplicate_reviews": {},
            "rating_check": {},
            "label_check": {},
            "class_distribution": {},
            "consistency_checks": {},
            "errors": [],
            "warnings": [],
        }

    def load_dataset(self) -> bool:
        """Check file existence and load CSV into pandas DataFrame."""
        if not self.data_path:
            self.report["errors"].append("Dataset path was not specified.")
            return False

        if not os.path.exists(self.data_path):
            self.report["errors"].append(
                f"File not found at specified path: '{self.data_path}'. "
                "Please verify the path or provide a valid CSV file."
            )
            return False

        try:
            file_size_bytes = os.path.getsize(self.data_path)
            self.df = pd.read_csv(self.data_path)
            self.report["file_info"] = {
                "path": self.data_path,
                "file_size_bytes": file_size_bytes,
                "file_size_mb": round(file_size_bytes / (1024 * 1024), 2),
                "total_rows": len(self.df),
                "total_columns": len(self.df.columns),
            }
            return True
        except Exception as e:
            self.report["errors"].append(f"Failed to read CSV file: {str(e)}")
            return False

    def validate_schema(self) -> None:
        """Validate presence of all expected columns and note extraneous columns."""
        if self.df is None:
            return

        current_cols = set(self.df.columns)
        expected_cols = set(self.expected_columns)

        missing_cols = [c for c in self.expected_columns if c not in current_cols]
        extra_cols = [c for c in self.df.columns if c not in expected_cols]

        self.report["schema_check"] = {
            "is_valid": len(missing_cols) == 0,
            "missing_columns": missing_cols,
            "extra_columns": extra_cols,
            "found_columns_count": len(current_cols),
            "expected_columns_count": len(self.expected_columns),
        }

        if missing_cols:
            self.report["errors"].append(
                f"Schema error: Missing {len(missing_cols)} required column(s): {missing_cols}"
            )
        if extra_cols:
            self.report["warnings"].append(
                f"Schema note: Found {len(extra_cols)} unexpected column(s): {extra_cols}"
            )

    def validate_missing_values(self) -> None:
        """Analyze missing (NaN/None) values per column and across records."""
        if self.df is None:
            return

        total_rows = len(self.df)
        if total_rows == 0:
            self.report["errors"].append("Dataset is empty (0 rows).")
            return

        null_counts = self.df.isnull().sum()
        columns_with_nulls = {
            col: {
                "null_count": int(count),
                "percentage": round(float(count / total_rows) * 100, 2),
            }
            for col, count in null_counts.items()
            if count > 0
        }

        rows_with_any_null = int(self.df.isnull().any(axis=1).sum())

        self.report["missing_values"] = {
            "total_null_cells": int(null_counts.sum()),
            "rows_with_nulls": rows_with_any_null,
            "rows_with_nulls_percentage": round(
                (rows_with_any_null / total_rows) * 100, 2
            ),
            "columns_with_nulls": columns_with_nulls,
        }

        if columns_with_nulls:
            for col, info in columns_with_nulls.items():
                if col in ["review_id", "review_text", "label"]:
                    self.report["errors"].append(
                        f"Critical column '{col}' has {info['null_count']} missing values ({info['percentage']}%)."
                    )
                else:
                    self.report["warnings"].append(
                        f"Column '{col}' has {info['null_count']} missing values ({info['percentage']}%)."
                    )

    def validate_duplicate_ids(self) -> None:
        """Check uniqueness of primary identifier (review_id)."""
        if self.df is None or "review_id" not in self.df.columns:
            return

        dup_mask = self.df["review_id"].duplicated(keep=False)
        duplicate_count = int(dup_mask.sum())
        unique_id_count = self.df["review_id"].nunique(dropna=False)

        sample_dups = []
        if duplicate_count > 0:
            sample_dups = (
                self.df[dup_mask]["review_id"].value_counts().head(5).to_dict()
            )

        self.report["duplicate_ids"] = {
            "total_records": len(self.df),
            "unique_ids": unique_id_count,
            "duplicate_id_rows": duplicate_count,
            "sample_duplicated_ids": sample_dups,
        }

        if duplicate_count > 0:
            self.report["errors"].append(
                f"Duplicate review_id detected: {duplicate_count} rows share non-unique IDs."
            )

    @staticmethod
    def _normalize_text(text: Any) -> str:
        """Lowercase, strip whitespace, and remove punctuation for normalized duplicate checking."""
        if pd.isna(text):
            return ""
        text_str = str(text).lower()
        text_str = re.sub(r"\s+", " ", text_str).strip()
        text_str = text_str.translate(str.maketrans("", "", string.punctuation))
        return text_str

    def validate_duplicate_reviews(self) -> None:
        """Detect exact duplicate review texts and normalized duplicate reviews."""
        if self.df is None or "review_text" not in self.df.columns:
            return

        total_rows = len(self.df)
        # 1. Exact duplicates
        exact_dup_mask = (
            self.df["review_text"].notna()
            & self.df["review_text"].duplicated(keep=False)
        )
        exact_dup_count = int(exact_dup_mask.sum())

        # 2. Normalized duplicates
        normalized_series = self.df["review_text"].apply(self._normalize_text)
        # Exclude empty strings from duplicate inflation
        non_empty_mask = normalized_series != ""
        norm_dup_mask = non_empty_mask & normalized_series.duplicated(keep=False)
        norm_dup_count = int(norm_dup_mask.sum())

        sample_exact_dups = []
        if exact_dup_count > 0:
            sample_exact_dups = (
                self.df[exact_dup_mask]["review_text"]
                .value_counts()
                .head(3)
                .to_dict()
            )

        self.report["duplicate_reviews"] = {
            "exact_duplicate_rows": exact_dup_count,
            "exact_duplicate_percentage": round(
                (exact_dup_count / total_rows) * 100, 2
            )
            if total_rows > 0
            else 0.0,
            "normalized_duplicate_rows": norm_dup_count,
            "normalized_duplicate_percentage": round(
                (norm_dup_count / total_rows) * 100, 2
            )
            if total_rows > 0
            else 0.0,
            "sample_exact_duplicates": sample_exact_dups,
        }

        if exact_dup_count > 0:
            self.report["warnings"].append(
                f"Found {exact_dup_count} rows with exact duplicate review text."
            )
        if norm_dup_count > exact_dup_count:
            diff = norm_dup_count - exact_dup_count
            self.report["warnings"].append(
                f"Found {diff} additional rows with duplicate review text after normalization (case/punctuation/spaces)."
            )

    def validate_ratings(self) -> None:
        """Verify rating values fall within valid range [1.0, 5.0] and are numeric."""
        if self.df is None or "rating" not in self.df.columns:
            return

        rating_series = pd.to_numeric(self.df["rating"], errors="coerce")
        non_numeric_count = int(rating_series.isna().sum() - self.df["rating"].isna().sum())
        out_of_bounds_mask = (rating_series < 1.0) | (rating_series > 5.0)
        out_of_bounds_count = int(out_of_bounds_mask.sum())

        value_counts = self.df["rating"].value_counts(dropna=False).to_dict()

        self.report["rating_check"] = {
            "out_of_bounds_count (not in [1.0, 5.0])": out_of_bounds_count,
            "non_numeric_values_count": non_numeric_count,
            "rating_distribution": {str(k): int(v) for k, v in value_counts.items()},
        }

        if non_numeric_count > 0:
            self.report["errors"].append(
                f"Rating column has {non_numeric_count} non-numeric value(s)."
            )
        if out_of_bounds_count > 0:
            self.report["errors"].append(
                f"Rating column has {out_of_bounds_count} value(s) outside the expected [1.0, 5.0] range."
            )

    def validate_labels(self) -> None:
        """Check that the target label contains valid binary values (0: Genuine, 1: Fake)."""
        if self.df is None or "label" not in self.df.columns:
            return

        total_rows = len(self.df)
        label_counts = self.df["label"].value_counts(dropna=False).to_dict()

        # Valid binary or standardized string labels
        valid_set = {0, 1, 0.0, 1.0, "0", "1", "fake", "genuine"}
        normalized_labels = self.df["label"].astype(str).str.lower().str.strip()
        invalid_mask = ~normalized_labels.isin(valid_set)
        invalid_count = int(invalid_mask.sum())

        distribution = {}
        for k, count in label_counts.items():
            label_key = "NaN" if pd.isna(k) else str(k)
            pct = round((count / total_rows) * 100, 2) if total_rows > 0 else 0.0
            distribution[label_key] = {"count": int(count), "percentage": pct}

        self.report["label_check"] = {
            "is_valid": invalid_count == 0,
            "invalid_labels_count": invalid_count,
            "distribution": distribution,
        }

        if invalid_count > 0:
            self.report["errors"].append(
                f"Label column has {invalid_count} row(s) with invalid labels. Expected [0, 1] or ['fake', 'genuine']."
            )

        # Check class balance
        fake_count = label_counts.get("fake", label_counts.get(1, label_counts.get(1.0, 0)))
        genuine_count = label_counts.get("genuine", label_counts.get(0, label_counts.get(0.0, 0)))

        if fake_count > 0 and genuine_count > 0 and total_rows > 0:
            ratio = min(fake_count, genuine_count) / max(fake_count, genuine_count)
            self.report["class_distribution"] = {
                "genuine_count": int(genuine_count),
                "fake_count": int(fake_count),
                "balance_ratio (min/max)": round(ratio, 4),
            }
            if ratio < 0.1:
                self.report["warnings"].append(
                    f"Severe class imbalance: minority class is only {round(ratio * 100, 2)}% of majority class."
                )
        else:
            self.report["class_distribution"] = {
                "distribution": distribution
            }

    def validate_consistency(self) -> None:
        """Validate logical domain rules between reviewer, product, and review features."""
        if self.df is None:
            return

        consistency_issues = []

        # 1. Non-negative constraints for counts, prices, and days
        # Note: days_since_previous_review allows -1 as sentinel for first review
        non_negative_cols = [
            "product_price",
            "product_review_count",
            "reviewer_review_count",
            "reviewer_verified_review_count",
            "reviewer_account_age_days",
        ]
        negative_summary = {}
        for col in non_negative_cols:
            if col in self.df.columns:
                num_series = pd.to_numeric(self.df[col], errors="coerce")
                neg_count = int((num_series < 0).sum())
                if neg_count > 0:
                    negative_summary[col] = neg_count
                    consistency_issues.append(
                        f"Column '{col}' has {neg_count} negative value(s), which is physically impossible."
                    )

        if "days_since_previous_review" in self.df.columns:
            days_series = pd.to_numeric(self.df["days_since_previous_review"], errors="coerce")
            invalid_days = int((days_series < -1).sum())
            if invalid_days > 0:
                negative_summary["days_since_previous_review"] = invalid_days
                consistency_issues.append(
                    f"Column 'days_since_previous_review' has {invalid_days} values < -1."
                )

        self.report["consistency_checks"]["negative_values"] = negative_summary

        # 2. Reviewer verified review count vs total review count
        if (
            "reviewer_verified_review_count" in self.df.columns
            and "reviewer_review_count" in self.df.columns
        ):
            v_count = pd.to_numeric(
                self.df["reviewer_verified_review_count"], errors="coerce"
            )
            t_count = pd.to_numeric(
                self.df["reviewer_review_count"], errors="coerce"
            )
            v_gt_t_mask = (v_count > t_count) & v_count.notna() & t_count.notna()
            v_gt_t_count = int(v_gt_t_mask.sum())
            self.report["consistency_checks"]["verified_gt_total_reviews"] = v_gt_t_count
            if v_gt_t_count > 0:
                consistency_issues.append(
                    f"{v_gt_t_count} row(s) have reviewer_verified_review_count > reviewer_review_count."
                )

        # 3. Product average rating bounds [1.0, 5.0]
        if "product_avg_rating" in self.df.columns:
            prod_avg = pd.to_numeric(self.df["product_avg_rating"], errors="coerce")
            invalid_prod_avg = int(((prod_avg < 1.0) | (prod_avg > 5.0)).sum())
            self.report["consistency_checks"]["invalid_product_avg_rating"] = invalid_prod_avg
            if invalid_prod_avg > 0:
                consistency_issues.append(
                    f"{invalid_prod_avg} row(s) have product_avg_rating outside [1.0, 5.0]."
                )

        # 4. Reviewer average rating bounds [1.0, 5.0]
        if "reviewer_avg_rating" in self.df.columns:
            rev_avg = pd.to_numeric(self.df["reviewer_avg_rating"], errors="coerce")
            invalid_rev_avg = int(((rev_avg < 1.0) | (rev_avg > 5.0)).sum())
            self.report["consistency_checks"]["invalid_reviewer_avg_rating"] = invalid_rev_avg
            if invalid_rev_avg > 0:
                consistency_issues.append(
                    f"{invalid_rev_avg} row(s) have reviewer_avg_rating outside [1.0, 5.0]."
                )

        # 5. Review date parsing check
        if "review_date" in self.df.columns:
            parsed_dates = pd.to_datetime(self.df["review_date"], errors="coerce")
            unparseable_dates = int(parsed_dates.isna().sum() - self.df["review_date"].isna().sum())
            self.report["consistency_checks"]["unparseable_dates"] = unparseable_dates
            if unparseable_dates > 0:
                consistency_issues.append(
                    f"{unparseable_dates} row(s) have unparseable review_date values."
                )

        for issue in consistency_issues:
            self.report["warnings"].append(f"Consistency: {issue}")

    def run_all(self) -> Dict[str, Any]:
        """Execute all validation stages and determine final status."""
        if not self.load_dataset():
            self.report["status"] = "FAILED"
            return self.report

        self.validate_schema()
        self.validate_missing_values()
        self.validate_duplicate_ids()
        self.validate_duplicate_reviews()
        self.validate_ratings()
        self.validate_labels()
        self.validate_consistency()

        if len(self.report["errors"]) > 0:
            self.report["status"] = "FAILED"
        elif len(self.report["warnings"]) > 0:
            self.report["status"] = "PASSED_WITH_WARNINGS"
        else:
            self.report["status"] = "PASSED"

        return self.report

    def print_summary(self) -> None:
        """Render a clean, human-readable terminal/notebook report."""
        rep = self.report
        print("=" * 70)
        print(" FAKE REVIEW DETECTION DATASET VALIDATION REPORT")
        print("=" * 70)
        print(f"Status: {rep['status']}")
        if "file_info" in rep and rep["file_info"]:
            fi = rep["file_info"]
            print(f"File Path    : {fi.get('path')}")
            print(f"Rows x Cols  : {fi.get('total_rows')} rows x {fi.get('total_columns')} columns")
            print(f"File Size    : {fi.get('file_size_mb')} MB ({fi.get('file_size_bytes')} bytes)")
        print("-" * 70)

        # Schema
        sc = rep.get("schema_check", {})
        print("1. Schema Check:")
        print(f"   - Valid: {sc.get('is_valid')}")
        print(f"   - Expected: {sc.get('expected_columns_count')}, Found: {sc.get('found_columns_count')}")
        if sc.get("missing_columns"):
            print(f"   - Missing Columns: {sc.get('missing_columns')}")
        if sc.get("extra_columns"):
            print(f"   - Extra Columns: {sc.get('extra_columns')}")

        # Missing values
        mv = rep.get("missing_values", {})
        print("2. Missing Values:")
        print(f"   - Rows with nulls: {mv.get('rows_with_nulls')} ({mv.get('rows_with_nulls_percentage')}%)")
        if mv.get("columns_with_nulls"):
            print("   - Null breakdown:")
            for col, stats in mv["columns_with_nulls"].items():
                print(f"     * {col}: {stats['null_count']} ({stats['percentage']}%)")
        else:
            print("   - No missing values found.")

        # Duplicates
        di = rep.get("duplicate_ids", {})
        dr = rep.get("duplicate_reviews", {})
        print("3. Duplication Check:")
        print(f"   - Duplicate review_ids: {di.get('duplicate_id_rows', 0)}")
        print(f"   - Exact duplicate review text: {dr.get('exact_duplicate_rows', 0)} ({dr.get('exact_duplicate_percentage', 0)}%)")
        print(f"   - Normalized duplicate review text: {dr.get('normalized_duplicate_rows', 0)} ({dr.get('normalized_duplicate_percentage', 0)}%)")

        # Ratings & Labels
        rc = rep.get("rating_check", {})
        lc = rep.get("label_check", {})
        print("4. Value Ranges & Target Labels:")
        print(f"   - Out-of-bounds ratings: {rc.get('out_of_bounds_count (not in [1.0, 5.0])', 0)}")
        print(f"   - Invalid labels: {lc.get('invalid_labels_count', 0)}")
        if "distribution" in lc:
            print(f"   - Label distribution: {lc['distribution']}")

        # Errors & Warnings
        print("-" * 70)
        print(f"Total Errors  : {len(rep['errors'])}")
        for err in rep["errors"]:
            print(f"  [ERROR] {err}")
        print(f"Total Warnings: {len(rep['warnings'])}")
        for warn in rep["warnings"]:
            print(f"  [WARN]  {warn}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Validate dataset schema and integrity for fake review detection."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the dataset CSV file (local or remote path).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to training_config.yaml to extract expected schema.",
    )
    args = parser.parse_args()

    validator = DatasetValidator(data_path=args.data_path)
    report = validator.run_all()
    validator.print_summary()

    if report["status"] == "FAILED":
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
