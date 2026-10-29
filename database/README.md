# Database & Persistence: Fake Review Detection v2

This directory contains the database schemas, migration scripts, and data models for historical tracking and review auditing.

> **Status**: Architecture scaffolded. Database schema migrations and models will be configured in subsequent development phases.

---

## Planned Schema Architecture

The persistence layer is designed for **PostgreSQL** (with **SQLite** supported for local development) and managed via **SQLAlchemy / Alembic**.

### Key Tables

1. **`reviews`**
   - Primary data store for ingested reviews and associated metadata.
   - Columns: `review_id` (PK), `reviewer_id`, `product_id`, `review_text`, `rating`, `verified_purchase`, `review_date`, `product_category`, product/reviewer metrics, `created_at`.

2. **`predictions`**
   - Historical log of inferences made by backend models.
   - Columns: `prediction_id` (PK), `review_id` (FK), `model_name`, `model_version`, `fake_probability`, `predicted_label`, `inference_latency_ms`, `timestamp`.

3. **`review_feedback`**
   - Feedback and manual reviews for active learning and model drift monitoring.
   - Columns: `feedback_id` (PK), `prediction_id` (FK), `user_label`, `notes`, `created_at`.

4. **`validation_reports`**
   - Logs of dataset validation runs and data quality diagnostics.
   - Columns: `report_id` (PK), `dataset_path`, `row_count`, `status`, `error_count`, `warning_count`, `report_json`, `created_at`.
