# Backend Models Directory

This directory contains trained machine learning and deep learning model artifacts for the Fake Review Detection system.

## DeBERTa-v3-base Text Classifier (`deberta_fp32_final/`)

- **Architecture**: `microsoft/deberta-v3-base` (86M parameters)
- **Fine-Tuning**: Binary text classification (0 = genuine, 1 = fake) with FP32 precision.
- **Max Sequence Length**: 256 tokens

### Storage & Weight Files Policy
Due to GitHub's **100 MB** single file size limit and to prevent repository bloat, heavy model weight binaries (`model.safetensors` ~738 MB) are excluded from standard Git tracking via `.gitignore`.

The configuration and metadata files are tracked in this repository:
- `config.json` (Hugging Face model architecture configuration)
- `tokenizer_config.json` & `special_tokens_map.json`
- `validation_metrics.json` (Verified test-set evaluation metrics)

### How to Place Model Weights

1. **From Google Colab Training**:
   Run the fine-tuning notebook [`training/notebooks/train_deberta.ipynb`](../../training/notebooks/train_deberta.ipynb). When training completes, download the saved checkpoint folder or zip file.
2. **Extract into this directory**:
   Ensure `model.safetensors` is placed directly in:
   ```text
   backend/models/deberta_fp32_final/
   ├── config.json
   ├── model.safetensors          <-- Place 738 MB weights file here
   ├── tokenizer.json
   ├── tokenizer_config.json
   ├── special_tokens_map.json
   └── validation_metrics.json
   ```
3. **Using Git LFS (Optional)**:
   If your deployment environment supports Git Large File Storage (LFS):
   ```bash
   git lfs install
   git lfs track "*.safetensors"
   git add .gitattributes backend/models/deberta_fp32_final/model.safetensors
   git commit -m "Add DeBERTa FP32 weights via Git LFS"
   ```
