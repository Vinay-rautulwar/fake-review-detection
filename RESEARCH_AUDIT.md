# FakeGuard research audit and next steps

**Audit date:** 29 October 2026  
**Scope:** Review of local data files, model/training/evaluation scripts, explainability utilities, and saved results; reproduction of the synthetic baseline; and new five-fold text-only experiments on the local hotel-review corpus using four TF--IDF models and a BiLSTM.

## Conclusion

Yes, substantial research remains before this can be presented as an evaluated multi-view deep-learning framework. In addition to reproducing the earlier synthetic baseline, this audit completed a leakage-aware text-only study on the local deceptive-opinion corpus. In hotel-disjoint five-fold evaluation, the linear SVM achieved the highest mean macro-F1 (0.8859, fold SD 0.0131); the corrected BiLSTM achieved 0.7526 (0.0367). These are controlled-corpus text results, not platform fraud results. There is still no valid evaluation of the complete text + behavior + graph model, and no verified claim that it beats prior methods.

The research direction is still viable if it is narrowed to a testable question: **does view-aware fusion improve fake-review detection under label-aware, leakage-controlled, cross-domain or held-out-generator evaluation?** Combining text, metadata, and relations is already established in prior work, so the modality list alone is not a novelty claim.

## What is accurate, and what is not

| Project item | Audit finding |
| --- | --- |
| **Baseline metrics** | `results/baseline_results.csv` is reproducible to rounding precision from the local 2,000-row development data using the classifiers and stratified 80/20 split in `notebooks/02_baseline_models.py`. The holdout has 400 rows. This confirms the calculation path, not real-world detection performance. |
| **Synthetic data** | `data/download_data.py` generates 2,000 records and chooses the label before generating rating and text length. Fake-labeled examples receive only 1- or 5-star ratings and 10–50 words; genuine-labeled examples use a broader rating distribution and 20–150 words. These artificial correlations make it unsuitable for research performance claims. |
| **Baseline feature validity** | `notebooks/02_baseline_models.py` loads `data/processed/dataset_with_features.csv`. User/product aggregates in that file are computed before the random split, so test-period and other held-out records contribute to features. Rows can also share users and products across train/test. Treat all saved metrics as development-only. |
| **YelpChi statistics** | The canonical dataset statistics cited by Rayana and Akoglu are 67,395 reviews, 38,063 users, and 201 businesses. The review labels are Yelp-filtered versus recommended, not independently verified fake versus genuine ground truth. The repository has no full YelpChi file; `new_data_test.csv` has 2,483 rows and undocumented provenance. |
| **Ott corpus** | The local `deceptive-opinion.csv` has 1,600 records, split 800/800 truthful/deceptive and 400 in each polarity-by-label group. Its structure matches the combined positive corpus (Ott et al., 2011) and negative extension (Ott et al., 2013); local archive provenance was not independently verified. It contains no reviewer histories, timestamps, or graph. |
| **New text-only experiments** | Four exact normalized-text duplicates were removed, leaving 1,596 rows from 20 hotels. Five-fold hotel-disjoint evaluation gave macro-F1 0.8859 (SD 0.0131) for the linear SVM and 0.7526 (SD 0.0367) for the corrected, packed-sequence BiLSTM. This supports only text-based classification within this controlled corpus. Scripts, fold metrics, split manifests, and data/software metadata are saved under `scripts/run_ott_experiments.py`, `scripts/run_ott_bilstm.py`, and `results/ott_experiments/`. |
| **40K corpus** | The local file has 40,432 rows, split 20,216/20,216 across `CG` and `OR`. The primary source and construction process are not documented in the repository. These labels distinguish generation source as represented in the file; they are not automatically labels of malicious deception or verified authenticity. |
| **Dataset adapter features** | `scripts/train_real_datasets.py` creates label-conditional synthetic behavioral columns for the Ott and 40K corpora. In the Ott adapter, the `rating` is also simulated from polarity. Any model using these columns can exploit artifacts of the adapter rather than behavior. They must be removed from evaluation. |
| **Graph model** | A GAT module exists, but `scripts/train.py` never constructs or passes graph tensors. `FakeGuard.forward` therefore supplies a zero graph embedding. The current training runner does not train or evaluate the advertised graph view. |
| **Fusion naming** | `src/models/fusion.py` uses PyTorch `MultiheadAttention` with the same projected view sequence as query, key, and value. This is self-attention across view tokens, not a separate cross-attention design. The module's formulation can be studied, but it should not be called novel without an explicit comparison to prior fusion approaches. |
| **Explainability** | `src/evaluation/explainability.py` uses random token weights and hard-coded cross-view matrices/importance values in its plotting methods. No plotted result from those functions is an explanation of a trained model. A SHAP wrapper exists but there is no verified SHAP output tied to a valid experiment. Attention weights alone also do not establish faithful explanations. |
| **Evaluation/figures** | `scripts/evaluate.py` generates random labels and scores without loading the supplied checkpoint. `scripts/generate_figures.py` creates synthetic scores and hard-coded attention values. Those outputs must not be used in the paper. |
| **Previous paper draft** | The previous draft claimed state-of-the-art performance, a novel cross-attention layer, completed ablations, and explanation findings while its result tables contained placeholders and the local pipeline did not support those claims. It also used `GCAN` for the review-spam paper by Li et al.; that paper names its method GAS and was published at CIKM 2019, not 2020. The current `paper/main.tex` is now a university-format project report and separates completed text-only results from untested multimodal claims. |

The report cover still has fields for the student, institution, guide, and academic year because those details were not present in the project files. The report bibliography was reduced to the sources it cites and checked against publisher or proceedings records. The report is being compiled in the existing open LaTeX editor after its final source update.

### Baseline reproduction note

The independent rerun used the current `dataset_with_features.csv`, the same stratified split and model parameters documented in the notebook, and recalculated the metrics stored in the CSV. All saved values agree to four decimal places (including rounding behavior for Naive Bayes macro precision). The key fact is the input provenance: the result table is a synthetic-data baseline, not an experiment on YelpChi, Amazon, or the Ott corpus.

For context, Combined LR is the best saved model by macro-F1 (0.7293) and ROC-AUC (0.8990), with fake-class F1 0.5838 and AP 0.5810. Linear SVM has the highest accuracy (0.8500) but a much lower fake-class F1 (0.3182); Naive Bayes predicts no fake examples at the default threshold (fake-class F1 0.0000). The held-out set has 63 fake rows out of 400 (15.75%). These are exact descriptions of the synthetic run, not real-world performance estimates.

## Additional research findings

1. **Multiview review fraud detection predates FakeGuard.** Rayana and Akoglu's SPEAGLE combines review metadata (text, timestamp, rating) with reviewer--review--product relations. Their Yelp labels mean filtered/spam versus recommended/non-spam, so later work should describe them as proxy labels. [Original KDD paper](https://doi.org/10.1145/2783258.2783370)
2. **The existing datasets answer different questions.** Ott et al.'s 1,600-review corpus concerns crowdsourced deceptive hotel text. `CG`/`OR` data concern machine-generated versus other text if the data source is verified. Yelp filtered/recommended labels concern platform filtering. Report separate datasets, label definitions, and claims. [Ott et al., ACL 2011](https://aclanthology.org/P11-1032/)
3. **LLM-generated fake reviews are an active direction.** Liyanage et al. (ECNLP 2024) generated hotel-review variants from the Ott corpus using several language-model and paraphrase methods. Gambetti and Han (ICWSM 2024) introduced AiGen-FoodReview with 20,144 restaurant review/image pairs, including machine-generated examples. These support an experiment on held-out generators or domains, but their task constructions do not produce directly comparable scores. [Liyanage et al.](https://aclanthology.org/2024.ecnlp-1.8/), [Gambetti & Han](https://doi.org/10.1609/icwsm.v18i1.31437)
4. **Attention should not be advertised as validated explanation by itself.** Jain and Wallace found that attention weights are not a fail-safe explanation of model outputs. Explanation claims need independent faithfulness and stability checks. [Jain & Wallace, NAACL 2019](https://aclanthology.org/N19-1357/)

## Recommended research plan

### 1. Establish data provenance and task boundaries

- Keep deception, machine-generated text, and Yelp filter-proxy detection as separate tasks.
- Record the primary source, version/hash, licensing terms, collection process, label definition, counts, and exclusions for each corpus.
- Treat `new_data_test.csv` as unverified until its source and sampling are recovered. Obtain and document the canonical YelpChi release if graph experiments are intended.
- Use Ott for text-only deception experiments. Use the 40K file only after confirming its primary source and whether `CG`/`OR` are truly generation-source labels.

### 2. Remove shortcuts and split leakage

- Remove adapter-generated label-conditional behavioral variables. Do not convert corpus source or polarity into fake reviewer IDs or behavioral features.
- Split before fitting scalers, imputers, feature selectors, thresholds, and aggregate statistics.
- For behavior/graph claims, use metadata that exists in the dataset and would be available before the review decision. Construct graph snapshots from past data only.
- Deduplicate exact and near-duplicate texts before splitting. Add temporal and reviewer/product-disjoint evaluation, plus a conventional random split only as a clearly labeled reference.

### 3. Make the comparison answer the research question

On each eligible dataset, compare TF--IDF + logistic regression, a pretrained text encoder, metadata-only, graph-only, simple concatenation/late fusion, and the proposed view-attention model. Include view ablations and missing-view tests. Keep hyperparameter tuning and threshold selection on validation data; use the final test set once.

Report fake-class precision/recall/F1, macro-F1, ROC-AUC, average precision (AP), balanced accuracy, calibration, and confidence intervals or variation across seeds. Report false-positive behavior because false flags can penalize legitimate reviewers. The current `auc_pr` field is calculated with `average_precision_score`; call it AP rather than an unspecified PR-curve area.

### 4. Validate any explanation claim

Replace random/hard-coded plots with explanations from a saved model and example inputs. Compare feature attributions with perturbation tests, test ranking stability across seeds, and measure prediction changes after removing purportedly important evidence. Report explanations as model diagnostics unless these tests support stronger language.

## Corrected references added to the paper

- Rayana, S. & Akoglu, L. (2015). *Collective Opinion Spam Detection: Bridging Review Networks and Metadata*. KDD. DOI: `10.1145/2783258.2783370`.
- Liyanage, V., Buscaldi, D. & Forcioli, P. (2024). *Detecting AI-enhanced Opinion Spambots: a study on LLM-generated Hotel Reviews*. ECNLP @ LREC-COLING, pp. 74–78.
- Gambetti, A. & Han, Q. (2024). *AiGen-FoodReview: A Multimodal Dataset of Machine-Generated Restaurant Reviews and Images on Social Media*. ICWSM 18(1), pp. 1935–1945. DOI: `10.1609/icwsm.v18i1.31437`.
- Jain, S. & Wallace, B. C. (2019). *Attention is not Explanation*. NAACL-HLT, pp. 3543–3556. DOI: `10.18653/v1/N19-1357`.

## Collaborator implementation added (1 November 2026)

Selected code and reports from Vinay Rautulwar's public repository, [`fake-review-detection`](https://github.com/Vinay-rautulwar/fake-review-detection), were integrated from commit `92c10b3` into `backend/`, `frontend/`, `training/`, and `reports/`. The additions include a 17-column data validation and cleaning pipeline, reviewer/near-duplicate constrained split code, a TF-IDF + LightGBM baseline, a DeBERTa-v3 text-only training notebook, and an API/dashboard prototype. These are separate from the local Ott benchmark and do not implement or validate FakeGuard's text + behavior + graph fusion.

The collaborator reports a 30,000-row template-generated dataset, 29,821 rows after excluding 179 invalid records, and a 70/15/15 reviewer-held-out split with near-duplicate clusters kept within splits. The split report also notes that product-level aggregates overlap across partitions and may contain future-data information. No dataset or split CSVs were included in the source repository, so those counts and split diagnostics are preserved as collaborator reports rather than independently reproduced findings.

The collaborator's DeBERTa test report claims accuracy 0.7143, fake-class precision 0.8674, recall 0.2809, F1 0.4243, ROC-AUC 0.7152, and average precision 0.6639 on 4,474 examples. The report names `model.safetensors` and `data/splits/test.csv`, but neither file is present in the fetched repository tree; only model configuration/tokenizer metadata and reported metrics are present. The scores therefore remain **unverified here** and must not be presented as results reproduced by this project. The companion baseline report is explicitly validation-set evidence, so its figures also must not be compared as a same-split test baseline.

The backend's model loader checks for actual weights and returns an unavailable status when they are missing. A frontend and prediction API are now present as a software prototype, but the repository does not contain a deployable trained model, and the API/UI addition is not evidence of a validated or production-ready detector.

