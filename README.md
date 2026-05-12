# Length-Confound Benchmark

Reference implementation and benchmark for "Auditing Length Confounds in Hallucination Detection Benchmarks" (submitted to NeurIPS 2026 Datasets and Benchmarks).

Anonymous review repository: https://anonymous.4open.science/r/length-confound-benchmark-B67F

## TL;DR

Response length alone discriminates hallucinated from truthful responses with AUROC up to 0.969 across 13 standard (model, dataset) conditions. This repository provides:

- A 13-condition cached benchmark of residual-stream hidden states, responses, and binary truthfulness labels across Qwen2.5-7B-Instruct, Mistral-7B-Instruct-v0.2, and Meta-Llama-3-8B-Instruct.
- A dual-correction audit protocol: cross-fitted polynomial residualization (Strategy I) plus caliper length-matching (Strategy II), with a length-share statistic and Nadeau-Bengio significance testing.
- Reference implementations for 9 hallucination detectors: SAPLMA, HARP, HaloScope-simplified, LayerCovariance, MiddleLayerProbe, XGB-AllLayers, SelfCheckGPT-Unigram, SemanticEntropy, and a Length-only cheating-floor baseline.

## Repository layout

    length-confound-benchmark/
    +-- extraction/                  Phase 1: generate responses, cache hidden states
    |   +-- label_response.py        Three-rule labeling (exact, substring, ROUGE-L)
    |   +-- prompts.py               Per-model chat templates
    |   +-- extract_closed_book.py   TriviaQA, NQ-Open, TruthfulQA (open-ended)
    |   +-- extract_passage_grounded.py  CoQA, TyDi-QA (passage-grounded, 400-tok cap)
    |   +-- extract_halueval.py      HaluEval (teacher-forced)
    +-- baselines/                   Phase 2: scalar baseline scorers
    |   +-- selfcheck_generate.py    SelfCheckGPT-Unigram (K=10 stochastic samples)
    |   +-- semantic_entropy.py      DeBERTa-MNLI bidirectional entailment clustering
    +-- audit/                       Phase 3: length-confound audit
    |   +-- methods.py               Detector feature builders + classifier registry
    |   +-- corrections.py           Polynomial residualization, caliper, bin-exact
    |   +-- significance.py          Nadeau-Bengio + meta-analytic + Wilcoxon
    |   +-- run_audit.py             Phase 3a + 3b driver
    +-- analysis/                    Phase 4: table and figure generation
    |   +-- build_tables.py          Tables 1-5
    |   +-- halueval_sensitivity.py  Table 6 (Section 7.5)
    +-- data/                        Pointer to Zenodo cache
        +-- README.md

## Quick start (audit only, using pre-extracted cache)

The full extraction is compute-intensive (about 30 hours on A100 across all 13 conditions). For most reproductions, download the pre-computed cache and run only the audit:

    pip install -r requirements.txt
    # Download cached hidden states (~15 GB); see data/README.md for the Zenodo link
    python audit/run_audit.py
    python analysis/build_tables.py
    python analysis/halueval_sensitivity.py

The first command produces `data/audit_results/phase3_main_results.json`. The next two produce the paper's Tables 1-6 in markdown.

## Re-running the full pipeline

    # Phase 1: hidden-state extraction (~30 hours A100, 13 conditions)
    python extraction/extract_closed_book.py
    python extraction/extract_passage_grounded.py
    python extraction/extract_halueval.py

    # Phase 2: stochastic-sample baselines (closed-book only, ~15 hours A100)
    python baselines/selfcheck_generate.py
    python baselines/semantic_entropy.py

    # Phase 3: audit
    python audit/run_audit.py

    # Phase 4: tables
    python analysis/build_tables.py
    python analysis/halueval_sensitivity.py

## Conditions audited

13 (model, dataset) pairings:

| Model | Datasets |
|---|---|
| Qwen2.5-7B-Instruct | TriviaQA, TruthfulQA, CoQA, TyDi-QA |
| Mistral-7B-Instruct-v0.2 | TriviaQA, TruthfulQA, CoQA, TyDi-QA |
| Meta-Llama-3-8B-Instruct | TriviaQA, NQ-Open, CoQA, TyDi-QA, HaluEval |

Five of the 18 possible pairings are excluded due to either prompt-format restrictions or insufficient sample yield (under 800 samples after labeling).

## Labeling regimes

Three regimes, depending on the dataset:

1. **Open-ended generation with three-rule labeling** (12 conditions). The model generates a response and the response is labeled truthful (y=0) if any of:
   - normalized exact match against a gold reference, OR
   - substring containment of a normalized gold reference of at least two characters, OR
   - ROUGE-L F1 against any gold reference at least 0.3.

   Otherwise hallucinated (y=1). See `extraction/label_response.py`.

2. **Teacher-forced with dataset label** (HaluEval only). The dataset-authored answer is appended to the assistant turn and a single forward pass is run; the dataset's `hallucination` flag is the label. No generation, no three-rule labeling.

3. **Single-reference labeling** (CoQA only, special case of regime 1). Each conversational turn has one human reference answer rather than a list of aliases.

## Detectors

Nine detectors are audited (Section 4 of the paper):

- **Length-only**: balanced logistic regression on the response length scalar. The cheating floor.
- **SAPLMA**: PCA-128 of the last-layer hidden state, MLP classifier.
- **MiddleLayerProbe**: PCA-128 of the middle-layer hidden state, MLP. Inspired by MIND.
- **HARP**: PCA-200 of V_R-projected per-layer features, XGBoost.
- **XGB-AllLayers**: PCA-200 of flattened all-layer hidden states, XGBoost. Also the V_R ablation for HARP.
- **HaloScope-simplified**: TruncatedSVD top-3 components on the last three layers, logistic regression. Supervised single-response adaptation.
- **LayerCovariance**: log-eigenspectrum of the per-sample cross-layer covariance, logistic regression. Single-response adaptation inspired by INSIDE.
- **SelfCheckGPT-Unigram**: 1 - mean unigram overlap across K=10 stochastic samples.
- **SemanticEntropy**: bidirectional entailment clustering of K=10 stochastic samples via DeBERTa-large-MNLI; Shannon entropy of the cluster distribution.

## Citation

    @inproceedings{anon2026lengthconfound,
      title={Auditing Length Confounds in Hallucination Detection Benchmarks},
      author={Anonymous},
      booktitle={NeurIPS Datasets and Benchmarks},
      year={2026}
    }

## License

MIT. See `LICENSE`.
