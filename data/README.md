# Cached benchmark data

The audit pipeline operates on pre-computed residual-stream hidden states,
not on the raw LLM models. This separation means the audit can be re-run
in under an hour once the cache is downloaded, without the ~30 hours of
forward passes required to build the cache from scratch.

## Availability during review

The full ~15 GB cache cannot be hosted on GitHub due to file-size limits.
During the NeurIPS review period the cache is available on request via the
OpenReview discussion forum. Upon paper de-anonymization the full cache
will be permanently released on Zenodo (DOI to be added at that time).

To validate the audit pipeline without the raw hidden-state cache,
`data/audit_results/phase3_main_results.json` contains the per-fold AUROC
values for all 9 detectors x 13 conditions x 4 evaluation modes. Running
`python analysis/build_tables.py` regenerates Tables 1-5 directly from
this JSON, no `.npz` files required.

## What the full cache contains

For each of the 13 (model, dataset) conditions, two files are produced:

- `{model_prefix}_{dataset}_rtraj_features.npz` (~10-50 MB)
  Contains: `proj_h_reasoning`, `proj_a_reasoning`, `proj_m_reasoning`,
  `labels`, `responses`, `questions`, `reasoning_dim`.

- `{model_prefix}_{dataset}_rtraj_hidden.npz` (~500 MB-1.5 GB)
  Contains: `hidden_states` of shape `(N, L+1, D)` where N is the number
  of samples, L is the number of transformer layers, and D is the hidden
  dimension.

Plus one shared file per model:

- `{model_prefix}_reasoning_subspace.npz`
  Contains: `V_R` (reasoning-subspace projection matrix from SVD of the
  unembedding matrix), `singular_values`, `semantic_dim`, `reasoning_dim`.

For SelfCheckGPT and SemanticEntropy, additional cached files live in
sibling directories (`data/selfcheck/` and `data/semantic_entropy/`).

## Expected directory layout after extraction

    data/
      cache/
        Qwen_Qwen2.5-7B-Instruct_triviaqa_rtraj_features.npz
        Qwen_Qwen2.5-7B-Instruct_triviaqa_rtraj_hidden.npz
        Qwen_Qwen2.5-7B-Instruct_truthfulqa_rtraj_features.npz
        ...
      selfcheck/
        Qwen_Qwen2.5-7B-Instruct_triviaqa_selfcheck.npz
        ...
      semantic_entropy/
        Qwen_Qwen2.5-7B-Instruct_triviaqa_semantic_entropy.npz
        ...
      audit_results/
        phase3_main_results.json        (committed in this repo)
        tables.md                       (produced by analysis/build_tables.py)

## Reproducing the cache from scratch

If you prefer to rebuild the cache yourself (about 30 hours on a single
A100-80GB across all 13 conditions), run:

    python extraction/extract_closed_book.py
    python extraction/extract_passage_grounded.py
    python extraction/extract_halueval.py
    python baselines/selfcheck_generate.py
    python baselines/semantic_entropy.py

This downloads the underlying HuggingFace datasets and the three
instruction-tuned LLMs (Qwen2.5-7B-Instruct, Mistral-7B-Instruct-v0.2,
Meta-Llama-3-8B-Instruct), then performs greedy decoding (or teacher-forced
forward passes for HaluEval) and saves the hidden states.

## Hardware requirements for reproduction

- A single GPU with at least 24 GB of memory (one A100-80GB is what the
  paper uses). The Llama-3-8B-Instruct model in fp16 requires ~16 GB; the
  hooks and KV cache push peak usage somewhat higher.
- About 20 GB of disk space for the cache.
- Internet access to download HuggingFace datasets and model checkpoints.

## Dataset licenses

The cache derives from publicly released datasets and their respective
licenses apply to any redistribution:

- TriviaQA: Apache 2.0
- NQ-Open: CC BY-SA 3.0
- TruthfulQA: Apache 2.0
- CoQA: research use only (Stanford CoQA terms)
- TyDi-QA: Apache 2.0
- HaluEval: MIT

Model licenses also apply to redistributed weights (not redistributed here).
