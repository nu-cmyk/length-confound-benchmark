"""SelfCheckGPT-Unigram baseline.

For each prompt:
  1. Take the original greedy response from the cached extraction features
     (already saved in *_rtraj_features.npz).
  2. Generate K=10 additional stochastic samples at temperature=1.0, top_p=0.9.
  3. Compute unigram overlap between the original and each sample.
  4. Aggregate: 1 - mean(overlap) is the inconsistency score (higher = more
     likely hallucinated).

Following Manakul et al. (2023), unigram-max is the strongest non-NLI variant
and runs ~20x faster than BERTScore. We report mean, min, and max variants;
the paper uses unigram-mean.

Restricted to the 6 closed-book conditions:
  Qwen2.5-7B-Instruct:      TriviaQA, TruthfulQA
  Mistral-7B-Instruct-v0.2: TriviaQA, TruthfulQA
  Llama-3-8B-Instruct:      TriviaQA, NQ-Open

This is the n=6 detector audited in Table 4 of the paper.
"""
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OPENBLAS_NUM_THREADS'] = '8'

import time
import gc
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List
import numpy as np
from tqdm import tqdm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'extraction'))
from prompts import format_prompt_closed_book


CACHE_DIR = './data/cache'
OUT_DIR = './data/selfcheck'
os.makedirs(OUT_DIR, exist_ok=True)


@dataclass
class Config:
    model_name: str = 'Qwen/Qwen2.5-7B-Instruct'
    dataset: str = 'triviaqa'
    n_samples: int = 10           # K in SelfCheckGPT
    temperature: float = 1.0
    top_p: float = 0.9
    max_new_tokens: int = 128


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_samples(model, tokenizer, prompt: str, cfg: Config,
                     device) -> List[str]:
    """Generate K stochastic samples for one prompt (batched)."""
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True,
                       max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_len = inputs['input_ids'].shape[1]

    out = model.generate(
        **inputs,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=True,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        num_return_sequences=cfg.n_samples,
        pad_token_id=tokenizer.eos_token_id,
    )

    samples = []
    for i in range(cfg.n_samples):
        gen_ids = out[i, prompt_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        samples.append(text)
    return samples


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def ngram_overlap(original: str, sample: str, n: int = 1) -> float:
    """Proportion of original's n-grams that appear in the sample."""
    def ngrams(s, n):
        toks = s.lower().split()
        if len(toks) < n:
            return set()
        return set(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))

    og = ngrams(original, n)
    sg = ngrams(sample, n)
    if not og:
        return 1.0 if not sg else 0.0
    return len(og & sg) / len(og)


def selfcheck_score(original: str, samples: List[str]) -> dict:
    """SelfCheckGPT-Unigram score: 1 - mean/min/max(overlap).

    Higher = more inconsistency = more likely hallucinated.
    """
    if not original.strip():
        return {'ngram_mean': 1.0, 'ngram_min': 1.0, 'ngram_max': 1.0}
    scores = [ngram_overlap(original, s) for s in samples if s.strip()]
    if not scores:
        return {'ngram_mean': 1.0, 'ngram_min': 1.0, 'ngram_max': 1.0}
    return {
        'ngram_mean': 1.0 - float(np.mean(scores)),
        'ngram_min':  1.0 - float(np.min(scores)),
        'ngram_max':  1.0 - float(np.max(scores)),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_condition(model_name: str, dataset: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    cfg = Config(model_name=model_name, dataset=dataset)

    prefix = model_name.replace('/', '_')
    ft_path = f"{CACHE_DIR}/{prefix}_{dataset}_rtraj_features.npz"
    out_path = f"{OUT_DIR}/{prefix}_{dataset}_selfcheck.npz"

    if Path(out_path).exists():
        print(f"  SKIP (cached): {out_path}")
        return out_path
    if not Path(ft_path).exists():
        print(f"  NO DATA: {ft_path}")
        return None

    print(f"\n{'=' * 70}\n{model_name} + {dataset}\n{'=' * 70}")
    ft = np.load(ft_path, allow_pickle=True)
    questions = ft['questions']
    responses = ft['responses']
    labels = ft['labels'].astype(int)
    N = len(questions)
    print(f"  N={N}, loading model...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map={'': 0},
        trust_remote_code=True)
    model.eval()
    device = next(model.parameters()).device

    scores_mean = np.zeros(N, dtype=np.float32)
    scores_min = np.zeros(N, dtype=np.float32)
    scores_max = np.zeros(N, dtype=np.float32)
    all_samples = []

    t0 = time.time()
    for i in tqdm(range(N), desc="SelfCheck"):
        q = str(questions[i])
        orig = str(responses[i])
        prompt = format_prompt_closed_book(q, model_name)

        try:
            samples = generate_samples(model, tokenizer, prompt, cfg, device)
        except Exception as e:
            print(f"  failed sample {i}: {e}")
            samples = []

        s = selfcheck_score(orig, samples)
        scores_mean[i] = s['ngram_mean']
        scores_min[i] = s['ngram_min']
        scores_max[i] = s['ngram_max']
        all_samples.append(samples)

        if (i + 1) % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    elapsed = (time.time() - t0) / 60
    print(f"  Done in {elapsed:.1f} min")

    np.savez(out_path,
             scores_mean=scores_mean,
             scores_min=scores_min,
             scores_max=scores_max,
             labels=labels,
             samples=np.array(all_samples, dtype=object))
    print(f"  Saved: {out_path}")

    # Quick sanity AUROC
    from sklearn.metrics import roc_auc_score
    for name, s in [('mean', scores_mean),
                    ('min', scores_min),
                    ('max', scores_max)]:
        try:
            a = roc_auc_score(labels, s)
            if a < 0.5:
                a = 1 - a
            print(f"    SelfCheck-Unigram-{name}: AUROC={a:.4f}")
        except Exception:
            pass

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return out_path


if __name__ == '__main__':
    experiments = [
        ('Qwen/Qwen2.5-7B-Instruct',            'truthfulqa'),
        ('mistralai/Mistral-7B-Instruct-v0.2',  'truthfulqa'),
        ('Qwen/Qwen2.5-7B-Instruct',            'triviaqa'),
        ('mistralai/Mistral-7B-Instruct-v0.2',  'triviaqa'),
        ('meta-llama/Meta-Llama-3-8B-Instruct', 'triviaqa'),
        ('meta-llama/Meta-Llama-3-8B-Instruct', 'nq_open'),
    ]
    for model_name, dataset in experiments:
        try:
            run_condition(model_name, dataset)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
    print("\n" + "=" * 70)
    print("SELFCHECKGPT GENERATION COMPLETE")
    print("=" * 70)
