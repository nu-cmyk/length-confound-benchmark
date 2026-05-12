"""Teacher-forced hidden-state extraction for HaluEval.

Covers 1 of 13 conditions in the benchmark:
  Llama-3-8B-Instruct: HaluEval (QA samples split)

HaluEval is special. Unlike the 12 open-ended conditions, we do NOT generate a
response. Instead, the dataset provides:
  - knowledge: a short evidence passage
  - question: a question
  - answer:    a pre-written answer (either truthful or hallucinated)
  - hallucination: a yes/no flag

We construct a prompt where the dataset-provided answer is appended directly to
the assistant turn (teacher-forced), then run a single forward pass and extract
the hidden state at the last token of the answer. The label is the dataset's
hallucination flag, NOT the 3-rule labeling.

This is the "teacher-forced acquisition" regime described in Section 3 of the
paper. Because the response text is dataset-authored rather than model-generated,
the response length reflects the dataset authors' writing tendencies rather than
the model's behavior. This is what makes HaluEval's length confound so extreme
(length-only AUROC = 0.969 on Llama-3).
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '8'
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import time
import gc
import warnings
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from tqdm import tqdm
import torch

from prompts import format_prompt_halueval

warnings.filterwarnings('ignore')


@dataclass
class Config:
    model_name: str = 'meta-llama/Meta-Llama-3-8B-Instruct'
    dataset: str = 'halueval'
    torch_dtype: str = 'float16'
    reasoning_subspace_ratio: float = 0.05
    max_samples: int = 3000
    cache_dir: str = './data/cache'
    seed: int = 42

    def __post_init__(self):
        os.makedirs(self.cache_dir, exist_ok=True)


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------
def load_halueval_samples(max_samples: int = 3000):
    """Load HaluEval QA samples.

    Each row is already a labeled sample (the dataset pairs truthful and
    hallucinated answers for each question separately and assigns the flag).
    No pairing logic required.
    """
    from datasets import load_dataset
    print("  Loading HaluEval (qa_samples split)...")
    ds = load_dataset('pminervini/HaluEval', 'qa_samples', split='data')

    samples = []
    for row in ds:
        q = row.get('question', '')
        knowledge = row.get('knowledge', '')
        answer = row.get('answer', '')
        hallu_str = str(row.get('hallucination', '')).strip().lower()
        if not (q and answer):
            continue
        label = 1 if hallu_str == 'yes' else 0
        samples.append({
            'question': q,
            'knowledge': knowledge,
            'answer': answer,
            'label': label,
        })

    print(f"  Total HaluEval samples: {len(samples)}")
    if len(samples) > max_samples:
        np.random.seed(42)
        idx = np.random.choice(len(samples), max_samples, replace=False)
        samples = [samples[i] for i in idx]
    print(f"  Sampled: {len(samples)}")
    n_pos = sum(s['label'] for s in samples)
    print(f"  Class balance: truthful={len(samples) - n_pos}, halluc={n_pos}")
    return samples


# ---------------------------------------------------------------------------
# Model and reasoning-subspace setup
# ---------------------------------------------------------------------------
def load_model(cfg: Config):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"\nLoading model: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name,
                                              trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = getattr(torch, cfg.torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=dtype, device_map={'': 0},
        trust_remote_code=True, output_hidden_states=True)
    model.eval()
    print(f"  Layers: {model.config.num_hidden_layers}, "
          f"D: {model.config.hidden_size}")
    return model, tokenizer


def load_or_compute_reasoning_subspace(model, cfg: Config) -> Tuple[np.ndarray, int]:
    cache_path = os.path.join(
        cfg.cache_dir,
        f"{cfg.model_name.replace('/', '_')}_reasoning_subspace.npz")
    if os.path.exists(cache_path):
        print(f"  Loading cached V_R: {cache_path}")
        data = np.load(cache_path)
        return data['V_R'], int(data['reasoning_dim'])

    print("  Computing V_R via SVD of W_unemb...")
    W = model.lm_head.weight.detach().float().cpu()
    _U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    V = Vt.T
    total = min(W.shape)
    sem_dim = int(total * (1 - cfg.reasoning_subspace_ratio))
    r_dim = total - sem_dim
    V_R = V[:, sem_dim:].numpy()
    np.savez(cache_path, V_R=V_R, singular_values=S.numpy(),
             semantic_dim=sem_dim, reasoning_dim=r_dim)
    return V_R, r_dim


# ---------------------------------------------------------------------------
# Per-sample feature extraction (forward-pass only, no generation)
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_features(model, tokenizer, prompt: str, V_R: np.ndarray):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True,
                       max_length=1024).to(device)

    hook_data = {'attn': {}, 'ffn': {}}
    hooks = []
    layers = model.model.layers
    for idx, layer in enumerate(layers):
        def _a_hook(i):
            def fn(_m, _inp, out):
                t = out[0] if isinstance(out, tuple) else out
                hook_data['attn'][i] = t.detach()
            return fn

        def _f_hook(i):
            def fn(_m, _inp, out):
                t = out[0] if isinstance(out, tuple) else out
                hook_data['ffn'][i] = t.detach()
            return fn

        hooks.append(layer.self_attn.register_forward_hook(_a_hook(idx)))
        hooks.append(layer.mlp.register_forward_hook(_f_hook(idx)))

    try:
        out = model(**inputs, output_hidden_states=True)
    finally:
        for h in hooks:
            h.remove()

    # Hidden state at the last token of the prompt+answer sequence
    hidden = np.stack([h[0, -1, :].cpu().float().numpy()
                       for h in out.hidden_states])
    n_layers = len(layers)
    D = hidden.shape[1]
    attn = np.stack([
        (hook_data['attn'][l][0, -1, :].cpu().float().numpy()
         if l in hook_data['attn'] else np.zeros(D))
        for l in range(n_layers)
    ])
    ffn = np.stack([
        (hook_data['ffn'][l][0, -1, :].cpu().float().numpy()
         if l in hook_data['ffn'] else np.zeros(D))
        for l in range(n_layers)
    ])

    proj_h = hidden @ V_R
    proj_a = attn @ V_R
    proj_m = ffn @ V_R

    return {
        'hidden_states': hidden,
        'proj_h': proj_h,
        'proj_a': proj_a,
        'proj_m': proj_m,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_extraction(cfg: Config):
    t0 = time.time()
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    prefix = cfg.model_name.replace('/', '_')
    ft_cache = os.path.join(cfg.cache_dir,
                            f"{prefix}_{cfg.dataset}_rtraj_features.npz")
    hs_cache = os.path.join(cfg.cache_dir,
                            f"{prefix}_{cfg.dataset}_rtraj_hidden.npz")
    if os.path.exists(ft_cache):
        print(f"  Cache exists: {ft_cache}")
        return ft_cache

    model, tokenizer = load_model(cfg)
    V_R, r_dim = load_or_compute_reasoning_subspace(model, cfg)
    samples = load_halueval_samples(cfg.max_samples)

    keys = ['hidden_states', 'proj_h', 'proj_a', 'proj_m',
            'labels', 'responses', 'questions']
    results = {k: [] for k in keys}
    n_success = 0
    n_fail = 0

    for i, s in enumerate(tqdm(samples, desc="HaluEval")):
        prompt = format_prompt_halueval(s['question'], s['knowledge'],
                                        s['answer'])
        try:
            r = extract_features(model, tokenizer, prompt, V_R)
        except Exception as e:
            if i < 3:
                print(f"  sample {i} failed: {e}")
            n_fail += 1
            continue
        results['hidden_states'].append(r['hidden_states'])
        results['proj_h'].append(r['proj_h'])
        results['proj_a'].append(r['proj_a'])
        results['proj_m'].append(r['proj_m'])
        results['labels'].append(s['label'])
        # For HaluEval, "response" is the dataset-authored answer
        results['responses'].append(s['answer'])
        results['questions'].append(s['question'])
        n_success += 1

        if (i + 1) % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    print(f"\n  Extracted: {n_success} ({n_fail} failed)")
    if n_success == 0:
        print("  Nothing extracted, aborting save.")
        return None

    labels_arr = np.array(results['labels'])
    print(f"  Labels: truthful={int((labels_arr == 0).sum())}, "
          f"halluc={int((labels_arr == 1).sum())}")

    np.savez(ft_cache,
             proj_h_reasoning=np.stack(results['proj_h']),
             proj_a_reasoning=np.stack(results['proj_a']),
             proj_m_reasoning=np.stack(results['proj_m']),
             labels=labels_arr,
             responses=np.array(results['responses'], dtype=object),
             questions=np.array(results['questions'], dtype=object),
             reasoning_dim=r_dim)
    np.savez(hs_cache, hidden_states=np.stack(results['hidden_states']))

    print(f"  Saved: {ft_cache}")
    print(f"  Time: {(time.time() - t0) / 60:.1f} min")
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return ft_cache


if __name__ == '__main__':
    cfg = Config()
    try:
        run_extraction(cfg)
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
    print("\nHALUEVAL EXTRACTION COMPLETE")
