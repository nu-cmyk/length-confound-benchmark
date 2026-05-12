"""Hidden-state extraction for closed-book QA conditions.

Covers 8 of 13 conditions in the benchmark:
  Qwen2.5-7B-Instruct:     TriviaQA, TruthfulQA
  Mistral-7B-Instruct-v0.2: TriviaQA, TruthfulQA
  Llama-3-8B-Instruct:     TriviaQA, NQ-Open

For each (model, dataset) we:
  1. Generate a response via greedy decoding (max_new_tokens=128).
  2. Cache the residual-stream hidden states at the final generated token
     across all layers (L+1, D).
  3. Cache attention and FFN contributions per layer.
  4. Project hidden states into a "reasoning subspace" V_R derived from the
     bottom 5% of singular values of the unembedding matrix.
  5. Assign a binary truthfulness label via the 3-rule labeling
     (extraction/label_response.py).

The output files are:
  {cache_dir}/{model_prefix}_{dataset}_rtraj_features.npz   # labels, responses, etc.
  {cache_dir}/{model_prefix}_{dataset}_rtraj_hidden.npz     # hidden states only

These files are the input to the audit pipeline.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '8'
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import time
import gc
import warnings
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from tqdm import tqdm
import torch

from label_response import label_response
from prompts import format_prompt_closed_book

warnings.filterwarnings('ignore')


@dataclass
class Config:
    model_name: str = 'Qwen/Qwen2.5-7B-Instruct'
    dataset: str = 'triviaqa'                # triviaqa | nq_open | truthfulqa
    max_new_tokens: int = 128
    torch_dtype: str = 'float16'
    reasoning_subspace_ratio: float = 0.05   # bottom 5% of singular values
    max_samples: int = 3000
    cache_dir: str = './data/cache'
    seed: int = 42

    def __post_init__(self):
        os.makedirs(self.cache_dir, exist_ok=True)


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------
def load_triviaqa_samples(max_samples: int = 3000) -> List[dict]:
    from datasets import load_dataset
    print("  Loading TriviaQA (rc.nocontext)...")
    ds = load_dataset('trivia_qa', 'rc.nocontext', split='validation')
    samples = []
    for row in ds:
        q = row.get('question', '')
        answer_obj = row.get('answer', {})
        if isinstance(answer_obj, dict):
            aliases = answer_obj.get('aliases', [])
            value = answer_obj.get('value', '')
            refs = aliases + ([value] if value else [])
        else:
            refs = [answer_obj] if isinstance(answer_obj, str) else []
        if q and refs:
            samples.append({'question': q, 'reference_answers': refs})
    if len(samples) > max_samples:
        np.random.seed(42)
        idx = np.random.choice(len(samples), max_samples, replace=False)
        samples = [samples[i] for i in idx]
    print(f"  TriviaQA: {len(samples)} samples")
    return samples


def load_nq_open_samples(max_samples: int = 3000) -> List[dict]:
    from datasets import load_dataset
    print("  Loading NQ-Open...")
    ds = load_dataset('nq_open', split='validation')
    samples = []
    for row in ds:
        q = row.get('question', '')
        answers = row.get('answer', [])
        if isinstance(answers, str):
            answers = [answers]
        if q and answers:
            samples.append({'question': q, 'reference_answers': answers})
    if len(samples) > max_samples:
        np.random.seed(42)
        idx = np.random.choice(len(samples), max_samples, replace=False)
        samples = [samples[i] for i in idx]
    print(f"  NQ-Open: {len(samples)} samples")
    return samples


def load_truthfulqa_samples(max_samples: int = 3000) -> List[dict]:
    from datasets import load_dataset
    print("  Loading TruthfulQA (generation)...")
    ds = load_dataset('truthful_qa', 'generation', split='validation')
    samples = []
    for row in ds:
        q = row.get('question', '')
        # correct_answers is a list; best_answer is a single string
        correct = row.get('correct_answers', []) or []
        best = row.get('best_answer', '')
        refs = list(correct)
        if best and best not in refs:
            refs.append(best)
        if q and refs:
            samples.append({'question': q, 'reference_answers': refs})
    if len(samples) > max_samples:
        np.random.seed(42)
        idx = np.random.choice(len(samples), max_samples, replace=False)
        samples = [samples[i] for i in idx]
    print(f"  TruthfulQA: {len(samples)} samples")
    return samples


_LOADERS = {
    'triviaqa':   load_triviaqa_samples,
    'nq_open':    load_nq_open_samples,
    'truthfulqa': load_truthfulqa_samples,
}


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
    gpu_mem = (torch.cuda.get_device_properties(0).total_memory / 1e9
               if torch.cuda.is_available() else 0)
    dev_map = {'': 0} if gpu_mem >= 30 else 'auto'
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=dtype, device_map=dev_map,
        trust_remote_code=True, output_hidden_states=True)
    model.eval()
    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    print(f"  Layers: {n_layers}, Hidden dim: {hidden_dim}")
    return model, tokenizer, n_layers, hidden_dim


def extract_reasoning_subspace(model, cfg: Config) -> Tuple[np.ndarray, int]:
    """SVD of the unembedding matrix; bottom 5% of singular directions = V_R."""
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
    print(f"  reasoning_dim={r_dim}, semantic_dim={sem_dim}")
    return V_R, r_dim


# ---------------------------------------------------------------------------
# Per-sample feature extraction
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_with_hooks(model, tokenizer, prompt: str,
                        cfg: Config, V_R: np.ndarray):
    """Generate a response and capture residual-stream + per-layer features."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True,
                       max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_len = inputs['input_ids'].shape[1]

    hook_data = {'attn_outputs': {}, 'ffn_outputs': {}}
    hooks = []
    inner = model.model if hasattr(model, 'model') else model
    layers = inner.layers

    def _attn_hook(idx):
        def fn(_m, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            hook_data['attn_outputs'][idx] = t.detach()
        return fn

    def _ffn_hook(idx):
        def fn(_m, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            hook_data['ffn_outputs'][idx] = t.detach()
        return fn

    for layer_idx, layer in enumerate(layers):
        if hasattr(layer, 'self_attn'):
            hooks.append(layer.self_attn.register_forward_hook(
                _attn_hook(layer_idx)))
        if hasattr(layer, 'mlp'):
            hooks.append(layer.mlp.register_forward_hook(
                _ffn_hook(layer_idx)))

    try:
        outputs = model.generate(
            **inputs, max_new_tokens=cfg.max_new_tokens, do_sample=False,
            output_hidden_states=True, return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id)
    finally:
        for h in hooks:
            h.remove()

    generated_ids = outputs.sequences[0][prompt_len:]
    if len(generated_ids) == 0:
        return None
    response_text = tokenizer.decode(generated_ids,
                                     skip_special_tokens=True).strip()

    all_hidden = outputs.hidden_states
    if not all_hidden:
        return None
    last_step = all_hidden[-1]
    n_layers_plus_1 = len(last_step)
    n_layers = n_layers_plus_1 - 1

    hidden_states = np.stack([
        last_step[l][0, -1, :].cpu().float().numpy()
        for l in range(n_layers_plus_1)
    ])

    D = hidden_states.shape[1]
    attn = np.stack([
        (hook_data['attn_outputs'][l][0, -1, :].cpu().float().numpy()
         if l in hook_data['attn_outputs'] else np.zeros(D))
        for l in range(n_layers)
    ])
    ffn = np.stack([
        (hook_data['ffn_outputs'][l][0, -1, :].cpu().float().numpy()
         if l in hook_data['ffn_outputs'] else np.zeros(D))
        for l in range(n_layers)
    ])

    proj_h = hidden_states @ V_R
    proj_a = attn @ V_R
    proj_m = ffn @ V_R

    return {
        'response': response_text,
        'hidden_states': hidden_states,
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

    print(f"\n{'#' * 80}")
    print(f"Extraction: {cfg.model_name} + {cfg.dataset}")
    print(f"{'#' * 80}")

    prefix = cfg.model_name.replace('/', '_')
    ft_cache = os.path.join(cfg.cache_dir,
                            f"{prefix}_{cfg.dataset}_rtraj_features.npz")
    hs_cache = os.path.join(cfg.cache_dir,
                            f"{prefix}_{cfg.dataset}_rtraj_hidden.npz")
    if os.path.exists(ft_cache):
        print(f"  Cache exists, skipping: {ft_cache}")
        return ft_cache

    model, tokenizer, _, _ = load_model(cfg)
    V_R, r_dim = extract_reasoning_subspace(model, cfg)

    if cfg.dataset not in _LOADERS:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")
    samples = _LOADERS[cfg.dataset](cfg.max_samples)

    keys = ['hidden_states', 'proj_h', 'proj_a', 'proj_m',
            'labels', 'responses', 'questions']
    results = {k: [] for k in keys}
    n_success = 0
    n_fail = 0

    for i, sample in enumerate(tqdm(samples, desc=f"Extract {cfg.dataset}")):
        prompt = format_prompt_closed_book(sample['question'], cfg.model_name)
        try:
            r = generate_with_hooks(model, tokenizer, prompt, cfg, V_R)
        except Exception as e:
            if i < 3:
                print(f"  sample {i} failed: {e}")
            n_fail += 1
            continue
        if r is None:
            n_fail += 1
            continue

        label, _rouge = label_response(r['response'], sample['reference_answers'])
        results['hidden_states'].append(r['hidden_states'])
        results['proj_h'].append(r['proj_h'])
        results['proj_a'].append(r['proj_a'])
        results['proj_m'].append(r['proj_m'])
        results['labels'].append(label)
        results['responses'].append(r['response'])
        results['questions'].append(sample['question'])
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
    experiments = [
        ('Qwen/Qwen2.5-7B-Instruct',            'triviaqa'),
        ('Qwen/Qwen2.5-7B-Instruct',            'truthfulqa'),
        ('mistralai/Mistral-7B-Instruct-v0.2',  'triviaqa'),
        ('mistralai/Mistral-7B-Instruct-v0.2',  'truthfulqa'),
        ('meta-llama/Meta-Llama-3-8B-Instruct', 'triviaqa'),
        ('meta-llama/Meta-Llama-3-8B-Instruct', 'nq_open'),
    ]

    for model_name, dataset in experiments:
        cfg = Config()
        cfg.model_name = model_name
        cfg.dataset = dataset
        try:
            run_extraction(cfg)
            print(f"  OK: {model_name.split('/')[-1]} + {dataset}")
        except Exception as e:
            print(f"  FAIL: {model_name.split('/')[-1]} + {dataset}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 80}\nCLOSED-BOOK EXTRACTION COMPLETE\n{'=' * 80}")
