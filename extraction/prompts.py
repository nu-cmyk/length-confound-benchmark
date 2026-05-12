"""Per-model chat templates for prompt formatting.

Each model family has its own native chat template. We use the model's
recommended format and do NOT include a system prompt, to keep behavior as
close to native as possible across the evaluation suite.

Three prompt regimes:

- format_prompt_closed_book: for TriviaQA, NQ-Open, TruthfulQA. The model
  receives only the question and is asked to answer concisely.

- format_prompt_passage_grounded: for CoQA, TyDi-QA. The model receives a
  passage (truncated to 400 whitespace tokens) plus any conversational
  history (CoQA only) and a question.

- format_prompt_halueval: for HaluEval (teacher-forced). The dataset-authored
  answer is appended to the assistant turn directly; no generation happens.
"""
from typing import Optional


def format_prompt_closed_book(question: str, model_name: str) -> str:
    """Closed-book QA prompt for TriviaQA, NQ-Open, TruthfulQA."""
    mn = model_name.lower()
    if 'llama-3' in mn or 'llama3' in mn:
        return (f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                f"Answer the following question concisely and factually.\n\n"
                f"Question: {question}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n")
    if 'mistral' in mn:
        return (f"[INST] Answer the following question concisely and "
                f"factually.\n\nQuestion: {question}\n\nAnswer: [/INST]")
    if 'qwen' in mn:
        return (f"<|im_start|>user\n"
                f"Answer the following question concisely and factually.\n\n"
                f"Question: {question}<|im_end|>\n"
                f"<|im_start|>assistant\n")
    return f"Question: {question}\nAnswer:"


def format_prompt_passage_grounded(question: str,
                                   passage: str,
                                   history: str = '',
                                   model_name: str = '') -> str:
    """Passage-grounded QA prompt for CoQA and TyDi-QA.

    Truncates passage to 400 whitespace tokens (Section 3 of the paper).
    `history` is the conversational prefix for CoQA (empty for TyDi-QA).
    """
    passage_tokens = passage.split()
    if len(passage_tokens) > 400:
        passage = ' '.join(passage_tokens[:400]) + '...'

    user_content = (f"Passage: {passage}\n\n"
                    f"{history}"
                    f"Question: {question}")

    mn = model_name.lower()
    if 'llama-3' in mn or 'llama3' in mn:
        return (f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                f"Answer the following question using the given passage. "
                f"Answer concisely and factually.\n\n"
                f"{user_content}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n")
    if 'mistral' in mn:
        return (f"[INST] Answer the following question using the given "
                f"passage. Answer concisely and factually.\n\n"
                f"{user_content}\n\nAnswer: [/INST]")
    if 'qwen' in mn:
        return (f"<|im_start|>user\n"
                f"Answer the following question using the given passage. "
                f"Answer concisely and factually.\n\n"
                f"{user_content}<|im_end|>\n"
                f"<|im_start|>assistant\n")
    return f"{user_content}\nAnswer:"


def format_prompt_halueval(question: str,
                           knowledge: str,
                           answer: str) -> str:
    """Teacher-forced prompt for HaluEval.

    The dataset-authored answer is appended directly to the assistant turn.
    We extract the hidden state at the last non-padding token of this
    concatenated sequence (no generation).

    Only Llama-3 is evaluated on HaluEval in our benchmark.
    """
    return (f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"Knowledge: {knowledge}\n\n"
            f"Question: {question}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{answer}")
