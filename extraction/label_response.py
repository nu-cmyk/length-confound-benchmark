"""Three-rule labeling function for hallucination detection benchmarks.

For each generated response and a set of gold reference answers, assign:
  y = 0 (truthful)  if any of the three rules fire
  y = 1 (hallucinated) otherwise

Rule 1: normalized exact match.
Rule 2: substring containment of a normalized gold answer (>=2 chars) inside
        the normalized response.
Rule 3: ROUGE-L F1 against any gold answer >= 0.3.

This is the labeling used for all 12 open-ended generation conditions in the
paper. HaluEval uses its dataset-provided hallucination flag instead.

Normalization is minimal by design: lowercase, whitespace collapsed, and a
small set of answer-prefixes ("answer:", "the answer is", "the answer is:")
stripped. No article-stripping, no punctuation removal.
"""
import re
from typing import List, Tuple


_ANSWER_PREFIXES = ('the answer is', 'answer:', 'the answer is:')


def normalize_text(text: str) -> str:
    """Lowercase, collapse whitespace, strip canned answer prefixes."""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    for prefix in _ANSWER_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    return text


def compute_rouge_l(prediction: str, reference: str) -> float:
    """Token-level ROUGE-L F1 between prediction and reference.

    Tokenization is whitespace splitting on the lowercased strings (NOT the
    normalized form). Returns 0.0 if either side is empty.
    """
    def _lcs(x: List[str], y: List[str]) -> int:
        m, n = len(x), len(y)
        if m == 0 or n == 0:
            return 0
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(curr[j - 1], prev[j])
            prev, curr = curr, [0] * (n + 1)
        return prev[n]

    pt = prediction.lower().split()
    rt = reference.lower().split()
    if not pt or not rt:
        return 0.0
    lcs = _lcs(pt, rt)
    p = lcs / len(pt) if pt else 0.0
    r = lcs / len(rt) if rt else 0.0
    if p + r == 0:
        return 0.0
    return 2.0 * p * r / (p + r)


def label_response(response: str,
                   reference_answers: List[str],
                   rouge_threshold: float = 0.3) -> Tuple[int, float]:
    """Assign a binary truthfulness label to a single response.

    Args:
        response: the model's generated response string.
        reference_answers: list of gold-answer strings (aliases included).
        rouge_threshold: ROUGE-L F1 threshold for Rule 3 (default 0.3).

    Returns:
        (label, best_rouge):
            label = 0 if truthful (any rule fires), 1 if hallucinated.
            best_rouge = maximum ROUGE-L F1 seen against any gold answer
                         (1.0 if Rule 1 or Rule 2 short-circuited).
    """
    if not response or not response.strip():
        return 1, 0.0

    best_rouge = 0.0
    for ref in reference_answers:
        if not ref or not ref.strip():
            continue

        # Rule 1: normalized exact match
        if normalize_text(response) == normalize_text(ref):
            return 0, 1.0

        # Rule 2: substring containment (gold must be >=2 chars after norm)
        if (len(normalize_text(ref)) >= 2
                and normalize_text(ref) in normalize_text(response)):
            return 0, 1.0

        # Rule 3 (deferred): track max ROUGE-L F1 across references
        score = compute_rouge_l(response, ref)
        best_rouge = max(best_rouge, score)

    label = 0 if best_rouge >= rouge_threshold else 1
    return label, best_rouge


if __name__ == '__main__':
    # Sanity checks
    refs = ['Paris', 'Paris, France']
    assert label_response('Paris', refs)[0] == 0          # Rule 1
    assert label_response('The capital is Paris.', refs)[0] == 0  # Rule 2
    assert label_response('Lyon', refs)[0] == 1            # no rule fires
    assert label_response('', refs)[0] == 1                # empty response
    print('label_response: OK')
