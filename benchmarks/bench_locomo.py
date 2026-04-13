#!/usr/bin/env python3
"""
Neural Memory Benchmark — LoCoMo (ACL 2024).

Evaluates Neural Memory on long-term conversational memory QA:
  1. Ingest all dialog turns from a conversation into Neural Memory
  2. For each QA question, neural_recall top-k relevant memories
  3. LLM generates short answer from recalled context
  4. Score using LoCoMo's F1 metric (token overlap with Porter stemming)

Modes:
  --mode neural     Neural Memory recall (default)
  --mode baseline   No memory — LLM sees truncated conversation
  --mode full       Full conversation context (if fits in window)

Usage:
    # Neural Memory mode (with local llama-server)
    python bench_locomo.py --mode neural --conversations 2

    # Baseline (no memory)
    python bench_locomo.py --mode baseline --conversations 2

    # NVIDIA NIM
    python bench_locomo.py --mode neural --llm nim

    # Custom top-k
    python bench_locomo.py --mode neural --top-k 10

Environment:
    LLAMA_SERVER_URL  — llama-server endpoint (default: http://localhost:8080/v1)
    LLAMA_MODEL       — model alias (default: GPT)
    NVIDIA_NIM_KEY    — NVIDIA NIM API key (for --llm nim)
    LOCOMO_DATA       — path to locomo10.json (default: ../locomo-bench/data/locomo10.json)
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

import numpy as np

# Ensure python/ is on path
_py_dir = str(Path(__file__).resolve().parent.parent / "python")
if _py_dir not in sys.path:
    sys.path.insert(0, _py_dir)

# LoCoMo eval code
_locomo_dir = str(Path(__file__).resolve().parent.parent.parent / "locomo-bench")
if _locomo_dir not in sys.path:
    sys.path.insert(0, _locomo_dir)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NVIDIA_NIM_KEY = os.environ.get("NVIDIA_NIM_KEY", "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")
NIM_MODEL = os.environ.get("NIM_MODEL", "meta/llama-3.1-405b-instruct")
LOCOMO_DATA = Path(os.environ.get("LOCOMO_DATA", Path.home() / "projects/locomo-bench/data/locomo10.json"))
RESULTS_DIR = Path(__file__).parent / "results" / "locomo"

# LoCoMo QA categories
QA_CATEGORIES = {
    1: "multi-hop",
    2: "temporal",
    3: "single-hop (specific)",
    4: "open-domain",
    5: "adversarial (unanswerable)",
}

# ---------------------------------------------------------------------------
# LLM Client (NVIDIA NIM — Llama 3.1 405B)
# ---------------------------------------------------------------------------

class NIMClient:
    """NVIDIA NIM client — Llama 3.1 405B."""

    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=NVIDIA_NIM_KEY,
        )
        self.model = NIM_MODEL

    def complete(self, prompt: str, max_tokens: int = 256) -> str:
        import time
        max_retries = 5
        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)  # 2, 4, 8, 16 seconds
                    print(f"[nim] Rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait)
                else:
                    print(f"[nim] Error: {e}")
                    return ""


class OpenRouterClient:
    """OpenRouter client — default: Llama 3.1 8B Instruct (free)."""

    def __init__(self, model="meta-llama/llama-3.1-8b-instruct:free"):
        from openai import OpenAI
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        )
        self.model = model

    def complete(self, prompt: str, max_tokens: int = 256) -> str:
        max_retries = 5
        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    print(f"[openrouter] Rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait)
                else:
                    print(f"[openrouter] Error: {e}")
                    return ""


class OllamaClient:
    """Local Ollama client — no rate limits, no API key needed."""

    def __init__(self, model="openhermes:7b-v2.5"):
        from openai import OpenAI
        self.client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",  # dummy — Ollama doesn't need a key
        )
        self.model = model

    def complete(self, prompt: str, max_tokens: int = 256) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"[ollama] Error: {e}")
            return ""


def get_llm(args):
    if hasattr(args, 'llm') and args.llm == 'openrouter':
        model = os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
        return OpenRouterClient(model=model)
    elif hasattr(args, 'llm') and args.llm == 'ollama':
        model = os.environ.get("OLLAMA_MODEL", "openhermes:7b-v2.5")
        return OllamaClient(model=model)
    return NIMClient()

def get_llm_name(args):
    if hasattr(args, 'llm') and args.llm == 'openrouter':
        return os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
    elif hasattr(args, 'llm') and args.llm == 'ollama':
        return os.environ.get("OLLAMA_MODEL", "openhermes:7b-v2.5")
    return NIM_MODEL

# ---------------------------------------------------------------------------
# LoCoMo scoring (ported from task_eval/evaluation.py)
# ---------------------------------------------------------------------------

import string
import unicodedata

def normalize_answer(s):
    s = s.replace(',', "")
    def remove_articles(text):
        return re.sub(r'\b(a|an|the|and)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    """F1 with Porter stemming — matches LoCoMo's eval exactly."""
    from nltk.stem import PorterStemmer
    ps = PorterStemmer()
    prediction_tokens = [ps.stem(w) for w in normalize_answer(prediction).split()]
    ground_truth_tokens = [ps.stem(w) for w in normalize_answer(ground_truth).split()]
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def f1_multi(prediction, ground_truth):
    """Multi-answer F1 — splits on comma, takes max per sub-answer."""
    predictions = [p.strip() for p in prediction.split(',')]
    ground_truths = [g.strip() for g in ground_truth.split(',')]
    return np.mean([max([f1_score(p, gt) for p in predictions]) for gt in ground_truths])


def score_qa(prediction: str, answer, category: int) -> float:
    """Score a single QA prediction against ground truth."""
    # Handle non-string answers (ints, lists)
    if isinstance(answer, list):
        answer = '; '.join(str(a) for a in answer)
    elif not isinstance(answer, str):
        answer = str(answer)

    if category == 3:
        answer = answer.split(';')[0].strip()

    if category in [2, 3, 4]:
        return f1_score(prediction, answer)
    elif category == 1:
        return f1_multi(prediction, answer)
    elif category == 5:
        pred_lower = prediction.lower()
        if 'no information available' in pred_lower or 'not mentioned' in pred_lower:
            return 1.0
        return 0.0
    return 0.0

# ---------------------------------------------------------------------------
# Conversation processing
# ---------------------------------------------------------------------------

def extract_dialogs(data: dict) -> List[dict]:
    """Extract all dialog turns from a LoCoMo conversation sample."""
    conv = data['conversation']
    dialogs = []
    session_nums = sorted([int(k.split('_')[-1]) for k in conv.keys()
                           if k.startswith('session_') and not k.endswith(('_date_time',))])

    for sid in session_nums:
        session_key = f'session_{sid}'
        date_key = f'session_{sid}_date_time'
        if session_key not in conv:
            continue
        date_time = conv.get(date_key, "")
        for turn in conv[session_key]:
            text = f'{turn["speaker"]} said, "{turn["text"]}"'
            if 'blip_caption' in turn and turn['blip_caption']:
                text += f' and shared {turn["blip_caption"]}'

            dialogs.append({
                'dia_id': turn['dia_id'],
                'speaker': turn['speaker'],
                'text': turn['text'],
                'session': sid,
                'date_time': date_time,
                'formatted': text,
            })
    return dialogs


def build_memory_label(dialog: dict) -> str:
    """Build a short label for a memory."""
    return f"D{dialog['dia_id']}|{dialog['date_time']}|{dialog['speaker']}"


def build_memory_content(dialog: dict) -> str:
    """Build full content string for a memory."""
    return f"[{dialog['date_time']}] {dialog['formatted']}"


def chunk_dialogs_by_session(dialogs: list, chunk_size: int = 3) -> list:
    """
    Group dialog turns into chunks of ~chunk_size turns from the same session.
    Each chunk preserves context by combining consecutive turns.
    """
    if not dialogs:
        return []

    chunks = []
    current_session = None
    current_turns = []

    for d in dialogs:
        sid = d['session']
        if sid != current_session and current_turns:
            # Flush previous session's turns as chunks
            for i in range(0, len(current_turns), chunk_size):
                group = current_turns[i:i + chunk_size]
                chunks.append(_merge_dialog_group(group))
            current_turns = []
        current_session = sid
        current_turns.append(d)

    # Flush last session
    if current_turns:
        for i in range(0, len(current_turns), chunk_size):
            group = current_turns[i:i + chunk_size]
            chunks.append(_merge_dialog_group(group))

    return chunks


def _merge_dialog_group(group: list) -> dict:
    """Merge a group of dialog turns into a single memory chunk."""
    first = group[0]
    last = group[-1]

    # Build combined text with all turns
    lines = []
    for d in group:
        lines.append(f"[{d['date_time']}] {d['formatted']}")

    combined_text = "\n".join(lines)

    # Label spans the range
    dia_ids = [str(d['dia_id']) for d in group]
    label = f"D{first['dia_id']}-{last['dia_id']}|{first['date_time']}|{first['speaker']}"

    return {
        'dia_id': first['dia_id'],
        'speaker': first['speaker'],
        'text': combined_text,
        'session': first['session'],
        'date_time': first['date_time'],
        'formatted': combined_text,
        'label': label,
        'num_turns': len(group),
    }

# ---------------------------------------------------------------------------
# Neural Memory mode
# ---------------------------------------------------------------------------

def run_neural_mode(data: dict, llm, args) -> Tuple[List[dict], dict]:
    """
    Ingest conversation into Neural Memory, recall for each QA, generate answers.

    Improvements:
      1. auto_connect=True — graph edges between similar memories
      2. Session-aware chunking — groups 3 turns per chunk (reduces noise)
      3. Graph-traversal — spreading activation expands beyond top-k cosine
    """
    from neural_memory import Memory

    sample_id = data['sample_id']
    dialogs = extract_dialogs(data)
    qa_pairs = data['qa']

    # Fresh memory DB per conversation
    db_path = str(RESULTS_DIR / f"neural_{sample_id}.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    # Chunk dialogs into groups of 3 turns per session
    chunks = chunk_dialogs_by_session(dialogs, chunk_size=args.chunk_size)
    print(f"\n[neural] Sample {sample_id}: {len(dialogs)} dialogs → {len(chunks)} chunks, {len(qa_pairs)} QA pairs")

    # Python backend (C++ SIMD conflicts with PyTorch/sentence-transformers in same process)
    use_cpp = False

    # Ingest with auto_connect=True for graph edges
    t0 = time.time()
    with Memory(db_path=db_path, use_cpp=False, embedding_backend="sentence-transformers") as mem:
        for c in chunks:
            mem.remember(
                text=c['text'],
                label=c['label'],
                auto_connect=True,      # Build graph connections between similar chunks
                detect_conflicts=False, # Skip O(n) conflict detection for benchmarks
            )
        ingest_time = time.time() - t0
        stats = mem.stats()
        print(f"[neural] Ingested {len(chunks)} chunks in {ingest_time:.1f}s "
              f"| backend={mem.backend} | graph: {stats.get('graph_nodes', '?')} nodes, "
              f"{stats.get('graph_edges', '?')} edges")

        # Answer questions
        results = []
        retrieve_times = []

        for i, qa in enumerate(qa_pairs):
            question = qa['question']
            ground_truth = qa.get('answer', qa.get('adversarial_answer', ''))

            # Modify question for category 2 (temporal)
            if qa['category'] == 2:
                question += " Use approximate date to answer."

            # Modify question for category 5 (adversarial)
            if qa['category'] == 5:
                question += " Select the correct answer: (a) Not mentioned in the conversation (b) " + ground_truth

            # === PHASE 1: Cosine similarity recall ===
            t1 = time.time()
            recalled = mem.recall(question, k=args.top_k)

            # === PHASE 2: Multi-hop graph-traversal expansion ===
            # Use built-in recall_multihop: direct cosine + spreading activation
            # Returns up to k*2 results, re-ranked by combined similarity + activation
            try:
                multihop = mem.recall_multihop(question, k=args.top_k, hops=2)
            except Exception:
                multihop = []

            retrieve_times.append(time.time() - t1)

            # Build context: direct recall first, then multihop expansion
            context_parts = []
            seen = set()
            # Direct recall (highest relevance)
            for r in recalled:
                content = r.get('content', '')
                if content and content not in seen:
                    context_parts.append(content)
                    seen.add(content)
            # Multi-hop expansion (graph-connected memories)
            for mh in multihop:
                content = mh.get('content', '')
                if content and content not in seen:
                    context_parts.append(content)
                    seen.add(content)

            context = "\n".join(context_parts)
            total_recalled = len(context_parts)  # Total unique memories used

            # Generate answer with category-specific prompts
            if qa['category'] == 2:
                # Temporal prompt
                prompt = f"""Based on these dated conversation excerpts, determine when the event occurred. Use dates from the context.

Conversation excerpts:
{context}

Question: {question}

Provide the approximate date/time period from the conversation.
Answer:"""
            elif qa['category'] == 5:
                # Adversarial prompt
                prompt = f"""Based on these conversation excerpts, determine if the question can be answered.

Conversation excerpts:
{context}

Question: {question}

If the answer IS mentioned in the excerpts, provide it. If NOT mentioned, respond "Not mentioned in the conversation".
Answer:"""
            else:
                # General extraction prompt (categories 1, 3, 4)
                prompt = f"""You are a precise memory retrieval system. Read the conversation excerpts below and answer the question using EXACT words from the context. Do not paraphrase or infer.

Conversation excerpts:
{context}

Question: {question}

Instructions:
- Extract the exact answer from the excerpts above
- Use the exact words/phrases from the conversation, do not rephrase
- If the answer requires combining information from multiple excerpts, connect them with minimal words
- Answer concisely (max 10 words)
- If the answer is not found, say "Not mentioned in the conversation"

Answer:"""

            prediction = llm.complete(prompt, max_tokens=128)

            # Score
            f1 = score_qa(prediction, ground_truth, qa['category'])

            results.append({
                'question': question,
                'answer': ground_truth,
                'category': qa['category'],
                'prediction': prediction,
                'f1': round(f1, 3),
                'evidence': qa.get('evidence', []),
                'num_recalled': total_recalled,
            })

            if (i + 1) % 25 == 0:
                avg_f1 = np.mean([r['f1'] for r in results])
                print(f"  [{i+1}/{len(qa_pairs)}] avg F1 so far: {avg_f1:.3f}")

    avg_retrieve = np.mean(retrieve_times) * 1000 if retrieve_times else 0
    print(f"[neural] Avg retrieve: {avg_retrieve:.1f}ms")

    meta = {
        'ingest_time_s': round(ingest_time, 1),
        'num_dialogs': len(dialogs),
        'num_qa': len(qa_pairs),
        'avg_retrieve_ms': round(avg_retrieve, 1),
        'backend': 'cpp',
    }
    return results, meta

# ---------------------------------------------------------------------------
# Baseline mode (no memory — truncate conversation to fit)
# ---------------------------------------------------------------------------

def run_baseline_mode(data: dict, llm, args) -> Tuple[List[dict], dict]:
    """
    Baseline: give LLM truncated conversation context (like LoCoMo's non-RAG mode).
    """
    sample_id = data['sample_id']
    conv = data['conversation']
    qa_pairs = data['qa']

    print(f"\n[baseline] Sample {sample_id}: {len(qa_pairs)} QA pairs (truncated context)")

    # Build conversation context (reverse chronological, truncate to ~8k tokens worth of chars)
    session_nums = sorted([int(k.split('_')[-1]) for k in conv.keys()
                           if k.startswith('session_') and not k.endswith(('_date_time',))],
                          reverse=True)

    MAX_CONTEXT_CHARS = 30000  # rough heuristic
    context_parts = []
    total_chars = 0

    for sid in session_nums:
        session_key = f'session_{sid}'
        date_key = f'session_{sid}_date_time'
        if session_key not in conv:
            continue
        date_time = conv.get(date_key, "")
        session_text = f"\n--- {date_time} ---\n"
        for turn in conv[session_key]:
            session_text += f'{turn["speaker"]}: {turn["text"]}\n'

        if total_chars + len(session_text) > MAX_CONTEXT_CHARS:
            break
        context_parts.insert(0, session_text)
        total_chars += len(session_text)

    full_context = "".join(context_parts)
    speakers = list(set([d['speaker'] for d in conv.get(f'session_{session_nums[-1]}', [])]))
    speaker_str = " and ".join(speakers) if speakers else "two people"

    start_prompt = f"Below is a conversation between {speaker_str}. The conversation takes place over multiple days.\n"

    results = []
    for i, qa in enumerate(qa_pairs):
        ground_truth = qa.get('answer', qa.get('adversarial_answer', ''))

        if qa['category'] == 2:
            prompt = f"{start_prompt}{full_context}\n\nQuestion: {qa['question']} Use DATE of CONVERSATION to answer with an approximate date.\nShort answer:"
        elif qa['category'] == 5:
            prompt = f"{start_prompt}{full_context}\n\nQuestion: {qa['question']} Select the correct answer: (a) Not mentioned in the conversation (b) {ground_truth}\nAnswer:"
        else:
            prompt = f"{start_prompt}{full_context}\n\nBased on the above context, write an answer in the form of a short phrase. Answer with exact words from the context whenever possible.\n\nQuestion: {qa['question']}\nShort answer:"

        prediction = llm.complete(prompt, max_tokens=64)
        f1 = score_qa(prediction, ground_truth, qa['category'])

        results.append({
            'question': qa['question'],
            'answer': ground_truth,
            'category': qa['category'],
            'prediction': prediction,
            'f1': round(f1, 3),
            'evidence': qa.get('evidence', []),
        })

        if (i + 1) % 25 == 0:
            avg_f1 = np.mean([r['f1'] for r in results])
            print(f"  [{i+1}/{len(qa_pairs)}] avg F1 so far: {avg_f1:.3f}")

    meta = {
        'context_chars': total_chars,
        'num_qa': len(qa_pairs),
        'mode': 'baseline_truncated',
    }
    return results, meta

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(all_results: Dict[str, List[dict]], mode: str):
    """Print results table by category."""
    # Aggregate across conversations
    by_cat = {}
    for sample_id, results in all_results.items():
        for r in results:
            c = r['category']
            if c not in by_cat:
                by_cat[c] = []
            by_cat[c].append(r['f1'])

    print(f"\n{'='*70}")
    print(f"  LoCoMo Benchmark Results — mode={mode}")
    print(f"{'='*70}")
    print(f"{'Category':<30} {'Count':>6} {'Avg F1':>8}")
    print(f"{'-'*50}")

    total_f1 = 0
    total_count = 0
    cat_keys = sorted(by_cat.keys())
    for c in cat_keys:
        scores = by_cat[c]
        avg = np.mean(scores)
        name = QA_CATEGORIES.get(c, f"cat-{c}")
        print(f"  {c}: {name:<25} {len(scores):>6} {avg:>8.3f}")
        total_f1 += sum(scores)
        total_count += len(scores)

    overall = total_f1 / total_count if total_count > 0 else 0
    print(f"{'-'*50}")
    print(f"  {'OVERALL':<30} {total_count:>6} {overall:>8.3f}")
    print(f"{'='*70}")

    return overall

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LoCoMo Neural Memory Benchmark")
    parser.add_argument('--mode', choices=['neural', 'baseline'], default='neural',
                        help='Evaluation mode')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Number of memories to recall (neural mode)')
    parser.add_argument('--conversations', type=int, default=0,
                        help='Number of conversations to evaluate (0 = all)')
    parser.add_argument('--sample-id', type=str, default=None,
                        help='Evaluate specific sample_id only')
    parser.add_argument('--data-file', type=str, default=str(LOCOMO_DATA))
    parser.add_argument('--llm', choices=['nim', 'openrouter', 'ollama'], default='nim')
    parser.add_argument('--chunk-size', type=int, default=3,
                        help='Dialog turns per chunk (default: 3)')
    parser.add_argument('--out-file', type=str, default=None,
                        help='Output JSON file (default: auto)')
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    with open(args.data_file) as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} conversations from {args.data_file}")

    # Filter
    if args.sample_id:
        samples = [s for s in samples if s['sample_id'] == args.sample_id]
        if not samples:
            print(f"Sample {args.sample_id} not found!")
            return
    elif args.conversations > 0:
        samples = samples[:args.conversations]

    # LLM
    llm = get_llm(args)
    llm_name = get_llm_name(args)
    print(f"LLM: {llm_name}")

    # Run
    all_results = {}
    all_meta = {}

    for sample in samples:
        sid = sample['sample_id']
        if args.mode == 'neural':
            results, meta = run_neural_mode(sample, llm, args)
        else:
            results, meta = run_baseline_mode(sample, llm, args)
        all_results[sid] = results
        all_meta[sid] = meta

    # Report
    overall = print_report(all_results, args.mode)

    # Save
    out_file = args.out_file or str(RESULTS_DIR / f"locomo_{args.mode}_results.json")
    output = {
        'mode': args.mode,
        'llm': llm_name,
        'top_k': args.top_k if args.mode == 'neural' else None,
        'overall_f1': round(overall, 3),
        'conversations': {
            sid: {
                'results': results,
                'meta': all_meta[sid],
            }
            for sid, results in all_results.items()
        }
    }
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == '__main__':
    main()
