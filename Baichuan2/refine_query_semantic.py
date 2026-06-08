# 최종 코드

import time
import re
import string
from functools import wraps

import jsonlines
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from sentence_transformers import SentenceTransformer, util as st_util


def print_running_time(name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            print(f'{name}: {elapsed_time:.1f}s')
            return result
        return wrapper
    return decorator


@print_running_time("chat")
def chat(prompt, model, tokenizer):
    model.generation_config = GenerationConfig.from_pretrained(model.name_or_path)
    model.generation_config.max_new_tokens = 64
    model.generation_config.temperature = 0.01
    model.generation_config.top_p = 0.8
    model.generation_config.do_sample = False
    model.generation_config.repetition_penalty = 1.05
    messages = [{"role": "user", "content": prompt}]
    response = model.chat(tokenizer, messages)
    return response


def normalize_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_content_words(text):
    stopwords = {
        "a", "an", "the", "is", "are", "was", "were", "be", "being", "been",
        "in", "on", "at", "to", "of", "for", "with", "and", "or", "as",
        "by", "from", "that", "this", "there", "here", "into", "while",
        "person", "someone", "something", "scene", "video", "shows", "showing",
        "some", "front", "near", "next", "around", "together"
    }
    norm = normalize_text(text)
    words = norm.split()
    return [w for w in words if len(w) > 2 and w not in stopwords]


def grounded_ratio(refined_query, source_text):
    refined_words = tokenize_content_words(refined_query)
    if not refined_words:
        return 0.0
    source_words = set(tokenize_content_words(source_text))
    supported = [w for w in refined_words if w in source_words]
    return len(supported) / len(refined_words)


def contains_bad_output(text):
    if not isinstance(text, str):
        return True
    bad_patterns = [
        "please provide", "more specific", "cannot be provided",
        "not provided", "no relevant information", "need more context",
        "clarification", "could not be refined", "refined query",
        "original query",
    ]
    norm = normalize_text(text)
    for p in bad_patterns:
        if p in norm:
            return True
    if re.search(r"[\u4e00-\u9fff]", text):
        return True
    return False


def compute_semantic_similarity(query, caption, sem_model):
    emb_q = sem_model.encode(query, convert_to_tensor=True)
    emb_c = sem_model.encode(caption, convert_to_tensor=True)
    return st_util.pytorch_cos_sim(emb_q, emb_c).item()


CAPTION_DIR = "data/qvhighlights/caption/val"


def load_video_caption_file(vid, caption_cache):
    if vid in caption_cache:
        return caption_cache[vid]
    caption_path = f"{CAPTION_DIR}/{vid}.jsonl"
    moment_to_desc = {}
    try:
        with jsonlines.open(caption_path, mode="r") as reader:
            for line in reader:
                moment = line.get("moment")
                desc = line.get("description", "")
                if moment is not None:
                    moment_to_desc[int(moment)] = desc
    except Exception as e:
        print(f"caption 파일 로드 실패: {caption_path} / {e}")
    caption_cache[vid] = moment_to_desc
    return moment_to_desc


def get_best_caption_sentences(vid, start_idx, end_idx, query, caption_cache, sem_model, top_n=3):
    moment_to_desc = load_video_caption_file(vid, caption_cache)

    descriptions = []
    for moment in range(int(start_idx), int(end_idx) + 1):
        desc = moment_to_desc.get(moment, "").strip()
        if desc:
            descriptions.append((moment, desc))

    if not descriptions:
        return "No caption available."

    seen = set()
    unique = []
    for moment, desc in descriptions:
        norm = normalize_text(desc)
        if norm not in seen:
            unique.append((moment, desc))
            seen.add(norm)

    texts = [desc for _, desc in unique]
    emb_query = sem_model.encode(query, convert_to_tensor=True)
    emb_descs = sem_model.encode(texts, convert_to_tensor=True)
    sims = st_util.pytorch_cos_sim(emb_query, emb_descs)[0]

    top_n_actual = min(top_n, len(unique))
    top_indices = sims.topk(top_n_actual).indices.tolist()
    top_indices_sorted = sorted(top_indices)  # 시간순 유지

    selected = [unique[i][1] for i in top_indices_sorted]
    return " ".join(selected)


def truncate_text(text, max_words=60):
    words = str(text).split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " ..."


def load_top1_caption(filepath, sem_model, gt_jsonl):
    caption_dict = {}
    caption_cache = {}

    query_dict = {str(line.get("qid", line.get("vid"))): line["query"] for line in gt_jsonl}

    try:
        with jsonlines.open(filepath, mode="r") as reader:
            for line in reader:
                qid = str(line.get("qid"))
                vid = line.get("vid")
                if not qid or not vid:
                    continue
                span_list = line.get("span_index_list", [])
                if not span_list:
                    continue
                top1 = span_list[0]
                if len(top1) < 2:
                    continue
                start_idx, end_idx = int(top1[0]), int(top1[1])
                score = float(top1[2]) if len(top1) >= 3 else 0.0

                query = query_dict.get(qid, "")
                caption_text = get_best_caption_sentences(
                    vid, start_idx, end_idx, query, caption_cache, sem_model, top_n=3
                )
                caption_dict[qid] = {
                    "caption": caption_text,
                    "score": score,
                    "span_index": [start_idx, end_idx],
                }
        print(f"Top-1 caption {len(caption_dict)}개 로드 완료")
    except Exception as e:
        print(f"baseline 파일 로드 실패: {e}")
    return caption_dict


HIGH_SIM = 0.7
LOW_SIM  = 0.4
MIN_QUERY_SIM = 0.6


def decide_action(query, caption, sem_model):
    similarity = compute_semantic_similarity(query, caption, sem_model)
    if similarity >= HIGH_SIM:
        return "keep", similarity
    elif similarity <= LOW_SIM:
        return "keep", similarity
    else:
        return "rewrite", similarity


def build_rewrite_prompt(query, caption):
    caption = truncate_text(caption, max_words=90)
    prompt = f"""You are a video retrieval assistant. Rewrite the query to be more visually specific using details from the caption. Keep the original intent. Output one English sentence only.

Example 1:
Query: A girl opening post office mails in a car
Caption: A woman sitting in a car holding up a piece of paper and looking at it while driving.
Rewritten query: A girl sitting in a car, holding up and opening a piece of mail.

Example 2:
Query: Man turns the machine to make spaghetti.
Caption: A person is using a blue pasta machine to make pasta. A close-up of a blue pasta machine with a wooden surface.
Rewritten query: A man turning a blue pasta machine on a wooden surface to make spaghetti.

Now rewrite:
Query: {query}
Caption: {caption}
Rewritten query:"""
    return prompt.strip()


def main():
    MODEL_PATH = "baichuan-inc/Baichuan2-7B-Chat"
    MAX_SAMPLES = 1550
    MIN_GROUNDED_RATIO = 0.65

    print("Loading Semantic Model...")
    sem_model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Loading LLM...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, use_fast=False, trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True,
    )

    gt_jsonl_path = "data/qvhighlights/gt/highlight_val_release.jsonl"
    baseline_path = "outputs/qvhighlights/infer_val_7B.jsonl"
    output_path   = "data/qvhighlights/query/val_semantic.jsonl"

    with jsonlines.open(gt_jsonl_path, mode="r") as reader:
        gt_jsonl = list(reader)

    caption_dict = load_top1_caption(baseline_path, sem_model, gt_jsonl)

    stats = {"keep_high": 0, "keep_low": 0, "rewrite_success": 0, "rewrite_fallback": 0}

    with jsonlines.open(output_path, mode="w") as writer:
        for line in tqdm(gt_jsonl[:MAX_SAMPLES]):
            raw = ""
            query = line["query"]
            qid   = str(line.get("qid", line.get("vid")))

            top1    = caption_dict.get(qid, {})
            caption = top1.get("caption", "")

            action, similarity = decide_action(query, caption, sem_model)

            if action == "keep":
                final_query = query
                reason = "high_sim" if similarity >= HIGH_SIM else "low_sim"
                if similarity >= HIGH_SIM:
                    stats["keep_high"] += 1
                else:
                    stats["keep_low"] += 1

            else:
                prompt   = build_rewrite_prompt(query, caption)
                raw      = chat(prompt, model, tokenizer).strip()

                first_line = ""
                raw_lower = raw.lower()
                if "rewritten query:" in raw_lower:
                    idx = raw_lower.find("rewritten query:")
                    first_line = raw[idx + len("rewritten query:"):].split("\n")[0].strip()
                if not first_line:
                    first_line = raw.split("\n")[-1].strip()

                source_text = f"{query} {caption}"
                g_ratio = grounded_ratio(first_line, source_text)
                query_sim = compute_semantic_similarity(query, first_line, sem_model)

                if contains_bad_output(first_line) or g_ratio < MIN_GROUNDED_RATIO or query_sim < MIN_QUERY_SIM:
                    final_query = query
                    reason = f"rewrite_fallback_grounded_{g_ratio:.2f}_qsim_{query_sim:.2f}"
                    stats["rewrite_fallback"] += 1
                else:
                    final_query = first_line
                    reason = f"rewrite_success_grounded_{g_ratio:.2f}_qsim_{query_sim:.2f}"
                    stats["rewrite_success"] += 1

            line["refined_query"]  = final_query
            line["original_query"] = query
            line["similarity"]     = round(similarity, 3)
            line["action"]         = action
            line["reason"]         = reason
            line["top1_caption"]   = caption

            writer.write(line)

    print(f"\n완료. 저장 위치: {output_path}")
    print(f"   keep (high sim) : {stats['keep_high']}")
    print(f"   keep (low sim)  : {stats['keep_low']}")
    print(f"   rewrite success : {stats['rewrite_success']}")
    print(f"   rewrite fallback: {stats['rewrite_fallback']}")


if __name__ == "__main__":
    main()