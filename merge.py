import os
import jsonlines

# 경로 설정
query_dir = "data/qvhighlights/query"
val_file = os.path.join(query_dir, "val.jsonl")
refined_file = os.path.join(query_dir, "val_semantic.jsonl")


def main():
    # 1. refined 결과에서 실제로 재작성된 것만 추출
    refined_dict = {}
    skipped = 0

    with jsonlines.open(refined_file, mode="r") as reader:
        for item in reader:
            qid = item.get("qid")
            action = item.get("action", "")
            refined_query = item.get("refined_query", "")
            original_query = item.get("original_query", "")

            if action == "rewrite" and refined_query != original_query and refined_query:
                refined_dict[qid] = refined_query
            else:
                skipped += 1

    print(f"재작성된 쿼리: {len(refined_dict)}개")
    print(f"원본 유지 (skip): {skipped}개")

    # 2. val.jsonl 읽어서 rewrite 케이스만 정제쿼리 1개로 교체
    new_data = []
    replaced_count = 0

    with jsonlines.open(val_file, mode="r") as reader:
        for line in reader:
            qid = line["qid"]
            if qid in refined_dict:
                line["rephrased_query"] = [refined_dict[qid]]  # 정제쿼리 1개만
                replaced_count += 1
            new_data.append(line)

    with jsonlines.open(val_file, mode="w") as writer:
        writer.write_all(new_data)

    print(f"총 {replaced_count}개 rephrased_query 교체 완료!")
    print(f"나머지 {len(new_data) - replaced_count}개는 원본 유지")
    print(f"이제 VTG-GPT 재추론 실행하면 돼!")


if __name__ == "__main__":
    main()