# Graduation Project

## 실행 순서

### 환경 설정
```bash
conda create -n vtg-gpt python=3.10
conda activate vtg-gpt
pip install -r requirements.txt
```

### 캡션 파일 압축 해제
```bash
cd data/qvhighlights/caption/
unzip val.zip
```

### 실행
```bash
# 1. 정제 쿼리 생성
python Baichuan2/refine_query_semantic.py

# 2. 베이스라인 쿼리 복원
cp data/qvhighlights/query/val_7B.jsonl data/qvhighlights/query/val.jsonl

# 3. 머지
python merge.py

# 4. 추론
python infer_qvhighlights.py val

# 5. 평가
bash standalone_eval/eval.sh

# 6. 분석
python analysis.py

# 7. 시각화
python visualize.py
```



## 연구 소개

VTG-GPT 기반 Zero-shot Video Moment Retrieval 성능 향상 연구.
Top-1 예측 구간의 캡션을 피드백으로 활용하는 쿼리 정제 파이프라인을 설계하여 추가 학습 없이 성능을 개선하였다.



## 연구 배경

Video Moment Retrieval(VMR)은 자연어 쿼리가 주어졌을 때, 비디오에서 해당 내용이 등장하는 시간 구간을 찾는 문제이다.

기존 VTG-GPT는 원본 쿼리와 paraphrased 쿼리를 사용하여 구간을 추론하지만, 모델이 쿼리의 핵심 행위를 찾지 못하는 경우가 존재한다. 예를 들어 "Woman holds up beverages in a car" 쿼리에서 "차 안에 여자가 있는 장면"은 찾지만 "음료를 들고 있는 구체적인 행위"는 찾지 못하는 경우가 있다.



## 제안 방법

무조건적인 쿼리 재작성은 오히려 성능을 저하시킨다는 문제를 발견하여, LLM 개입 여부를 먼저 판단하는 **Semantic Similarity Gate**를 도입하였다.

### 전체 파이프라인

1. **초기 검색**: 원본 쿼리로 VTG-GPT 추론 → Top-1 구간 획득
2. **Semantic Similarity Gate**: `all-MiniLM-L6-v2`로 원본 쿼리 ↔ Top-1 캡션 유사도 계산
   - HIGH (≥ 0.7): 원본 쿼리 유지
   - MID (0.4 ~ 0.7): Baichuan2-7B-Chat으로 쿼리 재작성
   - LOW (≤ 0.4): 원본 쿼리 유지 (캡션이 쿼리와 무관하여 환각 위험)
3. **최종 재검색**: 정제된 쿼리로 VTG-GPT 재추론 → 최종 구간 출력

### 쿼리 정제 프롬프트

Few-shot Chain-of-Thought 방식으로 캡션의 시각적 정보를 쿼리에 반영하도록 설계하였다.



## 실험 결과

### Baseline vs Ours

| 평가지표 | 7B 베이스라인 | 우리의 실험 |
|---|---|---|
| MR-full-mAP | 33.02 | **33.19 (+0.17)** |
| MR-long-mAP | 33.53 | 33.48 (-0.05) |
| MR-middle-mAP | 37.83 | **38.25 (+0.42)** |
| MR-short-mAP | 3.94 | 3.91 (-0.03) |
| MR-full-R1@0.5 | 58.39 | 57.87 (-0.52) |

### 캡션 수 최적화 실험 (full-mAP)

| 캡션 1개 | 캡션 2개 | 캡션 3개 | 캡션 4개 |
|---|---|---|---|
| 33.07 | 33.22 | **33.39** | 33.29 |

베이스라인(33.02) 대비 캡션 3개 사용 시 최고 성능 달성 (+0.37)