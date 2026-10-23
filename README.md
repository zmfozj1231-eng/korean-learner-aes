# 한국어 학습자 작문의 숙달도(급수) 자동 예측

숭실대학교 대학원 자연어처리 Term Project #1 (Automated Essay Scoring) — 이재영

데이터: 국립국어원 한국어 학습자 말뭉치 나눔터(https://kcorpus.korean.go.kr) 원시 말뭉치 검색의 문어 표본.
말뭉치 이용 약관상 원문 데이터는 저장소에 포함하지 않으며, `src/collect.py`로 재수집할 수 있다.

## 실행

```bash
pip install -r requirements.txt
python src/collect.py data/raw_sentences.jsonl   # 급수(1-6)별 문어 문장 수집, 약 5분
python src/run_experiments.py                    # 실험 E1-E5, 결과는 results/ 에 저장
```

CPU만으로 전체 실행이 약 10분 내에 끝난다 (GPU 불필요).

## 구성

| 파일 | 내용 |
|---|---|
| `src/collect.py` | 공개 검색 API에서 급수·문어 조건으로 문장 수집, (sampleID, sentIndex) 중복 제거 |
| `src/features.py` | Kiwi 형태소 분석 기반 언어 특징 25개 |
| `src/run_experiments.py` | E1 모델 비교, E2 분할 방식(누수), E3 문장→글 집계, E4 모어별 편향, E5 특징 분석 |
| `results/` | 결과 표(csv)와 그림(figures/) |
