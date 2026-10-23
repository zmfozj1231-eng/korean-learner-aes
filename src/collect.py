"""국립국어원 한국어 학습자 말뭉치 나눔터(kcorpus.korean.go.kr)의 공개 '원시 말뭉치 검색'에서
문어(쓰기) 표본 문장을 급수별로 수집한다.

검색 결과 한 행 = (모어, 급수, 왼쪽 문맥, 중심어, 오른쪽 문맥) 이고, 문맥을 이어 붙이면 학습자가 쓴 한 문장이 된다.
자주 쓰이는 어절 여러 개로 검색해 급수별로 비슷한 수의 문장을 모으고, (sampleID, sentIndex)로 중복을 제거한다.
"""
import json, sys, time, random
from pathlib import Path
import requests

BASE = "https://kcorpus.korean.go.kr"
GRADES = {"1112": 1, "1113": 2, "1114": 3, "1115": 4, "1116": 5, "1127": 6}
WRITTEN = "806"  # 문어
TERMS = ["있습니다.", "합니다.", "있다.", "했습니다.", "생각합니다.", "그래서", "하지만", "때문에",
         "저는", "나는", "사람들이", "것이", "좋습니다.", "싶습니다.", "한다.", "많이", "우리", "그리고"]
ROW_SIZE = 100
PAGES_PER_QUERY = 3
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw_sentences.jsonl")


def _params(term, grade_code):
    return [("sortCode", "0"), ("searchTerm", term), ("operator", "OR"),
            ("sCommGrade", grade_code), ("sCommSourcetype", WRITTEN)]


def count(sess, term, grade_code):
    r = sess.post(BASE + "/service/rawCorpusSearch-counting.data", data=_params(term, grade_code), timeout=60)
    return int(r.json()["searchRows"][0]["wordCount"])


def query(sess, term, grade_code, page):
    data = _params(term, grade_code) + [("page_no", str(page)), ("row_size", str(ROW_SIZE))]
    r = sess.post(BASE + "/service/rawCorpusSearch-data.data", data=data, timeout=60)
    r.raise_for_status()
    return r.json().get("dataList") or []


def main():
    sess = requests.Session()
    sess.headers["User-Agent"] = "Mozilla/5.0 (course project; SSU NLP)"
    sess.get(BASE + "/service/goRawCorpusSearch.do", timeout=30)
    # 검색 결과는 모어 가나다순으로 정렬되므로 앞 페이지만 읽으면 특정 모어에 쏠린다 -> 페이지를 무작위로 뽑는다
    rng = random.Random(42)
    seen, rows = set(), []
    for gcode, g in GRADES.items():
        for term in TERMS:
            n_pages = max(1, -(-count(sess, term, gcode) // ROW_SIZE))
            for page in rng.sample(range(1, n_pages + 1), min(PAGES_PER_QUERY, n_pages)):
                try:
                    lst = query(sess, term, gcode, page)
                except Exception as e:
                    print("fail", g, term, page, e, flush=True); time.sleep(3); continue
                new = 0
                for d in lst:
                    key = (d["sampleID"], d["sentIndex"])
                    if key in seen or d.get("sourceType") != "문어":
                        continue
                    seen.add(key); new += 1
                    rows.append({"sampleID": d["sampleID"], "sentIndex": d["sentIndex"], "grade": g,
                                 "language": d.get("language"), "nation": d.get("nation"),
                                 "environment": d.get("environment"), "purpose": d.get("learningPurpose"),
                                 "age": d.get("userAge"), "userSeq": d.get("userSeq"),
                                 "left": d.get("leftCon"), "center": d.get("centerWord"), "right": d.get("rightCon")})
                print(f"grade {g} '{term}' p{page}: {len(lst)} rows, {new} new, total {len(rows)}", flush=True)
                time.sleep(0.6 + random.random() * 0.4)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("saved", len(rows), "->", OUT)


if __name__ == "__main__":
    main()
