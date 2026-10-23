"""Kiwi 형태소 분석 기반 언어적 특징(1세대 AES 방식의 표면/문법 특징)."""
import math
import re
from collections import Counter

import numpy as np
from kiwipiepy import Kiwi

NO_CTX = re.compile(r"\{[좌우] 문맥 없음\}")

FEATURE_NAMES = [
    "n_char", "n_eojeol", "n_morph", "mean_eojeol_len", "morph_per_eojeol",
    "ttr_morph", "content_ratio",
    "r_josa", "josa_types", "r_EC", "r_ETM", "r_ETN", "r_EP", "r_VX", "r_NNB", "r_MAJ", "r_XSN_XSV",
    "end_hamnida", "end_haeyo", "end_handa",
    "r_unknown", "r_foreign", "r_SH", "n_clause",
    "mean_word_rank",
]


def build_sentence(left, center, right):
    parts = [NO_CTX.sub("", p or "").strip() for p in (left, center, right)]
    return " ".join(p for p in parts if p)


class FeatureExtractor:
    def __init__(self, texts_for_freq=None):
        self.kiwi = Kiwi()
        self.rank = {}
        if texts_for_freq is not None:
            # 학습 데이터 내 형태소 빈도 순위(낮을수록 흔한 단어) -> 어휘 수준(희귀어 사용) 근사
            cnt = Counter()
            for toks in self.tokenize_many(texts_for_freq):
                cnt.update(t.form for t in toks if t.tag.startswith(("NN", "VV", "VA", "MA")))
            self.rank = {w: i + 1 for i, (w, _) in enumerate(cnt.most_common())}

    def tokenize_many(self, texts):
        return [r[0][0] for r in self.kiwi.analyze(list(texts), top_n=1)]

    def morph_strings(self, texts):
        return [" ".join(f"{t.form}/{t.tag}" for t in toks) for toks in self.tokenize_many(texts)]

    def _feat(self, text, toks):
        eojeols = text.split()
        n_e = max(len(eojeols), 1)
        n_m = max(len(toks), 1)
        tags = [t.tag for t in toks]
        forms = [t.form for t in toks]
        c = Counter(tags)

        def r(*prefixes):
            return sum(v for k, v in c.items() if k.startswith(prefixes)) / n_m

        content = [t for t in toks if t.tag.startswith(("NN", "VV", "VA", "MA", "XR"))]
        josa = [t.form for t in toks if t.tag.startswith("J")]
        ef = [t.form for t in toks if t.tag == "EF"]
        last_ef = ef[-1] if ef else ""
        ranks = [self.rank.get(t.form, len(self.rank) + 1) for t in content]
        return [
            len(text), n_e, n_m, len(text.replace(" ", "")) / n_e, n_m / n_e,
            len(set(forms)) / n_m, len(content) / n_m,
            r("J"), len(set(josa)), r("EC"), r("ETM"), r("ETN"), r("EP"), r("VX"), r("NNB"), r("MAJ"), r("XSN", "XSV", "XSA"),
            float(last_ef.endswith(("니다", "니까", "ㅂ니다", "습니다"))),
            float(last_ef.endswith(("요", "어요", "아요", "에요", "예요", "죠"))),
            float(last_ef in ("다", "는다", "ㄴ다", "었다", "었다", "라", "자")),
            r("UN"), r("SL"), r("SH"), c.get("EC", 0) + c.get("ETM", 0) + 1,
            float(np.mean(np.log1p(ranks))) if ranks else 0.0,
        ]

    def transform(self, texts):
        toks_all = self.tokenize_many(texts)
        return np.array([self._feat(t, toks) for t, toks in zip(texts, toks_all)], dtype=float)
