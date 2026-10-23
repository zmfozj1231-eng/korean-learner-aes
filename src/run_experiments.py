"""한국어 학습자 작문의 숙달도(TOPIK 1-6급) 자동 예측 실험.

E1. 모델 비교 (길이 baseline / 언어 특징 / 문자·형태소 n-gram / 결합) — 학습자 단위 GroupKFold
E2. 데이터 누수: 무작위 분할 vs 학습자 단위 분할
E3. 문장 -> 글: 같은 표본(sampleID)의 문장 k개를 평균했을 때 성능 변화
E4. 모어(L1)별 편향: 같은 급인데 모어에 따라 과대/과소 예측되는가
E5. 급수에 따라 변하는 언어 특징 (Spearman)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import cohen_kappa_score, mean_absolute_error
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from features import FeatureExtractor, FEATURE_NAMES, build_sentence

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw_sentences.jsonl"
OUT = ROOT / "results"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
SEED = 42
N_FOLDS = 5


def metrics(y, p):
    pr = np.clip(np.rint(p), 1, 6).astype(int)
    return {
        "QWK": cohen_kappa_score(y, pr, weights="quadratic"),
        "Acc": float(np.mean(pr == y)),
        "Adj±1": float(np.mean(np.abs(pr - y) <= 1)),
        "MAE": mean_absolute_error(y, p),
        "Spearman": spearmanr(y, p).correlation,
    }


def load():
    rows = [json.loads(l) for l in DATA.open(encoding="utf-8")]
    df = pd.DataFrame(rows)
    df["text"] = [build_sentence(a, b, c) for a, b, c in zip(df.left, df.center, df.right)]
    df = df[df.text.str.len() >= 5].drop_duplicates("text").reset_index(drop=True)
    df["learner"] = df.userSeq.fillna(df.sampleID.astype(str))
    return df


def cv_predict(make_X, y, splitter, groups=None, model="ridge"):
    """make_X(train_idx, test_idx) -> (Xtr, Xte). 폴드마다 특징 변환기를 새로 학습해 누수를 막는다."""
    pred = np.zeros(len(y))
    for tr, te in splitter.split(np.zeros(len(y)), y, groups):
        Xtr, Xte = make_X(tr, te)
        if model == "ridge":
            m = Ridge(alpha=1.0)
        else:
            m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=SEED)
        m.fit(Xtr, y[tr])
        pred[te] = m.predict(Xte)
    return pred


def main():
    df = load()
    y = df.grade.values.astype(int)
    texts = df.text.tolist()
    groups = df.learner.values
    print("sentences", len(df), "learners", df.learner.nunique(), "samples", df.sampleID.nunique())

    # ---------- 데이터 통계 ----------
    stats = {
        "n_sent": len(df), "n_learner": int(df.learner.nunique()), "n_sample": int(df.sampleID.nunique()),
        "n_L1": int(df.language.nunique()),
        "grade_counts": df.grade.value_counts().sort_index().to_dict(),
        "top_L1": df.language.value_counts().head(10).to_dict(),
        "env": df.environment.value_counts().to_dict(),
    }

    # ---------- 특징 추출 ----------
    # 어휘 희귀도 특징(mean_word_rank)은 라벨을 쓰지 않는 빈도 통계이므로 전체 코퍼스 기준으로 계산
    fe = FeatureExtractor(texts_for_freq=texts)
    F = fe.transform(texts)
    morph = fe.morph_strings(texts)
    feat_df = pd.DataFrame(F, columns=FEATURE_NAMES)
    feat_df["grade"] = y
    feat_df["language"] = df.language.values
    feat_df.to_csv(OUT / "features.csv", index=False, encoding="utf-8-sig")

    def X_len(tr, te):
        L = F[:, [FEATURE_NAMES.index("n_eojeol"), FEATURE_NAMES.index("n_char")]]
        return L[tr], L[te]

    def X_feat(tr, te):
        s = StandardScaler().fit(F[tr])
        return s.transform(F[tr]), s.transform(F[te])

    def X_ngram(tr, te):
        v1 = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), min_df=2, sublinear_tf=True)
        v2 = TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        a = [texts[i] for i in tr]; b = [texts[i] for i in te]
        ma = [morph[i] for i in tr]; mb = [morph[i] for i in te]
        return (hstack([v1.fit_transform(a), v2.fit_transform(ma)]).tocsr(),
                hstack([v1.transform(b), v2.transform(mb)]).tocsr())

    def X_all(tr, te):
        A, B = X_ngram(tr, te)
        C, D = X_feat(tr, te)
        return hstack([A, csr_matrix(C)]).tocsr(), hstack([B, csr_matrix(D)]).tocsr()

    gkf = GroupKFold(n_splits=N_FOLDS)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # ---------- E1 ----------
    configs = [
        ("길이 baseline (Ridge)", X_len, "ridge"),
        ("언어 특징 25개 (Ridge)", X_feat, "ridge"),
        ("언어 특징 25개 (GBM)", X_feat, "gbm"),
        ("문자+형태소 n-gram (Ridge)", X_ngram, "ridge"),
        ("n-gram + 언어 특징 (Ridge)", X_all, "ridge"),
    ]
    res, preds = [], {}
    for name, fx, mdl in configs:
        p = cv_predict(fx, y, gkf, groups, mdl)
        preds[name] = p
        res.append({"model": name, "split": "학습자 단위", **metrics(y, p)})
        print(name, res[-1])
    best_name = max(res, key=lambda r: r["QWK"])["model"]
    best = preds[best_name]

    # ---------- E2 누수 ----------
    for name, fx, mdl in [configs[1], configs[3], configs[4]]:
        p = cv_predict(fx, y, kf, None, mdl)
        res.append({"model": name, "split": "무작위(문장)", **metrics(y, p)})
        print("random", name, res[-1])
    res_df = pd.DataFrame(res)
    res_df.to_csv(OUT / "e1_e2_results.csv", index=False, encoding="utf-8-sig")

    # ---------- E3 문장 -> 글 ----------
    df["pred"] = best
    rng = np.random.default_rng(SEED)
    # 같은 글 집합(문장 4개 이상 수집된 표본)에서 k만 바꿔 비교해야 공정하다. 무작위 추출 20회 평균.
    e3 = []
    K_MAX = 4
    multi = [g for _, g in df.groupby("sampleID") if len(g) >= K_MAX]
    yd = np.array([g.grade.iloc[0] for g in multi])
    for k in range(1, K_MAX + 1):
        ms = []
        for _ in range(20):
            pd_ = np.array([g.pred.values[rng.choice(len(g), size=k, replace=False)].mean() for g in multi])
            ms.append(metrics(yd, pd_))
        e3.append({"k": k, "n_docs": len(multi), **pd.DataFrame(ms).mean().to_dict()})
    e3_df = pd.DataFrame(e3)
    e3_df.to_csv(OUT / "e3_doc_aggregation.csv", index=False, encoding="utf-8-sig")
    print(e3_df)

    # ---------- E4 L1 편향 ----------
    df["err"] = df.pred - df.grade
    top_l1 = df.language.value_counts()
    top_l1 = top_l1[top_l1 >= 150].index.tolist()
    rows = []
    for l1 in top_l1:
        s = df[df.language == l1]
        # 급수 분포 차이를 통제: 급별 평균 오차를 전체 급 분포로 가중 평균
        by_g = s.groupby("grade").err.mean()
        w = df.grade.value_counts(normalize=True)
        adj = float((by_g * w.reindex(by_g.index)).sum() / w.reindex(by_g.index).sum())
        rows.append({"L1": l1, "n": len(s), "mean_grade": s.grade.mean(),
                     "bias_raw": s.err.mean(), "bias_grade_adj": adj,
                     "QWK": cohen_kappa_score(s.grade, np.clip(np.rint(s.pred), 1, 6).astype(int), weights="quadratic")})
    e4_df = pd.DataFrame(rows).sort_values("bias_grade_adj")
    e4_df.to_csv(OUT / "e4_l1_bias.csv", index=False, encoding="utf-8-sig")
    print(e4_df)

    # ---------- E4b 학습 환경(국내/국외)별 편향 ----------
    w = df.grade.value_counts(normalize=True)
    env_rows = []
    for env, s in df.groupby("environment"):
        for g, sg in s.groupby("grade"):
            env_rows.append({"environment": env, "grade": g, "n": len(sg), "bias": sg.err.mean()})
    env_df = pd.DataFrame(env_rows)
    env_df.to_csv(OUT / "e4b_env_bias.csv", index=False, encoding="utf-8-sig")
    df[["sampleID", "grade", "pred", "language", "environment", "text"]].to_csv(
        OUT / "predictions.csv", index=False, encoding="utf-8-sig")

    # ---------- E6 처음 보는 모어 (leave-one-L1-out) ----------
    # 같은 모어 학습자의 글을 학습에 넣었을 때(E1 CV 예측) vs 완전히 빼고 학습했을 때 비교
    best_fx = dict((n, f) for n, f, _ in configs)[best_name]
    e6 = []
    for l1 in top_l1:
        te = np.where(df.language.values == l1)[0]
        tr = np.where(df.language.values != l1)[0]
        Xtr, Xte = best_fx(tr, te)
        p = Ridge(alpha=1.0).fit(Xtr, y[tr]).predict(Xte)
        seen_m = metrics(y[te], best[te]); unseen_m = metrics(y[te], p)
        e6.append({"L1": l1, "n": len(te), "QWK_seen": seen_m["QWK"], "QWK_unseen": unseen_m["QWK"],
                   "MAE_seen": seen_m["MAE"], "MAE_unseen": unseen_m["MAE"],
                   "bias_seen": float(np.mean(best[te] - y[te])), "bias_unseen": float(np.mean(p - y[te]))})
    e6_df = pd.DataFrame(e6).sort_values("n", ascending=False)
    e6_df.to_csv(OUT / "e6_unseen_l1.csv", index=False, encoding="utf-8-sig")
    print(e6_df)

    # 모어 자체를 텍스트에서 맞힐 수 있는가 (급수 예측에 섞여 들어갈 수 있는 신호의 크기)
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    mask = df.language.isin(top_l1).values
    idx = np.where(mask)[0]
    l1_lab = df.language.values[idx]
    l1_pred = np.empty(len(idx), dtype=object)
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(idx, l1_lab, groups[idx]):
        Xtr, Xte = X_ngram(idx[tr], idx[te])
        l1_pred[te] = LogisticRegression(max_iter=2000, C=5).fit(Xtr, l1_lab[tr]).predict(Xte)
    l1_id = {"n_L1": len(top_l1), "acc": accuracy_score(l1_lab, l1_pred),
             "macroF1": f1_score(l1_lab, l1_pred, average="macro"),
             "majority": pd.Series(l1_lab).value_counts(normalize=True).iloc[0]}
    print("L1 identification", l1_id)

    # ---------- E5 특징 상관 ----------
    e5 = []
    for i, n in enumerate(FEATURE_NAMES):
        rho = spearmanr(F[:, i], y).correlation
        e5.append({"feature": n, "rho": rho, **{f"mean_g{g}": F[y == g, i].mean() for g in range(1, 7)}})
    e5_df = pd.DataFrame(e5).sort_values("rho", key=np.abs, ascending=False)
    e5_df.to_csv(OUT / "e5_feature_corr.csv", index=False, encoding="utf-8-sig")

    # 오분류 예시
    df["abs_err"] = df.err.abs()
    df.sort_values("abs_err", ascending=False).head(30)[["grade", "pred", "language", "text"]] \
        .to_csv(OUT / "worst_errors.csv", index=False, encoding="utf-8-sig")

    # 혼동행렬
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, np.clip(np.rint(best), 1, 6).astype(int), labels=range(1, 7))

    json.dump({"stats": stats, "best_model": best_name, "l1_identification": l1_id},
              open(OUT / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=float)
    make_figures(df, res_df, e3_df, e4_df, e5_df, cm, stats, e6_df, env_df)


def make_figures(df, res_df, e3_df, e4_df, e5_df, cm, stats, e6_df, env_df):
    C = ["#3b6ea8", "#e07b39", "#7aa95c", "#b05a8c", "#8c8c8c"]

    # Fig8 국내/국외 급별 편향
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for i, (env, s) in enumerate(env_df.groupby("environment")):
        ax.plot(s.grade, s.bias, "o-", color=C[i], label=env)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("실제 급"); ax.set_ylabel("평균 오차 (예측-실제)"); ax.legend()
    ax.set_title("학습 환경(국내·국외)별 급별 예측 편향")
    fig.tight_layout(); fig.savefig(FIG / "fig8_env_bias.png", dpi=200); plt.close(fig)

    # Fig1 데이터 분포
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    gc = pd.Series(stats["grade_counts"]).sort_index()
    ax[0].bar([f"{g}급" for g in gc.index], gc.values, color=C[0])
    ax[0].set_title("급수별 문장 수"); ax[0].set_ylabel("문장 수")
    tl = pd.Series(stats["top_L1"]).head(8)[::-1]
    ax[1].barh(tl.index, tl.values, color=C[1]); ax[1].set_title("모어 상위 8개")
    fig.tight_layout(); fig.savefig(FIG / "fig1_data.png", dpi=200); plt.close(fig)

    # Fig2 모델 비교 + 누수
    g = res_df[res_df.split == "학습자 단위"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.barh(g.model[::-1], g.QWK[::-1], color=C[0], label="학습자 단위 분할")
    r = res_df[res_df.split != "학습자 단위"].set_index("model").QWK
    ys = list(g.model[::-1])
    for i, m in enumerate(ys):
        if m in r:
            ax.plot(r[m], i, "D", color=C[1], ms=8, label="무작위 분할" if i == ys.index([x for x in ys if x in r][0]) else None)
    for i, v in enumerate(g.QWK[::-1]):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("QWK"); ax.set_xlim(0, 1); ax.legend(loc="lower right")
    ax.set_title("모델별 문장 단위 급수 예측 성능 (5-fold CV)")
    fig.tight_layout(); fig.savefig(FIG / "fig2_models.png", dpi=200); plt.close(fig)

    # Fig3 혼동행렬
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    cmn = cm / cm.sum(1, keepdims=True)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xticks(range(6), [f"{i}" for i in range(1, 7)]); ax.set_yticks(range(6), [f"{i}" for i in range(1, 7)])
    ax.set_xlabel("예측 급"); ax.set_ylabel("실제 급"); ax.set_title("혼동행렬 (행 정규화)")
    fig.tight_layout(); fig.savefig(FIG / "fig3_confusion.png", dpi=200); plt.close(fig)

    # Fig4 문장 수 k
    if len(e3_df):
        fig, ax = plt.subplots(figsize=(5, 3.3))
        ax.plot(e3_df.k, e3_df.QWK, "o-", color=C[0], label="QWK")
        ax.plot(e3_df.k, e3_df["Adj±1"], "s--", color=C[2], label="±1 정확도")
        ax.set_xlabel("한 글에서 사용한 문장 수 k"); ax.set_ylim(0, 1); ax.legend()
        ax.set_title("문장을 모을수록 급수 판단이 정확해지는가")
        fig.tight_layout(); fig.savefig(FIG / "fig4_k_sentences.png", dpi=200); plt.close(fig)

    # Fig5 L1 편향
    fig, ax = plt.subplots(figsize=(6, 0.35 * len(e4_df) + 1.2))
    cols = [C[1] if v > 0 else C[0] for v in e4_df.bias_grade_adj]
    ax.barh(e4_df.L1, e4_df.bias_grade_adj, color=cols)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("평균 예측 오차 (예측-실제, 급 분포 보정)")
    ax.set_title("모어별 편향: + 과대평가 / − 과소평가")
    fig.tight_layout(); fig.savefig(FIG / "fig5_l1_bias.png", dpi=200); plt.close(fig)

    # Fig7 처음 보는 모어
    d = e6_df.sort_values("QWK_seen")
    fig, ax = plt.subplots(figsize=(6, 0.38 * len(d) + 1.3))
    yy = np.arange(len(d))
    ax.hlines(yy, d.QWK_unseen, d.QWK_seen, color="#bbbbbb", lw=2)
    ax.plot(d.QWK_seen, yy, "o", color=C[0], label="학습에 같은 모어 포함")
    ax.plot(d.QWK_unseen, yy, "o", color=C[1], label="학습에서 그 모어 제외")
    ax.set_yticks(yy, d.L1); ax.set_xlabel("QWK"); ax.legend(loc="lower right", fontsize=8)
    ax.set_title("처음 보는 모어의 학습자도 채점할 수 있는가")
    fig.tight_layout(); fig.savefig(FIG / "fig7_unseen_l1.png", dpi=200); plt.close(fig)

    # Fig6 특징 상관
    top = e5_df.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(top.feature, top.rho, color=[C[0] if v > 0 else C[1] for v in top.rho])
    ax.axvline(0, color="k", lw=0.8); ax.set_xlabel("급수와의 Spearman ρ")
    ax.set_title("급이 오를수록 달라지는 언어 특징 (상위 12)")
    fig.tight_layout(); fig.savefig(FIG / "fig6_features.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
