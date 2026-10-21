# 개발용 (제출 X). 보고서(report/main.typ)에 들어가는 수치를 전부 다시 계산해
# 화면에 찍고 report/figures/results.json 으로 저장한다 (그림은 make_figures.py).
#   1. 데이터 통계
#   2. Task1 / Task2 / Task3 val warm RMSE (+ 명세 기준 user 199/458/529)
#   3. Task1 변형: Pearson의 평균을 공통 평가 항목 평균으로 둔 경우
#   4. Task2 K sweep (val warm RMSE, train 관측치 재현 RMSE)
#   5. Task3 ablation (train 내부 홀드아웃, val warm) — 제출 설정(seed 42) 그대로
#   6. Task3 하이퍼파라미터: K x reg 격자 + one-at-a-time sweep, seed 0~4 평균.
#      선택은 train 내부 10% 홀드아웃으로만 하고 val은 참고로만 기록 (병렬 실행)
#   7. movie 인기도 구간별 RMSE
# 사용: python experiments.py

import os

# worker 여러 개를 띄우므로 BLAS 스레드는 1개로 (numpy import 전에 설정해야 적용됨)
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import contextlib
import json
import multiprocessing as mp
import time

import numpy as np
import pandas as pd

import main as M

KEY_USERS = (199, 458, 529)
HOLDOUT_FRAC, HOLDOUT_SEED = 0.1, 0
SEEDS = tuple(range(5))
N_WORKERS = min(24, os.cpu_count() or 1)
OUT_JSON = os.path.join(M.BASE_DIR, "report", "figures", "results.json")

GRID_K = (2, 5, 10, 20, 50, 100, 200)
GRID_REG = (0.05, 0.1, 0.2)
SWEEPS = {
    "BIAS_REG": (0.0, 1.0, 3.0, 5.0, 10.0, 25.0),
    "MF_LR": (0.005, 0.01, 0.02),
    "MF_EPOCHS": (20, 40, 60, 100, 150),
}
BLENDS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
PREVIOUS = {"MF_K": 10}              # K 재선택 전 제출 설정 (나머지는 동일)


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


@contextlib.contextmanager
def patched(**params):
    """main.py의 하이퍼파라미터 상수를 잠시 바꿔 fit 함수를 그대로 재사용."""
    old = {k: getattr(M, k) for k in params}
    for k, v in params.items():
        setattr(M, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(M, k, v)


class Eval:
    """ds 기준 warm 행(user, movie 모두 ds에 있음)만 평가."""
    def __init__(self, ds, df):
        w = df[df["user_id"].isin(ds.uid2idx) & df["movie_id"].isin(ds.mid2idx)]
        self.u = w["user_id"].map(ds.uid2idx).to_numpy()
        self.m = w["movie_id"].map(ds.mid2idx).to_numpy()
        self.y = w["rating"].to_numpy()
        self.uid = w["user_id"].to_numpy()

    def score(self, pred):
        return rmse(self.y, pred)

    def per_user(self, pred):
        # 홀드아웃에는 기준 user 행이 없을 수 있음 -> None
        return {str(k): (rmse(self.y[self.uid == k], pred[self.uid == k]) if (self.uid == k).any() else None)
                for k in KEY_USERS}


def cf_matrix(ds, S=None):
    """user_based_cf()와 같은 식을 전체 행렬로 계산 (모든 유사도 0 -> movie 평균)."""
    S = M.pearson_similarity(ds) if S is None else S
    num = S @ ds.D
    den = np.abs(S) @ ds.C
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, ds.user_mean[:, None] + num / den, ds.item_mean[None, :])


def corated_mean_similarity(ds):
    """변형: s(u,v)의 r_bar를 공통 평가 항목 I_uv 위에서의 평균으로 둔 Pearson."""
    R, C = ds.R, ds.C
    N = C @ C.T
    Su = R @ C.T                         # Su[u, v] = sum_{i in I_uv} r_ui
    Suu = (R ** 2) @ C.T
    Suv = R @ R.T
    with np.errstate(divide="ignore", invalid="ignore"):
        cov = Suv - Su * Su.T / N
        var_u = Suu - Su ** 2 / N
        S = np.where((N > 0) & (var_u > 1e-12) & (var_u.T > 1e-12), cov / np.sqrt(var_u * var_u.T), 0.0)
    np.fill_diagonal(S, 0.0)
    return S


def svd_factors(ds):
    F = ds.R * ds.C + ds.item_mean[None, :] * (1 - ds.C)
    return np.linalg.svd(F, full_matrices=False)


def svd_reconstruct(factors, k):
    U, s, Vt = factors
    return (U[:, :k] * s[:k]) @ Vt[:k]


def fit_components(ds, **params):
    with patched(**params):
        mu, bu, bi = M.fit_baseline(ds)
        P, Q = M.fit_latent_factors(ds, mu, bu, bi)
    return mu, bu, bi, P, Q


def task3_preds(ev, comps, cf, blend):
    mu, bu, bi, P, Q = comps
    base = mu + bu[ev.u] + bi[ev.m]
    mf = base + np.einsum("ij,ij->i", P[ev.u], Q[ev.m])
    final = np.clip((1 - blend) * mf + blend * cf[ev.u, ev.m], M.RATING_MIN, M.RATING_MAX)
    return base, mf, final


def load_splits():
    """val 평가용(train 전체로 학습)과 홀드아웃 평가용(train 90%로 학습) 두 벌."""
    train = pd.read_csv(M.TRAIN_FILE)
    val = pd.read_csv(M.VAL_FILE)
    hold = np.random.default_rng(HOLDOUT_SEED).random(len(train)) < HOLDOUT_FRAC
    ds = M.build_dataset(train)
    ds_h = M.build_dataset(train[~hold])
    return {
        "val": (ds, Eval(ds, val), cf_matrix(ds)),
        "holdout": (ds_h, Eval(ds_h, train[hold]), cf_matrix(ds_h)),
    }, train, val, hold


# ---------------- 병렬 하이퍼파라미터 실험 ----------------

_SPLITS = {}


def _init_worker():
    _SPLITS.update(load_splits()[0])


def _run_job(job):
    """(config, split, seed) 하나 학습 -> 블렌딩 가중치별 최종 RMSE."""
    key, params, split, seed = job
    ds, ev, cf = _SPLITS[split]
    comps = fit_components(ds, MF_SEED=seed, **params)
    return job, [ev.score(task3_preds(ev, comps, cf, w)[2]) for w in BLENDS]


def run_configs(configs):
    """configs: {key: params}. seed 0~4 x {holdout, val} 를 병렬로 돌려 평균/표준편차."""
    jobs = [(key, params, split, seed) for key, params in configs.items()
            for split in ("holdout", "val") for seed in SEEDS]
    # 오래 걸리는 작업(K, epoch 큰 것)부터 넣어 부하 균형
    cost = lambda j: j[1].get("MF_K", M.MF_K) * j[1].get("MF_EPOCHS", M.MF_EPOCHS)
    jobs.sort(key=cost, reverse=True)
    raw = {}
    start = time.perf_counter()
    with mp.get_context("spawn").Pool(N_WORKERS, initializer=_init_worker) as pool:
        for n, (job, scores) in enumerate(pool.imap_unordered(_run_job, jobs), 1):
            raw[(job[0], job[2], job[3])] = scores
            if n % 25 == 0 or n == len(jobs):
                print("  {}/{} jobs, {:.0f}s".format(n, len(jobs), time.perf_counter() - start), flush=True)

    blend_idx = BLENDS.index(M.BLEND_CF)
    summary = {}
    for key in configs:
        out = {}
        for split in ("holdout", "val"):
            arr = np.array([raw[(key, split, s)] for s in SEEDS])      # (seeds, blends)
            out[split + "_mean"] = float(arr[:, blend_idx].mean())
            out[split + "_std"] = float(arr[:, blend_idx].std())
            out[split + "_by_blend_mean"] = arr.mean(axis=0).tolist()
            out[split + "_by_blend_std"] = arr.std(axis=0).tolist()
        summary[key] = out
    return summary


def config_key(params):
    return ",".join("{}={}".format(k, params[k]) for k in sorted(params)) or "center"


def ablation(split_data, comps):
    ds, ev, cf = split_data
    base, mf, final = task3_preds(ev, comps, cf, M.BLEND_CF)
    rows = [
        ("task2", svd_reconstruct(svd_factors(ds), M.TOPK)[ev.u, ev.m]),
        ("task1", cf[ev.u, ev.m]),
        ("global_mean", np.full(len(ev.y), ds.global_mean)),
        ("baseline", base),
        ("latent", mf),
        ("clip", np.clip(mf, M.RATING_MIN, M.RATING_MAX)),
        ("final", final),
    ]
    return {name: {"rmse": ev.score(p), "per_user": ev.per_user(p)} for name, p in rows}


def fmt_users(d):
    return " ".join("-" if d[str(k)] is None else "{:.4f}".format(d[str(k)]) for k in KEY_USERS)


def section(title):
    print("\n== {} ==".format(title), flush=True)


def main():
    splits, train, val, hold = load_splits()
    ds, ev, cf = splits["val"]
    out = {"config": {"BIAS_REG": M.BIAS_REG, "MF_K": M.MF_K, "MF_REG": M.MF_REG, "MF_LR": M.MF_LR,
                      "MF_EPOCHS": M.MF_EPOCHS, "BLEND_CF": M.BLEND_CF, "MF_SEED": M.MF_SEED},
           "seeds": list(SEEDS), "blends": list(BLENDS)}

    # 1. 데이터 통계
    section("data")
    n_u, n_i = ds.R.shape
    item_cnt = ds.C.sum(axis=0)
    user_cnt = ds.C.sum(axis=1)
    wu = val["user_id"].isin(ds.uid2idx)
    wm = val["movie_id"].isin(ds.mid2idx)
    out["data"] = {
        "train_rows": len(train), "users": n_u, "movies": n_i,
        "density": float(ds.C.sum() / ds.C.size), "global_mean": ds.global_mean,
        "user_cnt_min": int(user_cnt.min()), "user_cnt_median": float(np.median(user_cnt)),
        "user_cnt_max": int(user_cnt.max()),
        "movie_cnt_median": float(np.median(item_cnt)), "movies_1_rating": int((item_cnt == 1).sum()),
        "val_rows": len(val), "val_users": int(val["user_id"].nunique()),
        "val_warm_users": int(val.loc[wu, "user_id"].nunique()), "val_warm_rows": int((wu & wm).sum()),
        "val_cold_user_only": int((~wu & wm).sum()), "val_cold_movie_only": int((wu & ~wm).sum()),
        "val_cold_both": int((~wu & ~wm).sum()),
        "holdout_rows": int(hold.sum()), "holdout_warm": len(splits["holdout"][1].y),
        "key_users": {str(k): {"train": int(user_cnt[ds.uid2idx[k]]), "val_warm": int((ev.uid == k).sum())}
                      for k in KEY_USERS},
    }
    for k, v in out["data"].items():
        print("{}: {}".format(k, v))

    # 2. Task1 / Task2 / Task3 (main.py 함수 그대로)
    section("tasks on val warm (main.py functions)")
    pairs = list(zip(val.loc[wu & wm, "user_id"], val.loc[wu & wm, "movie_id"]))
    preds, out["tasks_val"] = {}, {}
    for name, fn in [("task1", M.user_based_cf), ("task2", M.matrix_factorization), ("task3", M.optimized)]:
        t = time.perf_counter()
        preds[name] = np.array(fn(ds, pairs))
        out["tasks_val"][name] = {"rmse": ev.score(preds[name]), "per_user": ev.per_user(preds[name]),
                                  "seconds": time.perf_counter() - t}
        print("{}: RMSE {:.4f}  u199/458/529 {}  ({:.1f}s)".format(
            name, out["tasks_val"][name]["rmse"], fmt_users(out["tasks_val"][name]["per_user"]),
            out["tasks_val"][name]["seconds"]))

    # 3. Task1 변형
    section("Task1 variants")
    S = M.pearson_similarity(ds)
    assert np.allclose(cf[ev.u, ev.m], preds["task1"])
    cf_corated = cf_matrix(ds, corated_mean_similarity(ds))[ev.u, ev.m]
    n_common = ds.C @ ds.C.T
    off = ~np.eye(n_u, dtype=bool)
    out["task1_variants"] = {
        "corated_mean": {"rmse": ev.score(cf_corated), "per_user": ev.per_user(cf_corated)},
        "pairs_no_common": float((n_common[off] == 0).mean()),
        "pairs_common_le2_of_nonzero": float(((n_common[off] > 0) & (n_common[off] <= 2)).sum()
                                             / (n_common[off] > 0).sum()),
        "abs_sim_1_of_nonzero": float(np.isclose(np.abs(S[off]), 1.0).sum() / (S[off] != 0).sum()),
    }
    print(json.dumps(out["task1_variants"]))

    # 4. Task2 K sweep
    section("Task2 SVD K sweep")
    fac = svd_factors(ds)
    s = fac[1]
    tr_u, tr_m = np.nonzero(ds.C)
    tr_y = ds.R[tr_u, tr_m]
    energy = np.cumsum(s ** 2) / np.sum(s ** 2)
    out["task2_k"] = []
    for k in (1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500, len(s)):
        Pk = svd_reconstruct(fac, k)
        row = {"k": k, "val": ev.score(Pk[ev.u, ev.m]), "train_observed": rmse(tr_y, Pk[tr_u, tr_m]),
               "energy": float(energy[k - 1])}
        out["task2_k"].append(row)
        print(row)
    P400 = svd_reconstruct(fac, M.TOPK)
    out["task2_diag"] = {
        "item_mean_only_val": ev.score(ds.item_mean[ev.m]),
        "global_mean_val": ev.score(np.full(len(ev.y), ds.global_mean)),
        "val_mean_abs_diff_to_item_mean": float(np.mean(np.abs(P400[ev.u, ev.m] - ds.item_mean[ev.m]))),
        "energy_k1": float(energy[0]),
    }
    print(out["task2_diag"])

    # 5. Task3 ablation (제출 설정, seed 42)
    section("Task3 ablation (submitted config, seed {})".format(M.MF_SEED))
    comps = fit_components(ds)
    comps_h = fit_components(splits["holdout"][0])
    out["ablation"] = {"val": ablation(splits["val"], comps), "holdout": ablation(splits["holdout"], comps_h)}
    assert np.isclose(out["ablation"]["val"]["final"]["rmse"], out["tasks_val"]["task3"]["rmse"])
    for name in out["ablation"]["val"]:
        print("{:<12} holdout {:.4f} | val {:.4f}  u199/458/529 {}".format(
            name, out["ablation"]["holdout"][name]["rmse"], out["ablation"]["val"][name]["rmse"],
            fmt_users(out["ablation"]["val"][name]["per_user"])))
    prev = {split: ev_.score(task3_preds(ev_, fit_components(ds_, **PREVIOUS), cf_, M.BLEND_CF)[2])
            for split, (ds_, ev_, cf_) in splits.items()}
    out["previous_config_seed42"] = {"params": PREVIOUS, **prev}
    print("previous config {} seed {}: {}".format(PREVIOUS, M.MF_SEED, prev))

    # 6. 하이퍼파라미터 (seed 0~4 평균, 병렬)
    section("Task3 hyperparameters, seeds {} x {{holdout, val}}, {} workers".format(list(SEEDS), N_WORKERS))
    configs = {}
    for k in GRID_K:
        for reg in GRID_REG:
            if k in (2, 5) and reg != M.MF_REG:
                continue
            p = {kk: vv for kk, vv in {"MF_K": k, "MF_REG": reg}.items() if vv != getattr(M, kk)}
            configs[config_key(p)] = p
    for param, values in SWEEPS.items():
        for v in values:
            p = {} if v == getattr(M, param) else {param: v}
            configs[config_key(p)] = p
    summary = run_configs(configs)

    def entry(p, **extra):
        r = summary[config_key({kk: vv for kk, vv in p.items() if vv != getattr(M, kk)})]
        return {**extra, **{k: r[k] for k in ("holdout_mean", "holdout_std", "val_mean", "val_std")}}

    out["grid"] = [entry({"MF_K": k, "MF_REG": reg}, k=k, reg=reg)
                   for k in GRID_K for reg in GRID_REG if not (k in (2, 5) and reg != M.MF_REG)]
    out["sweeps"] = {param: [entry({param: v}, value=v) for v in values] for param, values in SWEEPS.items()}
    c = summary["center"]
    out["sweeps"]["BLEND_CF"] = [
        {"value": w, "holdout_mean": c["holdout_by_blend_mean"][j], "holdout_std": c["holdout_by_blend_std"][j],
         "val_mean": c["val_by_blend_mean"][j], "val_std": c["val_by_blend_std"][j]} for j, w in enumerate(BLENDS)]
    for row in out["grid"]:
        print("K={k:<4} reg={reg:<5} holdout {holdout_mean:.4f}±{holdout_std:.4f} | val {val_mean:.4f}±{val_std:.4f}"
              .format(**row))
    for param, rows in out["sweeps"].items():
        best = min(rows, key=lambda r: r["holdout_mean"])["value"]
        for row in rows:
            print("{}={:<6} holdout {:.4f}±{:.4f} | val {:.4f}±{:.4f}{}{}".format(
                param, row["value"], row["holdout_mean"], row["holdout_std"], row["val_mean"], row["val_std"],
                " *" if row["value"] == getattr(M, param) else "", "  <- holdout best" if row["value"] == best else ""))

    # 7. movie 인기도 구간
    section("val warm RMSE by movie train count")
    base = task3_preds(ev, comps, cf, M.BLEND_CF)[0]
    cnt = item_cnt[ev.m]
    out["by_movie_count"] = []
    for lo, hi, label in [(1, 1, "1"), (2, 5, "2-5"), (6, 20, "6-20"), (21, 10 ** 9, "21+")]:
        b = (cnt >= lo) & (cnt <= hi)
        row = {"label": label, "n": int(b.sum()), "baseline": rmse(ev.y[b], base[b]),
               **{n: rmse(ev.y[b], p[b]) for n, p in preds.items()}}
        out["by_movie_count"].append(row)
        print(row)

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("\nwrote", OUT_JSON)


if __name__ == "__main__":
    main()
