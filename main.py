# Project 2: Neighborhood-based / Model-Based Recommendation
# 2025-27567 배문성
#
# 제출: main.py (+ 필요시 requirements.txt)
# 채점: `python main.py` 로 input.txt를 읽어 output.txt를 생성합니다.
#       각 user_id, movie_id 쌍마다 3줄씩 출력됩니다.
#       (첫째 줄: User-based, 둘째 줄: Matrix factorization, 셋째 줄: Optimized result)
#       제출 전 이전에 만든 변수가 하나도 남아있지 않은 새 환경에서
#       이 파일 하나만으로 끝까지 도는지 반드시 확인 후 제출하세요.
#       100개의 입력에 대해 10분 안에 프로그램이 종료되어야 합니다.

# 이 코드가 사용하는 라이브러리(NumPy, Pandas 등)는 전부 requirements.txt에 명시하세요.
# (설치 코드를 main.py에 넣지 마세요 — 채점 시 `pip install -r requirements.txt`를
#  먼저 실행한 뒤 `python main.py`로 채점합니다.)

import os
from typing import NamedTuple

import numpy as np
import pandas as pd

# input.txt / output.txt / data/ 는 main.py와 같은 위치 기준 (실행 위치와 무관)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_FILE = os.path.join(DATA_DIR, "ratings_train.csv")
VAL_FILE = os.path.join(DATA_DIR, "ratings_val.csv")   # 성능 확인용. 학습(fit)에는 사용하지 말 것
INPUT_FILE = os.path.join(BASE_DIR, "input.txt")     # 한 줄에 "user_id,movie_id"
OUTPUT_FILE = os.path.join(BASE_DIR, "output.txt")
TOPK = 400

# Task 3 하이퍼파라미터. train 내부 10% 홀드아웃에서 seed 0~4 평균 RMSE로 탐색해 확정
# (ratings_val.csv 미사용)
BIAS_REG, BIAS_ITERS = 3.0, 15
MF_K, MF_LR, MF_REG, MF_EPOCHS, MF_BATCH, MF_SEED = 100, 0.01, 0.1, 60, 4096, 42
BLEND_CF = 0.2                       # optimized = (1 - w) * MF + w * user_based_cf
RATING_MIN, RATING_MAX = 0.5, 5.0

# read input.txt : 한 줄에 "user_id,movie_id"
# BOM, 필드 앞뒤 공백, "user_id,movie_id" 헤더나 주석 줄이 섞여 있어도 죽지 않게 한다
def read_input():
    pairs = []
    with open(INPUT_FILE, "r", encoding="utf-8-sig") as f:
        for l in f.readlines():
            l = l.strip()
            if not l or l.startswith("#"):
                continue
            fields = [x.strip() for x in l.split(",")]
            if len(fields) < 2 or not (fields[0].isdigit() and fields[1].isdigit()):
                continue
            uid, mid = fields[0], fields[1]
            pairs.append((uid, mid))
    return pairs


# pairs와 세 task의 prediction 리스트(각각 같은 길이/순서)를 받아
# 각 pair마다 "user_id,movie_id,prediction_score" 3줄(User-based/MF/Optimized 순)을 씀
def write_output(pairs, pred_user_based, pred_mf, pred_optimized):
    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="\n") as f:
        for (uid, mid), s1, s2, s3 in zip(pairs, pred_user_based, pred_mf, pred_optimized):
            f.write("{},{},{:.4f}\n".format(uid, mid, s1))
            f.write("{},{},{:.4f}\n".format(uid, mid, s2))
            f.write("{},{},{:.4f}\n".format(uid, mid, s3))

# ---------------- 데이터 레이어 ----------------

class Dataset(NamedTuple):
    R: np.ndarray            # (n_users, n_items) 평점. 미평가 칸은 0
    C: np.ndarray            # (n_users, n_items) 평가 여부 마스크 0/1
    D: np.ndarray            # (n_users, n_items) user 전체 평균으로 센터링. 미평가 칸은 0
    user_mean: np.ndarray    # (n_users,) 평가한 항목만으로 낸 user 평균
    item_mean: np.ndarray    # (n_items,) 평가한 user만으로 낸 movie 평균
    global_mean: float
    uid2idx: dict            # raw user_id -> 행 인덱스
    mid2idx: dict            # raw movie_id -> 열 인덱스


def build_dataset(train_df):
    """train_df를 [# unique users, # unique movies] 밀집 행렬과 부수 통계로 변환.
    user_id/movie_id가 희소하므로 raw id 대신 0-base 인덱스로 매핑한다."""
    uids = np.sort(train_df["user_id"].unique())
    mids = np.sort(train_df["movie_id"].unique())
    uid2idx = {int(u): i for i, u in enumerate(uids)}
    mid2idx = {int(m): i for i, m in enumerate(mids)}

    rows = train_df["user_id"].map(uid2idx).to_numpy()
    cols = train_df["movie_id"].map(mid2idx).to_numpy()
    vals = train_df["rating"].to_numpy(dtype=np.float64)

    R = np.zeros((len(uids), len(mids)), dtype=np.float64)
    C = np.zeros((len(uids), len(mids)), dtype=np.float64)
    R[rows, cols] = vals
    C[rows, cols] = 1.0

    global_mean = float(vals.mean())
    user_mean = R.sum(axis=1) / C.sum(axis=1)   # train user는 최소 6개 평가 -> 0 나눗셈 없음
    item_cnt = C.sum(axis=0)
    item_mean = np.divide(R.sum(axis=0), item_cnt,
                          out=np.full(len(mids), global_mean, dtype=np.float64),
                          where=item_cnt > 0)

    # 미평가 칸 0 -> I_uv 제한이 행렬곱으로 자동 처리됨
    D = (R - user_mean[:, None]) * C

    return Dataset(R, C, D, user_mean, item_mean, global_mean, uid2idx, mid2idx)


def cold_start_fallback(ds, u, m):
    """train에 없는 user/movie 폴백. 어떤 입력에도 유한한 값을 내야 한다."""
    if m is not None:
        return float(ds.item_mean[m])    # cold user  -> 해당 movie 평균
    if u is not None:
        return float(ds.user_mean[u])    # cold movie -> 해당 user 평균
    return ds.global_mean                # 둘 다 cold


#### TODO: 아래 함수들을 실제 구현으로 교체하세요 ####

def pearson_similarity(ds):
    """공통 평가 항목 I_uv 기준 Pearson 유사도 행렬 (n_users, n_users).
    r_u_bar는 공통 항목이 아닌 user 전체 평균. 분모의 두 합도 I_uv로 제한된다."""
    num = ds.D @ ds.D.T                  # sum_{i in I_uv} (r_ui - r_u_bar)(r_vi - r_v_bar)
    A = (ds.D ** 2) @ ds.C.T             # A[u, v] = sum_{i in I_uv} (r_ui - r_u_bar)^2
    den = np.sqrt(A * A.T)               # A.T[u, v] = A[v, u] = sum_{i in I_uv} (r_vi - r_v_bar)^2

    with np.errstate(divide="ignore", invalid="ignore"):
        S = np.where(den > 0, num / den, 0.0)   # 공통 항목 없음/분산 0 -> 유사도 0
    np.fill_diagonal(S, 0.0)             # 예측 시 v != u 조건을 자동 충족
    return S


def user_based_cf(ds, pairs):
    """Task 1: r_ui_hat = r_u_bar + sum_v s(u,v)(r_vi - r_v_bar) / sum_v |s(u,v)|.
    분모는 반드시 절댓값 합. 부호 있는 합은 0 근처에서 발산한다."""
    S = pearson_similarity(ds)
    abs_S = np.abs(S)

    predictions = []
    for user_id, movie_id in pairs:
        u = ds.uid2idx.get(int(user_id))
        m = ds.mid2idx.get(int(movie_id))
        if u is None or m is None:
            predictions.append(cold_start_fallback(ds, u, m))
            continue

        # i 미평가 v는 D=0/C=0이라 자동 제외
        numerator = float(S[u] @ ds.D[:, m])
        denominator = float(abs_S[u] @ ds.C[:, m])

        if denominator > 0:
            predictions.append(float(ds.user_mean[u]) + numerator / denominator)
        else:
            # 모든 유사도 0 -> 아이템 평균 평점 (명세 명시)
            predictions.append(float(ds.item_mean[m]))

    return predictions


def matrix_factorization(ds, pairs):
    """Task 2: 빈칸을 item 평균으로 채운 뒤 SVD로 분해(K=400)해 복원.
    0으로 채우면 모델이 0 쪽으로 학습된다."""
    F = ds.R * ds.C + ds.item_mean[None, :] * (1 - ds.C)
    U, s, Vt = np.linalg.svd(F, full_matrices=False)

    k = min(TOPK, len(s))                # full_matrices=False -> len(s) = n_users = 547
    user_factors = U[:, :k] * s[:k]
    item_factors = Vt[:k, :]
    P = user_factors @ item_factors
    
    predictions = []
    for user_id, movie_id in pairs:
        u = ds.uid2idx.get(int(user_id))
        m = ds.mid2idx.get(int(movie_id))
        if u is None or m is None:
            predictions.append(cold_start_fallback(ds, u, m))
            continue
        predictions.append(float(P[u, m]))

    return predictions


def observed_entries(ds):
    """관측된 (user, item, rating)만 1차원 배열로. 결측 대치 없이 학습하기 위함."""
    rows, cols = np.nonzero(ds.C)
    return rows, cols, ds.R[rows, cols]


def fit_baseline(ds):
    """mu + b_u + b_i 를 ALS closed-form으로 교대 추정. bincount라 완전 벡터화."""
    rows, cols, vals = observed_entries(ds)
    n_users, n_items = ds.R.shape
    mu = ds.global_mean
    bu = np.zeros(n_users)
    bi = np.zeros(n_items)
    user_cnt = np.bincount(rows, minlength=n_users)
    item_cnt = np.bincount(cols, minlength=n_items)

    for _ in range(BIAS_ITERS):
        bu = np.bincount(rows, vals - mu - bi[cols], n_users) / (user_cnt + BIAS_REG)
        bi = np.bincount(cols, vals - mu - bu[rows], n_items) / (item_cnt + BIAS_REG)
    return mu, bu, bi


def fit_latent_factors(ds, mu, bu, bi):
    """baseline 잔차에 minibatch SGD로 p_u, q_i 학습. 관측치에만 손실을 건다."""
    rows, cols, vals = observed_entries(ds)
    resid = vals - (mu + bu[rows] + bi[cols])
    n_users, n_items = ds.R.shape

    rng = np.random.default_rng(MF_SEED)
    P = rng.normal(0, 0.1, (n_users, MF_K))
    Q = rng.normal(0, 0.1, (n_items, MF_K))
    order = np.arange(len(vals))

    for _ in range(MF_EPOCHS):
        rng.shuffle(order)
        for start in range(0, len(order), MF_BATCH):
            idx = order[start:start + MF_BATCH]
            u, i = rows[idx], cols[idx]
            pu, qi = P[u], Q[i]
            err = (resid[idx] - np.einsum("ij,ij->i", pu, qi))[:, None]
            # 같은 u/i가 배치 안에서 겹치면 add.at이 누적 적용 (Hogwild 식 허용)
            np.add.at(P, u, MF_LR * (err * qi - MF_REG * pu))
            np.add.at(Q, i, MF_LR * (err * pu - MF_REG * qi))
    return P, Q


def optimized(ds, pairs):
    """Task 3: Task 2 개선. 결측 대치 대신 관측치에만 손실을 거는 baseline + biased MF.
    user-based CF와 블렌딩 후 평점 범위로 clip."""
    mu, bu, bi = fit_baseline(ds)
    P, Q = fit_latent_factors(ds, mu, bu, bi)

    mf_pred = np.empty(len(pairs))
    for t, (user_id, movie_id) in enumerate(pairs):
        u = ds.uid2idx.get(int(user_id))
        m = ds.mid2idx.get(int(movie_id))
        if u is None or m is None:
            mf_pred[t] = cold_start_fallback(ds, u, m)
        else:
            mf_pred[t] = mu + bu[u] + bi[m] + P[u] @ Q[m]

    cf_pred = np.array(user_based_cf(ds, pairs))
    blended = (1.0 - BLEND_CF) * mf_pred + BLEND_CF * cf_pred
    return list(np.clip(blended, RATING_MIN, RATING_MAX))

#### TODO end ####


if __name__ == "__main__":
    # 학습은 ratings_train.csv만 사용. ratings_val.csv 기반 성능 확인은 eval.py (제출 X)
    train_df = pd.read_csv(TRAIN_FILE)
    pairs = read_input()
    ds = build_dataset(train_df)

    pred_user_based = user_based_cf(ds, pairs)
    pred_mf = matrix_factorization(ds, pairs)
    pred_optimized = optimized(ds, pairs)

    write_output(pairs, pred_user_based, pred_mf, pred_optimized)
