# 개발용 (제출 X). 두 가지를 확인한다.
#   1. ratings_val.csv warm 쌍 기준 RMSE + 명세 per-user 기준 (Task1, Task2)
#   2. 스모크 테스트: 다른 작업 디렉터리에서 `python main.py`를 새 프로세스로 두 번 돌려
#      output.txt 포맷(쌍당 3줄, 콤마, 소수 4자리, 입력 순서), 재현성, 실행 시간 검사
# ratings_val.csv는 여기서 평가에만 쓴다. main.py의 학습 경로는 읽지 않는다.
# 사용: python make_input.py && python eval.py

import math
import os
import re
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd

from main import (BASE_DIR, OUTPUT_FILE, TOPK, TRAIN_FILE, VAL_FILE, build_dataset,
                  matrix_factorization, optimized, read_input, user_based_cf)

MAIN_PY = os.path.join(BASE_DIR, "main.py")
TIME_LIMIT_SEC = 600                 # 명세: 입력 100개에 10분 이내
LINE_RE = re.compile(r"^(\d+),(\d+),(-?\d+\.\d{4})$")
TASKS = [                            # (이름, 함수, 명세 per-user RMSE 기준)
    ("Task1 user-based", user_based_cf, {199: 0.65, 458: 0.75, 529: 0.75}),
    ("Task2 SVD K={}".format(TOPK), matrix_factorization, {199: 0.60, 458: 0.70, 529: 0.75}),
    ("Task3 optimized", optimized, {199: None, 458: None, 529: None}),
]


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def check(ok, msg):
    print("  [{}] {}".format("OK" if ok else "FAIL", msg))
    return ok


def report_val_rmse(ds, val_df):
    # train에 없는 user/movie는 명세상 제외 -> warm 행만 평가
    warm = val_df[val_df["user_id"].isin(ds.uid2idx) & val_df["movie_id"].isin(ds.mid2idx)]
    pairs = list(zip(warm["user_id"], warm["movie_id"]))
    y = warm["rating"].to_numpy()
    users = warm["user_id"].to_numpy()

    ok = True
    for label, fn, limits in TASKS:
        pred = np.array(fn(ds, pairs))
        print("{}: warm RMSE {:.4f}  (n={})".format(label, rmse(y, pred), len(y)))
        for user_id, limit in limits.items():
            score = rmse(y[users == user_id], pred[users == user_id])
            if limit is None:
                print("  user {:>3}: RMSE {:.4f}".format(user_id, score))
            else:
                ok &= check(score < limit, "user {:>3}: RMSE {:.4f}  (limit <{:.2f})".format(
                    user_id, score, limit))
    return ok


def run_main():
    # main.py 옆의 input.txt/output.txt를 쓰는지 보려고 일부러 다른 디렉터리에서 실행
    with tempfile.TemporaryDirectory() as cwd:
        start = time.perf_counter()
        proc = subprocess.run([sys.executable, MAIN_PY], cwd=cwd, capture_output=True, text=True)
        elapsed = time.perf_counter() - start
    with open(OUTPUT_FILE, "r", encoding="utf-8", newline="") as f:
        return proc, elapsed, f.read()


def smoke_test(ds, val_df):
    pairs = read_input()
    print("input.txt: {} pairs".format(len(pairs)))
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    proc, elapsed, raw = run_main()
    if not check(proc.returncode == 0, "python main.py exit code {}".format(proc.returncode)):
        print(proc.stderr)
        return False
    ok = check(elapsed < TIME_LIMIT_SEC, "elapsed {:.1f}s (limit {}s)".format(elapsed, TIME_LIMIT_SEC))

    lines = raw.split("\n")
    ok &= check(lines[-1] == "", "trailing newline, LF only" if "\r" not in raw else "CRLF found")
    lines = lines[:-1]
    ok &= check(len(lines) == 3 * len(pairs), "{} lines == 3 x {} pairs".format(len(lines), len(pairs)))

    scores = np.full((len(pairs), 3), np.nan)
    bad = []
    for k, line in enumerate(lines[:3 * len(pairs)]):
        m = LINE_RE.match(line)
        if not m or (m.group(1), m.group(2)) != pairs[k // 3] or not math.isfinite(float(m.group(3))):
            bad.append((k + 1, line))
            continue
        scores[k // 3, k % 3] = float(m.group(3))
    ok &= check(not bad, "format 'user_id,movie_id,score(.4f)' in input order"
                + ("" if not bad else "  first bad: line {} {!r}".format(*bad[0])))

    # 같은 코드를 import 경로로 돌린 값과 일치해야 함 (출력 순서 User-based / MF / Optimized)
    expected = np.array([[float("{:.4f}".format(x)) for x in fn(ds, pairs)] for _, fn, _ in TASKS]).T
    ok &= check(np.array_equal(scores, expected), "subprocess output == in-process predictions")
    ok &= check(np.all((scores[:, 2] >= 0.5) & (scores[:, 2] <= 5.0)), "Task3 scores within [0.5, 5.0]")

    _, _, raw2 = run_main()
    ok &= check(raw2 == raw, "second run byte-identical (seed fixed)")

    # input.txt 중 val 정답이 있는 warm 쌍에 대한 참고용 RMSE
    truth = {(str(u), str(m)): r for u, m, r in val_df.itertuples(index=False)}
    idx = [t for t, (u, m) in enumerate(pairs)
           if (u, m) in truth and int(u) in ds.uid2idx and int(m) in ds.mid2idx]
    y = np.array([truth[pairs[t]] for t in idx])
    print("input.txt warm pairs with val rating: n={}".format(len(idx)))
    for col, (label, _, _) in enumerate(TASKS):
        print("  {}: RMSE {:.4f}".format(label, rmse(y, scores[idx, col])))
    return ok


if __name__ == "__main__":
    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VAL_FILE)
    ds = build_dataset(train_df)
    print("matrix {} x {}, density {:.2%}\n".format(
        ds.R.shape[0], ds.R.shape[1], ds.C.sum() / ds.C.size))

    print("== ratings_val.csv (warm) ==")
    val_ok = report_val_rmse(ds, val_df)
    print("\n== smoke test: python main.py ==")
    smoke_ok = smoke_test(ds, val_df)

    print("\nALL OK" if val_ok and smoke_ok else "\nSOME CHECKS FAILED")
    sys.exit(0 if val_ok and smoke_ok else 1)
