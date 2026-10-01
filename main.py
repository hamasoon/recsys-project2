# Project 2: Neighborhood-based / Model-Based Recommendation
# 2026-00000 홍길동
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
import numpy as np
import pandas as pd

DATA_DIR = "data"
TRAIN_FILE = os.path.join(DATA_DIR, "ratings_train.csv")
VAL_FILE = os.path.join(DATA_DIR, "ratings_val.csv")   # 성능 확인용. 학습(fit)에는 사용하지 말 것
INPUT_FILE = "input.txt"     # 한 줄에 "user_id,movie_id"
OUTPUT_FILE = "output.txt"


# read input.txt : 한 줄에 "user_id,movie_id"
def read_input():
    pairs = []
    with open(INPUT_FILE, "r") as f:
        for l in f.readlines():
            l = l.strip()
            if not l:
                continue
            uid, mid = l.split(",")
            pairs.append((uid, mid))
    return pairs


# pairs와 세 task의 prediction 리스트(각각 같은 길이/순서)를 받아
# 각 pair마다 "user_id,movie_id,prediction_score" 3줄(User-based/MF/Optimized 순)을 씀
def write_output(pairs, pred_user_based, pred_mf, pred_optimized):
    with open(OUTPUT_FILE, "w") as f:
        for (uid, mid), s1, s2, s3 in zip(pairs, pred_user_based, pred_mf, pred_optimized):
            f.write("{},{},{:.4f}\n".format(uid, mid, s1))
            f.write("{},{},{:.4f}\n".format(uid, mid, s2))
            f.write("{},{},{:.4f}\n".format(uid, mid, s3))


#### TODO: 아래 함수들을 실제 구현으로 교체하세요 ####

def user_based_cf(train_df, pairs):
    # Task 1: user-item sparse matrix + Pearson Correlation 기반 User-based CF
    # test implementation — 항상 3.5점으로 예측
    return [3.5 for _ in pairs]


def matrix_factorization(train_df, pairs):
    # Task 2: user-item sparse matrix를 SVD로 분해 (K=400) 후 예측
    # test implementation — 항상 3.5점으로 예측
    return [3.5 for _ in pairs]


def optimized(train_df, pairs):
    # Task 3: Task 1 또는 Task 2를 기반으로 한 최적화 버전
    # test implementation — 항상 3.5점으로 예측
    return [3.5 for _ in pairs]

#### TODO end ####


train_df = pd.read_csv(TRAIN_FILE)
pairs = read_input()

pred_user_based = user_based_cf(train_df, pairs)
pred_mf = matrix_factorization(train_df, pairs)
pred_optimized = optimized(train_df, pairs)

write_output(pairs, pred_user_based, pred_mf, pred_optimized)
