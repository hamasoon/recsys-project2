# 개발용 (제출 X). 채점 형식과 같은 테스트용 input.txt 100줄을 만든다.
#
# 구성 (seed 고정 -> 매번 같은 파일):
#   - PDF 예시 4쌍                         (1,31) (2,10) (3,1235) (4,10)
#   - 기준 user 199/458/529의 val warm 쌍  각 10개
#   - 그 외 val warm 쌍                    60개
#   - cold-start 방어 확인용                cold user 2 / cold movie 2 / 둘 다 cold 2
# warm = user, movie 모두 train에 있는 쌍. val 정답이 있으므로 eval.py가 RMSE도 낸다.

import numpy as np
import pandas as pd

from main import INPUT_FILE, TRAIN_FILE, VAL_FILE

SEED = 0
PDF_EXAMPLES = [(1, 31), (2, 10), (3, 1235), (4, 10)]
KEY_USERS, PER_KEY_USER = (199, 458, 529), 10
N_OTHER_WARM = 60
N_COLD_EACH = 2


def main():
    train = pd.read_csv(TRAIN_FILE)
    val = pd.read_csv(VAL_FILE)
    warm_user = val["user_id"].isin(train["user_id"])
    warm_movie = val["movie_id"].isin(train["movie_id"])

    warm = val[warm_user & warm_movie]
    key = warm[warm["user_id"].isin(KEY_USERS)]
    key = key.groupby("user_id", group_keys=False).sample(PER_KEY_USER, random_state=SEED)
    other = warm[~warm["user_id"].isin(KEY_USERS)].sample(N_OTHER_WARM, random_state=SEED)
    cold = pd.concat([
        val[~warm_user & warm_movie].sample(N_COLD_EACH, random_state=SEED),
        val[warm_user & ~warm_movie].sample(N_COLD_EACH, random_state=SEED),
        val[~warm_user & ~warm_movie].sample(N_COLD_EACH, random_state=SEED),
    ])

    rest = pd.concat([key, other, cold]).sample(frac=1, random_state=SEED)
    pairs = PDF_EXAMPLES + list(zip(rest["user_id"], rest["movie_id"]))

    with open(INPUT_FILE, "w", encoding="utf-8", newline="\n") as f:
        for uid, mid in pairs:
            f.write("{},{}\n".format(uid, mid))
    print("wrote {} pairs -> {}".format(len(pairs), INPUT_FILE))


if __name__ == "__main__":
    main()
