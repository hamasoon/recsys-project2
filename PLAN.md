# Project 2 구현 계획 — Neighborhood-based / Model-Based Recommendation

> 근거 문서: `Term Project 2.pdf` / 마감: 2026-10-22 23:59
> 제출물: `PRJ2_202600000_홍길동.zip` = 보고서 PDF + `data/` + `main.py`
> 채점: `pip install -r requirements.txt` → `python main.py` (입력 100쌍, **10분 이내 종료**)

---

## 1. 데이터 실측

`data/` 두 파일을 직접 집계한 수치다. 설계 판단의 근거라 먼저 적는다.

| 항목 | train | val |
|---|---|---|
| rows | 80,036 | 10,071 |
| unique users | 547 | 84 |
| unique movies | 7,357 | 3,383 |

- 컬럼: `user_id,movie_id,rating` (헤더 있음)
- rating 범위: 0.5 ~ 5.0, 0.5 단위. train 평균 **3.5483**
- id는 희소: `max(user_id)=671`, `max(movie_id)=94833` → **raw id를 행렬 인덱스로 쓰면 안 됨**. 반드시 `id → index` 매핑 필요 (명세도 `[# unique users, # unique movies]` 요구)
- 밀도: 80,036 / (547 × 7,357) = **1.99%**
- user당 rating: min 6 / median 64 / max 2,061
- movie당 rating: min 1 / median 3 / max 265 — **rating이 1개뿐인 movie가 2,267개** (전체 7,357개 중 31%)

### 1.1 val의 cold-start 구조 (중요)

| val row 분류 | 개수 |
|---|---|
| warm (user·movie 둘 다 train에 존재) | **1,405** |
| cold user | 7,668 |
| cold movie | 1,874 |
| 둘 다 cold | 876 |

val 84명 중 **64명이 train에 없다**. 명세가 "val에 있고 train에 없는 movie/user는 계산에서 제외"라고 한 이유. 따라서 **전체 val RMSE는 의미 없고, warm 1,405행 기준으로만 평가**해야 한다.

### 1.2 명세의 RMSE 기준점

명세는 특정 user 3명의 **per-user RMSE** 로 검증 기준을 준다. 세 명 모두 train·val 양쪽에 존재한다.

| user_id | train ratings | val ratings | Task1 기준 | Task2 기준 |
|---|---|---|---|---|
| 199 | 358 | 64 | < 0.65 | < 0.60 |
| 458 | **18** | 58 | < 0.75 | < 0.70 |
| 529 | 508 | 96 | < 0.75 | < 0.75 |

user 458은 train rating이 18개뿐 → 가장 어렵고 기준도 느슨하다. 로컬 검증 하네스는 이 3명 per-user RMSE를 1급 지표로 출력해야 한다.

### 1.3 연산 비용 (실측, numpy 1.23.5)

- `np.linalg.svd(547×7357, full_matrices=False)` → **0.66초** (U: 547×547, Vt: 547×7357)
- `X @ X.T` (547×7357) → **0.014초**
- dense 행렬 float64 메모리: 547 × 7357 × 8B = **32MB** / 행렬 하나당

→ **런타임은 전혀 제약이 아니다.** 10분 예산 대비 수 초. 희소 자료구조·근사 알고리즘 최적화는 불필요하고, 밀집 행렬 + 벡터화로 전부 처리한다. `K=400` 도 `min(547, 7357)=547` 이하라 유효하다.

---

## 2. 전체 구조

제출 트리상 소스는 `main.py` **한 파일**이다. 개발용 도구는 별도 파일로 두고 zip에서 제외한다.

```
recsys-project2/
├── main.py              # 제출. 단일 파일, 외부 import는 numpy/pandas만
├── requirements.txt     # 제출. numpy, pandas 버전 핀
├── data/
│   ├── ratings_train.csv
│   └── ratings_val.csv
├── input.txt            # 채점 시 TA가 배치 (제출 X)
├── output.txt           # 실행 시 생성 (제출 X)
│
├── eval.py              # 개발용. 제출 X — val 기반 RMSE 하네스
├── tune.py              # 개발용. 제출 X — Task3 하이퍼파라미터 탐색
└── PLAN.md / README.md  # 제출 X
```

### 2.1 `main.py` 내부 레이어

```
[0] 상수 / 경로              DATA_DIR, TRAIN_FILE, INPUT_FILE, OUTPUT_FILE
[1] I/O                      read_input(), write_output()        ← 스켈레톤 그대로 유지
[2] 데이터 레이어             build_dataset(train_df) -> Dataset
[3] Task 1                   pearson_similarity(), user_based_cf()
[4] Task 2                   matrix_factorization()
[5] Task 3                   optimized()
[6] 오케스트레이션            main()
```

**단일 전처리 원칙**: 전처리는 `build_dataset()` 에서 한 번만 수행하고, 세 Task 함수는 그 결과를 **인자로 받는다**. 스켈레톤의 `(train_df, pairs)` 시그니처를 `(ds, pairs)` 로 바꾼다 — 전역 캐시보다 단순하고, `eval.py` 에서 임의 조합으로 호출하기도 쉽다.

```python
def user_based_cf(ds, pairs):       ...
def matrix_factorization(ds, pairs): ...
def optimized(ds, pairs):            ...
```

### 2.2 `Dataset` 가 보유할 것

전처리는 전부 여기서 끝내고, 세 Task는 이 객체만 읽는다.

| 필드 | 형태 | 설명 |
|---|---|---|
| `uid2idx`, `mid2idx` | dict | raw id → 0-base 행렬 인덱스 |
| `R` | (547, 7357) float | rating, 미평가 = 0 |
| `C` | (547, 7357) float | 평가 여부 마스크 0/1 |
| `user_mean` | (547,) | **평가한 항목만**으로 계산한 user 평균 (`R.sum(1) / C.sum(1)`) |
| `item_mean` | (7357,) | 평가한 user만으로 계산한 movie 평균. rating 0개 movie는 train에 없으므로 발생 X |
| `D` | (547, 7357) float | mean-centered. `(R - user_mean[:,None]) * C` — 미평가 위치는 **반드시 0** |
| `global_mean` | scalar | 3.5483 |

`D`에서 미평가 칸을 0으로 눌러두는 게 핵심이다. 이 덕분에 "공통 평가 아이템 $I_{u,v}$ 에 대해서만 합산"하는 명세의 제약이 행렬곱 한 번으로 자동 충족된다.

### 2.3 실행 흐름

```
train_df = pd.read_csv(TRAIN_FILE)
pairs    = read_input()                      # [(user_id, movie_id), ...] — str로 읽힘, int 변환 필요
ds       = build_dataset(train_df)           # 1회
p1 = user_based_cf(ds, pairs)
p2 = matrix_factorization(ds, pairs)
p3 = optimized(ds, pairs)
write_output(pairs, p1, p2, p3)              # pair당 3줄, 소수점 4자리
```

`read_input()`은 `uid, mid` 를 **문자열로** 반환한다 (스켈레톤 그대로). 출력 시 입력 문자열을 그대로 다시 써야 포맷이 어긋나지 않으므로, 예측 계산용으로만 `int()` 변환하고 출력에는 원본 문자열을 쓴다.

### 2.4 클래스 / 파일 분할 여부 — 둘 다 불필요

**결론: 동작을 가진 클래스는 쓰지 않는다. 제출 소스는 `main.py` 한 파일로 유지한다.**

#### 파일 분할을 안 하는 이유

1. 명세가 단일 파일을 전제한다. 제출 상태 트리에 `main.py` 하나뿐이고, 본문도 "`main.py`를 기반으로 구현". 모듈을 쪼개면 채점 환경에서 import 실패 위험만 생기고 얻는 게 없다.
2. 규모가 작다. 예상 분량:

   | 블록 | 줄 수 |
   |---|---|
   | 헤더·import·상수 | ~25 |
   | `read_input` / `write_output` (스켈레톤) | ~20 |
   | `build_dataset` | ~35 |
   | `pearson_similarity` + 폴백 헬퍼 | ~40 |
   | `user_based_cf` | ~30 |
   | `matrix_factorization` | ~25 |
   | `fit_mf` + `optimized` | ~70 |
   | `main` | ~15 |
   | **합계** | **~260** |

   260줄은 단일 파일로 읽는 게 오히려 쉽다. 분할이 이득을 내기 시작하는 건 대략 800줄 이상부터다.
3. 이미 필요한 분리는 되어 있다. `eval.py` / `tune.py`는 **제출물과 개발 도구**를 가르는 경계이고, 이건 모듈 분할이 아니라 배포 단위 분리다. 세 Task가 `(ds, pairs)` 시그니처만 쓰므로 `eval.py`에서 `from main import ...` 로 그대로 재사용된다 (실행부를 `if __name__` 아래로 내리는 이유).

#### 클래스를 안 쓰는 이유

이 코드에 상태를 들고 메서드로 조작할 대상이 없다. 전부 "**불변 데이터 묶음 + 순수 함수**" 형태다.

- `Dataset`: 필드 7개를 보유하지만 메서드가 없다. `build_dataset()` 이후 변하지 않는다.
- `fit_mf()` 결과: `(P, Q, b_u, b_i, mu)` 5개 역시 학습 끝나면 불변.

클래스를 세우면 `self.`가 붙은 getter·wrapper만 늘고, TA가 `user_based_cf` 수식 대응을 확인하기만 어려워진다.

#### 대신 쓸 것: `NamedTuple` 레코드

필드 7개를 위치 인자로 돌리는 건 터지기 쉽고, `dict`는 `ds["use_mean"]` 같은 오타가 런타임까지 간다. `typing.NamedTuple`이면 불변 + 속성 접근 + 타입 힌트를 10줄로 얻는다. 문법상 `class`지만 **행위가 없는 레코드 선언**이라 위 판단과 모순되지 않는다.

```python
class Dataset(NamedTuple):
    R: np.ndarray            # (n_users, n_items) rating, 미평가 0
    C: np.ndarray            # 마스크 0/1
    D: np.ndarray            # mean-centered, 미평가 0
    user_mean: np.ndarray
    item_mean: np.ndarray
    global_mean: float
    uid2idx: dict
    mid2idx: dict

class MFParams(NamedTuple):
    P: np.ndarray            # (n_users, k)
    Q: np.ndarray            # (n_items, k)
    bu: np.ndarray
    bi: np.ndarray
    mu: float
```

`dataclass(frozen=True)`도 동등하지만 `NamedTuple`이 import 하나로 끝나고 언패킹도 된다.

#### 이 판단을 뒤집을 조건

§5.3의 개선안을 전부 쌓아 `optimized()`가 150줄을 넘기면, 그때 `fit_mf` / `fit_item_knn` / `blend` 로 **함수를 더 쪼갠다**. 그래도 파일과 클래스는 그대로 둔다 — 함수 분해로 충분하다.

---

---

## 3. Task 1 — User-based CF

### 3.1 유사도: 공통 평가 항목 기준 Pearson

명세 수식:

$$s(u,v) = \frac{\sum_{i \in I_{u,v}} (r_{ui}-\bar r_u)(r_{vi}-\bar r_v)}{\sqrt{\sum_{i \in I_{u,v}} (r_{ui}-\bar r_u)^2} \cdot \sqrt{\sum_{i \in I_{u,v}} (r_{vi}-\bar r_v)^2}}$$

**주의**: 분모의 두 합도 $I_{u,v}$ 로 제한된다. 즉 전역 정규화 후 코사인을 때리는 것과 **다르다** — 분모가 쌍마다 달라진다. 그래도 완전 벡터화된다:

```
Num  = D @ D.T                 # (547,547)  미평가=0이라 I_uv 밖은 자동 소거
A    = (D ** 2) @ C.T          # A[u,v] = Σ_{i∈I_uv} (r_ui - r̄_u)^2
S    = Num / sqrt(A * A.T)     # A.T[u,v] = A[v,u] = Σ_{i∈I_uv} (r_vi - r̄_v)^2
```

총 비용 ~0.05초. 명세 요구사항 반영:
- 공통 평가 항목 없음 → `Num=0`, `A=0` → 0 나눗셈 → **`np.errstate`로 무시하고 결과를 0으로 치환**
- 분모 0 (한쪽이 공통 항목에서 상수 평점) → 동일하게 0
- 대각 `S[u,u]`은 예측 시 자기 자신을 제외해야 하므로 **0으로 세팅**

### 3.2 예측

$$\hat r_{ui} = \bar r_u + \frac{\sum_{v \neq u} s(u,v)\,(r_{vi}-\bar r_v)}{\sum_{v \neq u} |s(u,v)|}$$

여기서 $v$ 는 아이템 $i$ 를 평가한 user로 한정된다. `D`, `C`가 그 한정을 대신한다:

```
num = S[u] @ D[:, i]          # i를 평가 안 한 v는 D=0 → 자동 제외
den = abs(S[u]) @ C[:, i]     # i를 평가한 v의 |s| 합
pred = user_mean[u] + num/den
```

`S`의 대각을 0으로 두었으므로 $v \neq u$ 조건도 자동 충족.

100쌍이면 pair별 루프로 충분하다 (각 1e-4초 수준). 전체 val 평가 시에도 `S @ D` / `|S| @ C` 를 통째로 돌리면 0.1초.

### 3.3 폴백 (명세 명시 + 포맷 방어)

| 상황 | 예측값 |
|---|---|
| `den == 0` (모든 유사도 0) | **해당 movie의 평균 평점** ← 명세 명시 |
| movie가 train에 없음 (cold item) | 해당 user의 평균 평점. user도 없으면 `global_mean` |
| user가 train에 없음 (cold user) | 해당 movie의 평균 평점. movie도 없으면 `global_mean` |
| 둘 다 없음 | `global_mean` |

cold 항목은 RMSE 평가에선 제외되지만 **`output.txt`에는 반드시 한 줄이 나가야 한다**. 입력 쌍 수와 출력 줄 수가 어긋나면 포맷 감점이다. 어떤 입력이 와도 `NaN`/예외 없이 유한한 실수를 내도록 폴백을 끝까지 채운다.

### 3.4 Task 1에서 하지 말 것

평가 기준 1-a는 "TA가 계산한 점수와의 **허용 범위 내 오차**"다. 즉 Task 1·2는 RMSE 경쟁이 아니라 **레퍼런스 구현 재현**이다. 따라서 Task 1에서는:

- clipping(`[0.5, 5.0]` 자르기) **금지**
- top-K 이웃 절삭, significance weighting, shrinkage **금지**
- 음수 유사도 버리기 **금지** (명세는 가중합에 절댓값을 쓰라고 명시 — 음수 유사도를 쓰겠다는 뜻)

이런 개선은 전부 Task 3에 몰아넣는다.

---

## 4. Task 2 — Matrix Factorization (SVD)

명세가 절차를 못 박았으므로 그대로 따른다.

1. user-item 행렬 생성 (Task 1과 동일한 인덱스 매핑 재사용)
2. **빈칸을 각 movie의 평균 평점으로 채움** (0으로 두면 0 쪽으로 학습됨 — 명세 근거)
   ```
   F = R * C + item_mean[None, :] * (1 - C)
   ```
3. `U, s, Vt = np.linalg.svd(F, full_matrices=False)` → 0.66초
4. 상위 **K=400** 특이값만 사용
   ```
   user_factors = U[:, :400] * s[:400]        # (547, 400)
   item_factors = Vt[:400, :]                 # (400, 7357)
   P = user_factors @ item_factors            # (547, 7357) 복원
   ```
5. `pred = P[u_idx, m_idx]`

- `full_matrices=False`면 특이값이 547개뿐 → K=400은 유효. 다만 `K = min(400, len(s))` 로 방어한다.
- 복원 행렬 `P`는 32MB. 한 번 만들어 재사용한다 (pair마다 재분해 금지).
- Task 1과 마찬가지로 **clipping 금지**.
- 폴백: cold user/movie는 해당 축 평균 → `global_mean` 순으로. Task 1과 동일한 폴백 함수를 공유한다.

> 데이터 특성 주의: rating이 1개뿐인 movie가 2,267개라, 그 열은 "실측 1개 + 나머지 전부 그 값"이 되어 사실상 상수 열이다. 이 때문에 SVD가 많은 열에서 movie 평균을 그대로 뱉는다. 예상된 동작이고, 명세대로 가는 게 맞다. 이 한계가 바로 Task 3의 출발점이다.

---

## 5. Task 3 — Optimization

Task 3만 자유도가 있다. 평가 기준 1-c는 **실제 rating과의 RMSE**다 (레퍼런스 재현이 아님). 제약은 하나: **학습에 `ratings_val.csv`를 쓰면 안 된다.**

### 5.1 Task 2를 최적화 대상으로 선택

이유: SVD-on-imputed의 결함이 구조적이고 크다.

- 결측치를 movie 평균으로 **채워 넣은 뒤** 분해하므로, 모델이 "실제 평점"이 아니라 "내가 만들어 넣은 평균값"을 재현하도록 학습된다. 밀도가 1.99%니 학습 신호의 98%가 가짜다.
- K=400은 547 rank 중 400 → 거의 전체. 정규화가 없어 과적합 방향.
- → **관측된 평점에만 손실을 거는 biased MF (FunkSVD)** 로 교체하면 개선 폭이 가장 크고, 보고서에서 "왜 개선됐는지"를 설명하기도 깔끔하다.

### 5.2 핵심 로직: baseline + biased MF (SGD)

$$\hat r_{ui} = \mu + b_u + b_i + p_u^\top q_i$$

손실 (관측된 $(u,i)$ 에만):

$$\min \sum_{(u,i) \in \mathcal{K}} (r_{ui} - \hat r_{ui})^2 + \lambda(b_u^2 + b_i^2 + \|p_u\|^2 + \|q_i\|^2)$$

SGD 업데이트 (epoch마다 셔플):
```
e      = r_ui - pred
b_u   += lr * (e - reg * b_u)
b_i   += lr * (e - reg * b_i)
p_u   += lr * (e * q_i - reg * p_u)
q_i   += lr * (e * p_i - reg * q_i)
```

- 80,036개 관측치 × ~30 epoch = 2.4M 업데이트. 순수 파이썬 루프면 느리므로 **epoch 단위 numpy 벡터화 또는 미니배치**로 짜고, 최악의 경우에도 10분 예산 대비 여유가 크다. 먼저 단순하게 짜고 실측해서 필요하면 최적화한다.
- 초기값: `p, q ~ N(0, 0.1)`, `b_u = b_i = 0`
- **난수 시드 고정** (`np.random.default_rng(42)`) — 채점 재현성 확보

### 5.3 추가로 쌓을 개선 (각각 val warm으로 효과 측정 후 채택)

우선순위 순. 하나씩 붙여가며 per-user RMSE(199/458/529)와 warm RMSE가 실제로 내려갈 때만 유지한다.

1. **clipping** `np.clip(pred, 0.5, 5.0)` — 거의 공짜로 개선. 가장 먼저 적용
2. **cold/저관측 방어**: user 458처럼 train rating이 적은 경우 MF가 불안정. baseline(`μ + b_u + b_i`)과 MF 예측을 관측 수 기반으로 섞는다 — $w = n_u / (n_u + \beta)$
3. **item-based CF 잔차 보정**: baseline 잔차에 대해 item-item 유사도 kNN. user 수(547)보다 item 수(7,357)가 많지만 item-item 행렬 7357² × 4B = 216MB (float32) → 메모리 주의. top-K만 보관하거나 생략
4. **Task 1 개선판과 블렌딩**: significance weighting $\frac{\min(|I_{u,v}|, N)}{N}$ 적용한 user-based CF와 MF를 가중 평균. 서로 다른 오류 구조라 블렌딩 이득이 보통 있다

### 5.4 하이퍼파라미터

`tune.py`에서 탐색하고, **최종 상수만 `main.py`에 하드코딩**한다 (제출본이 튜닝 코드를 들고 있을 필요 없음).

| 파라미터 | 탐색 범위 | 초기값 |
|---|---|---|
| `K` (latent dim) | 10 ~ 100 | 30 |
| `lr` | 0.002 ~ 0.02 | 0.005 |
| `reg` | 0.02 ~ 0.2 | 0.05 |
| `epochs` | 20 ~ 100 | 40 |
| `beta` (blend) | 5 ~ 50 | 20 |

**튜닝 시 과적합 주의**: val은 warm row가 1,405개뿐이고, 기준 user 3명의 val은 각각 64/58/96개다. 이 작은 집합에 과하게 맞추면 TA의 `ratings_test.csv`에서 무너진다. 탐색은 거친 격자로만 하고, 소수점 셋째 자리 개선은 쫓지 않는다. 보수적인 정규화 쪽을 택한다.

---

## 6. 검증 하네스 (`eval.py`, 제출 X)

`main.py`의 함수를 import해서 돌린다. `main.py`가 모듈 레벨에서 바로 실행되는 구조(스켈레톤 그대로)면 import가 곤란하므로, **실행부를 `if __name__ == "__main__":` 아래로 옮긴다.** 채점은 `python main.py` 직접 실행이라 동작은 동일하다.

출력 지표:

```
                 warm RMSE   u199     u458     u529
Task1 user-CF      0.xxxx   0.xxxx*  0.xxxx*  0.xxxx*     (* 기준: <0.65 / <0.75 / <0.75)
Task2 SVD          0.xxxx   0.xxxx*  0.xxxx*  0.xxxx*     (* 기준: <0.60 / <0.70 / <0.75)
Task3 optimized    0.xxxx   0.xxxx   0.xxxx   0.xxxx
```

- warm RMSE = val 1,405행 (cold user/movie 제외) 기준
- 기준 미달 셀은 눈에 띄게 표시 → Task1/2 구현이 명세와 어긋났는지 즉시 감지
- cold 제외 규칙은 `eval.py` 안에만 둔다. `main.py`는 어떤 입력이 와도 값을 내야 하므로 제외 개념이 없다

보조 스크립트로, val의 warm 쌍에서 100개를 뽑아 `input.txt`를 만들고 `python main.py`를 **새 프로세스로** 돌려 `output.txt` 줄 수(=300)와 포맷을 검사하는 스모크 테스트를 둔다.

---

## 7. 출력 포맷 체크리스트

명세의 포맷 감점 조항에 그대로 대응한다.

- [ ] pair당 **3줄**, 순서 = User-based → Matrix factorization → Optimized
- [ ] 필드 구분자는 **공백 없는 쉼표** — `1,31,2.5000`
- [ ] `prediction_score`는 **소수점 4자리 반올림** — `"{:.4f}".format(x)`
- [ ] `input.txt` / `output.txt`는 `main.py`와 **같은 디렉터리**
- [ ] 입력 쌍 수 N → 출력 줄 수 정확히 3N. 어떤 입력에도 예외/NaN 없음
- [ ] `output.txt`는 매 실행마다 덮어쓰기 (`"w"` 모드 — 스켈레톤대로)

스켈레톤의 `write_output()`은 이 조건을 이미 만족한다. **건드리지 않는다.**

---

## 8. 제출 전 체크리스트

- [ ] **깨끗한 새 프로세스에서 `python main.py` 단독 실행** — 노트북 잔여 변수 의존 없음 (명세 경고 사항)
- [ ] `requirements.txt`에 사용 라이브러리 전부 명시 + 버전 핀. `main.py`에 `pip install` 코드 금지
- [ ] **Python 3.12에서 검증** — 명세 지정 버전. 로컬 anaconda는 3.10.9, 기본 python은 3.13이므로 `py -3.12`로 별도 확인 필요
- [ ] 입력 100쌍 기준 전체 실행 시간 측정 → 10분 한참 이내 (현 설계상 수 초 예상)
- [ ] 학습 경로에서 `ratings_val.csv`를 **절대** 읽지 않음 → `main.py`에서 `VAL_FILE` 상수를 아예 제거하거나, 읽는 코드가 없는지 grep으로 확인
- [ ] 난수 시드 고정 → 재실행 시 동일 출력
- [ ] zip 구조: `PRJ2_<학번>_<이름>.pdf`, `data/`, `main.py` (+ `requirements.txt`). `PLAN.md`, `eval.py`, `tune.py`, `.git/`, `input.txt`, `output.txt` **제외**
- [ ] 파일명 양식 준수: `PRJ2_202600000_홍길동.zip` — 학번/이름 실제 값으로 교체
- [ ] 보고서 A4 4장 이내 PDF

---

## 9. 보고서 구성 (A4 4장)

명세가 요구한 항목에 대응시킨다.

1. **Task 1 간단 설명** (~0.5장) — Pearson 수식, $I_{u,v}$ 제약을 행렬곱으로 푼 방법, 폴백 규칙
2. **Task 2 간단 설명** (~0.5장) — movie 평균 대치, SVD, K=400
3. **Task 3 상세 설명** (~2장)
   - 최적화 대상: **Task 2** / 선택 근거 = SVD-on-imputed가 학습 신호의 98%를 합성값에 쓰는 구조적 결함 (§5.1)
   - 주요 로직: baseline + biased MF(SGD, 관측치 손실), clipping, 저관측 user 블렌딩
   - 결과 분석: Task2 → Task3 RMSE 변화 표 (warm RMSE + 기준 user 3명), 개선 요소별 ablation, user 458(train 18개)에서의 거동
4. **피드백 의견** (~0.5장)

§6 하네스가 뽑는 표를 그대로 3-c에 쓰면 된다. **ablation 수치는 개발 중에 기록해 둬야 한다** — 나중에 재현하려면 번거롭다.

---

## 10. 구현 순서

1. `main.py` 실행부를 `if __name__ == "__main__":` 로 이동 + `build_dataset()` 작성
2. `eval.py` 하네스 작성 (아직 예측은 상수 3.5 → 파이프라인부터 검증)
3. Task 1 구현 → 기준 user 3명 RMSE가 0.65/0.75/0.75 아래로 내려가는지 확인
4. Task 2 구현 → 0.60/0.70/0.75 확인
5. Task 3: biased MF 구현 → clipping → 블렌딩 순으로 쌓으며 매 단계 측정값 기록
6. `tune.py`로 거친 격자 탐색 → 상수 확정해 `main.py`에 하드코딩
7. `input.txt` 100줄 스모크 테스트 (새 프로세스, Python 3.12) → 포맷·시간 검증
8. 보고서 작성 → zip 패키징 → §8 체크리스트 완주

3·4단계에서 기준을 못 맞추면 **최적화가 아니라 명세 해석 오류**를 먼저 의심한다. 특히 ① `r̄_u` 를 전체 평균이 아닌 공통 항목 평균으로 계산했는지, ② Pearson 분모를 $I_{u,v}$ 로 제한했는지, ③ 예측에서 자기 자신($v=u$)을 제외했는지 세 가지가 흔한 실수다.
