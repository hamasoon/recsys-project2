# 개발용 (제출 X). report/figures/results.json (experiments.py 출력)으로 보고서 그림을 그린다.
# 사용: python experiments.py && python make_figures.py

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report", "figures")
CM = 1 / 2.54
COL_W, FULL_W = 8.55 * CM, 17.8 * CM        # report/main.typ 2단 폭 / 전체 폭

# 색: Task 정체성은 모든 그림에서 고정 (dataviz 기본 팔레트 1~3번, all-pairs 검증 통과)
TASK3, TASK2, TASK1 = "#2a78d6", "#eb6834", "#1baf7a"
TASK_COLOR = {"task1": TASK1, "task2": TASK2, "task3": TASK3}
TASK_LABEL = {"task1": "Task 1 User-based CF", "task2": "Task 2 SVD (K=400)", "task3": "Task 3 Optimized"}
SHORT_LABEL = {"task1": "Task 1 (User-CF)", "task2": "Task 2 (SVD)", "task3": "Task 3 (최종)"}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, NEUTRAL = "#e1e0d9", "#c3c2b7", "#a3a29c"

plt.rcParams.update({
    "font.family": "Malgun Gothic",   # Noto Sans KR은 가변 폰트라 matplotlib이 굵기를 못 고름
    "font.size": 7,
    "axes.titlesize": 7.5,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 6,
    "axes.labelsize": 7,
    "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "grid.linestyle": "-",
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "xtick.major.pad": 3,
    "ytick.major.pad": 3,
    "legend.fontsize": 6.5,
    "legend.frameon": False,
    "lines.linewidth": 1.5,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "svg.fonttype": "path",
    "figure.dpi": 150,
})

DOT = dict(s=30, zorder=3, edgecolors="white", linewidths=1.0)   # >= 8px 점 + 표면색 링


def load():
    with open(os.path.join(FIG_DIR, "results.json"), encoding="utf-8") as f:
        return json.load(f)


PREVIEW_DIR = os.environ.get("FIG_PREVIEW_DIR")   # 지정하면 확인용 PNG도 저장


def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02, transparent=True)
    if PREVIEW_DIR:
        fig.savefig(os.path.join(PREVIEW_DIR, name.replace(".svg", ".png")), bbox_inches="tight",
                    pad_inches=0.02, dpi=220, facecolor="white")
    plt.close(fig)
    print("wrote", path)


def fmt4(x):
    return "{:.4f}".format(x)


# ---------- 그림 1: 명세 per-user 기준 ----------
def fig_spec_check(r):
    users = ["199", "458", "529"]
    limits = {"task1": {"199": 0.65, "458": 0.75, "529": 0.75},
              "task2": {"199": 0.60, "458": 0.70, "529": 0.75}}
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 3.4 * CM), sharey=True, sharex=True)
    for ax, task in zip(axes, ["task1", "task2", "task3"]):
        vals = [r["tasks_val"][task]["per_user"][u] for u in users]
        ys = list(range(len(users)))
        for y, u, v in zip(ys, users, vals):
            if task in limits:
                lim = limits[task][u]
                ax.plot([lim, lim], [y - 0.32, y + 0.32], color=INK2, lw=1.2, zorder=2)
                ax.text(lim + 0.006, y + 0.2, "기준 {:.2f}".format(lim), color=INK2, fontsize=6, va="center")
            ax.scatter([v], [y], color=TASK_COLOR[task], **DOT)
            ax.text(v - 0.01, y - 0.06, fmt4(v), color=INK, fontsize=6.5, ha="right", va="center")
        ax.set_title(TASK_LABEL[task] + ("" if task in limits else "  (명세 기준 없음)"))
        ax.set_yticks(ys, ["user {}".format(u) for u in users])
        ax.set_ylim(len(users) - 0.5, -0.5)
        ax.set_xlim(0.38, 0.82)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("val RMSE (해당 user의 warm 행)")
    save(fig, "fig1_spec_check.svg")


# ---------- 그림 2: Task 2 SVD의 K 의존성 ----------
def fig_task2_k(r):
    ks = [row["k"] for row in r["task2_k"]]
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL_W, 4.6 * CM))
    for ax in (a, b):
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator([1, 10, 100, 400]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: "{:g}".format(x)))
        ax.set_xlabel("K (사용한 특이값 수, 로그 축)")
        ax.axvline(400, color=AXIS, lw=0.6, zorder=1)

    tr = [row["train_observed"] for row in r["task2_k"]]
    a.plot(ks, tr, color=TASK2, marker="o", ms=3.5, mec="white", mew=0.8)
    a.set_title("(a) train 관측 평점 재현 RMSE")
    a.set_ylim(0, 1.0)
    k400 = next(row for row in r["task2_k"] if row["k"] == 400)
    a.annotate("K=400: {:.4f}".format(k400["train_observed"]), (400, k400["train_observed"]),
               xytext=(-6, 14), textcoords="offset points", ha="right", fontsize=6.5, color=INK,
               arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))

    va = [row["val"] for row in r["task2_k"]]
    b.plot(ks, va, color=TASK2, marker="o", ms=3.5, mec="white", mew=0.8)
    d = r["task2_diag"]
    for y, x, label in [(d["item_mean_only_val"], 1.05, "movie 평균만 예측 {:.4f}".format(d["item_mean_only_val"])),
                        (d["global_mean_val"], 12, "전역 평균 예측 {:.4f}".format(d["global_mean_val"]))]:
        b.axhline(y, color=NEUTRAL, lw=0.8, zorder=1)
        b.text(x, y + 0.0012, label, color=INK2, fontsize=6, va="bottom")
    b.set_title("(b) val warm RMSE")
    b.set_ylim(1.055, 1.090)
    b.annotate("K=400: {:.4f}".format(k400["val"]), (400, k400["val"]),
               xytext=(-6, -16), textcoords="offset points", ha="right", fontsize=6.5, color=INK,
               arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    save(fig, "fig2_task2_k.svg")


# ---------- 그림 3: movie 인기도 구간별 RMSE ----------
def fig_by_movie_count(r):
    rows = r["by_movie_count"]
    xs = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(COL_W, 5.0 * CM))
    for task in ["task2", "task1", "task3"]:
        ys = [row[task] for row in rows]
        ax.plot(xs, ys, color=TASK_COLOR[task], marker="o", ms=4, mec="white", mew=0.8,
                label=SHORT_LABEL[task])
    for x, row in zip(xs, rows):       # 최종 모델 값만 직접 라벨
        ax.text(x, row["task3"] - 0.06, fmt4(row["task3"]), ha="center", va="top", fontsize=6, color=INK)
    ax.set_xticks(xs, ["{}개\n(n={})".format(row["label"], row["n"]) for row in rows])
    ax.set_xlabel("해당 movie의 train 평점 수 구간")
    ax.set_ylabel("val warm RMSE")
    ax.set_ylim(0.70, 1.32)
    ax.set_xlim(-0.3, len(rows) - 0.7)
    ax.grid(axis="x", visible=False)
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(SHORT_LABEL[t]) for t in ("task1", "task2", "task3")]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower left",
              bbox_to_anchor=(0, 1.0), ncol=3, handlelength=1.4, borderaxespad=0.1, columnspacing=1.0)
    save(fig, "fig3_by_movie_count.svg")


# ---------- 그림 4: Task 3 ablation ----------
ABLATION_ROWS = [
    ("task2", "Task 2 SVD (K=400)", TASK2),
    ("task1", "Task 1 User-based CF", TASK1),
    ("global_mean", "전역 평균 μ", NEUTRAL),
    ("baseline", r"① baseline $\mu+b_u+b_i$", TASK3),
    ("latent", r"② + 잠재 요인 $p_u^{\top} q_i$", TASK3),
    ("clip", "③ + [0.5, 5] clip", TASK3),
    ("final", "④ + User-CF 블렌딩 (최종)", TASK3),
]


def fig_ablation(r):
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 5.0 * CM), sharey=True)
    for ax, split, title in [(axes[0], "holdout", "(a) train 내부 10% 홀드아웃 (선택 기준)"),
                             (axes[1], "val", "(b) ratings_val.csv warm (참고)")]:
        ys = list(range(len(ABLATION_ROWS)))
        vals = [r["ablation"][split][key]["rmse"] for key, _, _ in ABLATION_ROWS]
        ax.plot(vals[3:], ys[3:], color=TASK3, lw=0.8, alpha=0.5, zorder=2)
        for y, (key, _, color), v in zip(ys, ABLATION_ROWS, vals):
            ax.scatter([v], [y], color=color, **DOT)
            ax.text(v + 0.006, y, fmt4(v), va="center", fontsize=6.5, color=INK)
        ax.set_yticks(ys, [label for _, label, _ in ABLATION_ROWS])
        ax.set_ylim(len(ys) - 0.5, -0.5)
        ax.set_xlim(0.84, 1.12)
        ax.set_title(title)
        ax.set_xlabel("RMSE")
        ax.grid(axis="y", visible=False)
    save(fig, "fig4_ablation.svg")


# ---------- 그림 5: Task 3 하이퍼파라미터 (seed 0~4 평균) ----------
REG_RAMP = {0.05: "#86b6ef", 0.1: "#2a78d6", 0.2: "#104281"}   # 순서형: 한 hue, 밝음 -> 어두움
SWEEP_TITLES = {
    "BLEND_CF": "(b) 블렌딩 가중치 w",
    "BIAS_REG": r"(c) baseline 정규화 $\lambda_b$",
    "MF_LR": "(d) 학습률",
    "MF_EPOCHS": "(e) epoch 수",
}


def fig_hyper(r):
    cfg = r["config"]
    fig = plt.figure(figsize=(FULL_W, 8.6 * CM))
    gs = fig.add_gridspec(2, 3, hspace=0.75, wspace=0.32)

    # (a) K x λ 격자: 홀드아웃 절대 RMSE
    a = fig.add_subplot(gs[0, :2])
    for reg, color in REG_RAMP.items():
        rows = sorted((g for g in r["grid"] if g["reg"] == reg), key=lambda g: g["k"])
        ks = [g["k"] for g in rows]
        m = [g["holdout_mean"] for g in rows]
        sd = [g["holdout_std"] for g in rows]
        a.fill_between(ks, [x - y for x, y in zip(m, sd)], [x + y for x, y in zip(m, sd)],
                       color=color, alpha=0.12, lw=0)
        a.plot(ks, m, color=color, marker="o", ms=3.5, mec="white", mew=0.8, label="λ = {:g}".format(reg))
    a.set_xscale("log")
    a.xaxis.set_major_locator(FixedLocator(sorted({g["k"] for g in r["grid"]})))
    a.xaxis.set_minor_locator(NullLocator())
    a.xaxis.set_major_formatter(FuncFormatter(lambda x, _: "{:g}".format(x)))
    a.set_xlabel("잠재 요인 수 K (로그 축)")
    a.set_ylabel("홀드아웃 RMSE")
    a.set_title("(a) K × λ 격자 — 홀드아웃, seed 0~4 평균 ± 1σ")
    a.set_ylim(0.876, 0.918)
    a.yaxis.set_major_locator(FixedLocator([0.88, 0.89, 0.90, 0.91]))
    a.legend(loc="upper right", ncol=3, handlelength=1.4, borderaxespad=0.1)
    chosen = next(g for g in r["grid"] if g["k"] == cfg["MF_K"] and g["reg"] == cfg["MF_REG"])
    old = next(g for g in r["grid"] if g["k"] == 10 and g["reg"] == cfg["MF_REG"])
    a.scatter([chosen["k"]], [chosen["holdout_mean"]], s=70, facecolors="none", edgecolors=INK, lw=1.0, zorder=4)
    a.annotate("선택 K={} {:.4f}".format(chosen["k"], chosen["holdout_mean"]), (chosen["k"], chosen["holdout_mean"]),
               xytext=(0, -15), textcoords="offset points", ha="center", fontsize=6.5, color=INK)
    a.annotate("K=10 {:.4f}".format(old["holdout_mean"]), (10, old["holdout_mean"]),
               xytext=(0, -11), textcoords="offset points", ha="center", fontsize=6.5, color=INK2)

    # (b)~(e) one-at-a-time sweep: 선택 설정 대비 ΔRMSE (홀드아웃, val 공통 기준선 0)
    order = ["BLEND_CF", "BIAS_REG", "MF_LR", "MF_EPOCHS"]
    cells = [gs[0, 2], gs[1, 0], gs[1, 1], gs[1, 2]]
    for param, cell in zip(order, cells):
        ax = fig.add_subplot(cell)
        rows = r["sweeps"][param]
        xs = list(range(len(rows)))
        c = next(row for row in rows if row["value"] == cfg[param])
        h = [row["holdout_mean"] - c["holdout_mean"] for row in rows]
        hs = [row["holdout_std"] for row in rows]
        v = [row["val_mean"] - c["val_mean"] for row in rows]
        ax.axhline(0, color=AXIS, lw=0.6, zorder=1)
        ax.fill_between(xs, [x - y for x, y in zip(h, hs)], [x + y for x, y in zip(h, hs)],
                        color=TASK3, alpha=0.12, lw=0)
        ax.plot(xs, v, color=NEUTRAL, lw=1.0, marker="o", ms=2.5, mec="white", mew=0.5, label="val (참고)")
        ax.plot(xs, h, color=TASK3, marker="o", ms=3.5, mec="white", mew=0.8, label="홀드아웃 (선택 기준)")
        ci = xs[[row["value"] for row in rows].index(cfg[param])]
        ax.scatter([ci], [0], s=60, facecolors="none", edgecolors=INK, lw=1.0, zorder=4)
        ax.set_xticks(xs, ["{:g}".format(row["value"]) for row in rows])
        ax.set_title(SWEEP_TITLES[param])
        ax.set_ylabel("ΔRMSE (선택 대비)")
        ax.grid(axis="x", visible=False)
        top = max(max(h), max(v))
        ax.set_ylim(-0.004, max(0.012, top * 1.1))
        if param == "BLEND_CF":
            ax.legend(loc="upper left", handlelength=1.4, borderaxespad=0.1)
    save(fig, "fig5_hyper.svg")


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    res = load()
    fig_spec_check(res)
    fig_task2_k(res)
    fig_by_movie_count(res)
    fig_ablation(res)
    fig_hyper(res)
