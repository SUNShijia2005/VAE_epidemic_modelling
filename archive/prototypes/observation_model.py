"""
第4步，观测模型：把「真实感染」变成「监测看到的通报病例」
真实感染(上帝视角,谁都看不到)→ 延迟卷积(往后拖) → 乘 rho(砍到只剩1%左右) 
→ 负二项抽样(变成一个抖动的整数) → 通报病例(现实中疾控部门真正拿到手的数据)
=====================================================
对齐 proposal §3.2 后半。模拟器产生潜在感染后，再套一个显式观测过程：
  (1) 通报率 rho   ：每个真实感染只有一定概率被通报（流感约 1%，Reed 等 [39]）。
  (2) 通报延迟     ：感染到被通报之间有几天延迟（用一个延迟分布卷积）。
  (3) 过度离散     ：通报数的方差比 Poisson 更大，用负二项分布（NegBinomial）刻画。

这三个参数在正式研究里是 nuisance（干扰参数），跟 theta 一起被推断后边缘化掉
（proposal §3.4 / §3.2）。这里先把机制实现出来、看效果。

【数值来源】
- 通报率 rho：Reed 等 [39] 估每个通报病例约对应 79 个感染（90%区间 47–148），
  即 rho ~ 1/79 ≈ 1.3%。本档用此量级。
- 延迟、过度离散：proposal 说给弱信息先验（延迟是常规流感监测的几天滞后；
  过度离散从近 Poisson 到明显过度离散）。本档用示意值并标明。

依赖：需与 sir_age_hh.py 放在同一资料夹（本档从它 import 模拟器）。
"""

import numpy as np
import matplotlib.pyplot as plt

from sir_age_hh import (
    build_population, beta_c_for_target_R0, beta_h_from_SAR, simulate,
    Q_ZHAO, ALPHA, M_WALLINGA, AGE_LABELS,
    TARGET_COMMUNITY_R0,
)


def to_daily(new_inf_by_age, tau):
    """把每步（tau=0.25）的新感染汇总成每日新感染。回传 (n_days, A)。"""
    spd = int(round(1.0 / tau))                 # 每天的步数
    rec = new_inf_by_age[1:]                     # 丢掉 t=0
    n_full = (len(rec) // spd) * spd
    daily = rec[:n_full].reshape(-1, spd, rec.shape[1]).sum(axis=1)
    return daily


def nb_sample(mean, size, rng):
    """以指定平均 mean 与离散参数 size 抽负二项。size 越小越离散；size->inf 趋近 Poisson。
    NegBinomial 平均 = mean，方差 = mean + mean^2/size。
    把预期值变成一个更爱忽高忽低的随机整数"""
    mean = np.asarray(mean, dtype=float)
    out = np.zeros(mean.shape, dtype=np.int64)
    pos = mean > 0
    p = size / (size + mean[pos])               # 转成 numpy 的 (n=size, p) 参数化
    out[pos] = rng.negative_binomial(size, p)
    return out


def observation_model(new_inf_by_age, tau, rho, delay_pmf, nb_size, rng):
    """把潜在每步感染 -> 每日通报病例（按年龄）。
    回传 true_daily, reported_daily，皆形状 (n_days, A)。"""
    true_daily = to_daily(new_inf_by_age, tau)           # 真实每日感染
    n_days, A = true_daily.shape

    # (2) 延迟：把每日真实感染与延迟分布卷积，得到「会在某天被通报」的期望时序
    delayed = np.zeros_like(true_daily, dtype=float)
    for a in range(A):
        conv = np.convolve(true_daily[:, a], delay_pmf)[:n_days]
        delayed[:, a] = conv

    # (1)+(3) 通报率与过度离散：期望通报 = rho * delayed，实际通报 ~ NegBinomial
    expected_reported = rho * delayed
    reported_daily = np.zeros_like(true_daily, dtype=np.int64)
    for a in range(A):
        reported_daily[:, a] = nb_sample(expected_reported[:, a], nb_size, rng)

    return true_daily, reported_daily


if __name__ == "__main__":
    rng = np.random.default_rng(seed=7)

    # ---- 先跑第 3 步的双层模拟器，拿到潜在感染 ----
    age, hh, n_k = build_population(50000, rng)
    N_by_age = np.bincount(age, minlength=len(AGE_LABELS))
    beta_c = beta_c_for_target_R0(TARGET_COMMUNITY_R0, Q_ZHAO, ALPHA, M_WALLINGA, N_by_age)
    beta_h = beta_h_from_SAR(0.15, ALPHA)
    I0_idx = rng.choice(len(age), size=40, replace=False)
    tau = 0.25
    t, new_inf_by_age, ever_inf = simulate(
        beta_c, beta_h, Q_ZHAO, ALPHA, M_WALLINGA, age, hh, n_k, N_by_age,
        I0_idx, tau=tau, t_max=220, rng=rng,
    )

    # ---- 观测模型参数 ----
    rho = 1.0 / 79.0                                     # 通报率（Reed [39]）
    # 延迟分布（示意）：感染到通报的天数 PMF，平均约 3 天
    delay_pmf = np.array([0.02, 0.10, 0.22, 0.26, 0.20, 0.12, 0.06, 0.02])
    delay_pmf = delay_pmf / delay_pmf.sum()
    nb_size = 5.0                                        # 过度离散（示意；越小越离散）

    true_daily, reported_daily = observation_model(
        new_inf_by_age, tau, rho, delay_pmf, nb_size, rng
    )

    true_total = true_daily.sum(axis=1)
    reported_total = reported_daily.sum(axis=1)
    days = np.arange(len(true_total))

    print(f"通报率 rho = {rho:.4f}  (1/79, Reed [39])")
    print(f"延迟分布平均 = {(np.arange(len(delay_pmf))*delay_pmf).sum():.2f} 天")
    print(f"过度离散 nb_size = {nb_size}（方差 = 均值 + 均值^2/{nb_size:.0f}）")
    print(f"\n真实总感染 = {true_total.sum()}")
    print(f"通报总病例 = {reported_total.sum()}  (约为真实的 {reported_total.sum()/true_total.sum()*100:.1f}%)")
    print(f"真实高峰在第 {true_total.argmax()} 天，通报高峰在第 {reported_total.argmax()} 天（看延迟造成的右移）")

    # ---- 画图：真实 vs 通报 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(days, true_total, color="#999999", lw=2, label="true infections (latent)")
    ax2 = ax.twinx()
    ax2.bar(days, reported_total, color="#d95f0e", alpha=0.6, label="reported cases")
    ax.set_xlabel("day"); ax.set_ylabel("true infections / day", color="#666666")
    ax2.set_ylabel("reported cases / day", color="#d95f0e")
    ax.set_title("Latent infections vs observed reports (total)\nunder-reported, delayed, overdispersed")
    ax.legend(loc="upper left"); ax2.legend(loc="upper right")

    ax = axes[1]
    colors = ["#1f78b4", "#33a02c", "#ff7f00", "#6a3d9a", "#e31a1c", "#666666"]
    for a, lab in enumerate(AGE_LABELS):
        ax.plot(days, reported_daily[:, a], color=colors[a], label=lab, lw=1)
    ax.set_xlabel("day"); ax.set_ylabel("reported cases / day")
    ax.set_title("Observed reports by age\n(this noisy, sparse series is the real data x)")
    ax.legend(title="age", fontsize=8)

    plt.rcParams["axes.unicode_minus"] = False
    plt.tight_layout()
    plt.savefig("observation_demo.png", dpi=130)
    print("\n图已存成 observation_demo.png")
