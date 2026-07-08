"""
第 2 步：引入年龄结构 age-structured Poisson tau-leap 隨機 SIR
================================================================

【方法無關、保留的改進】
- S/I/R 與人口用 int64（離散隨機模型）。
- 初始感染者按人口比例播種，避免小年齡組初期患病率偏高。
- FOI 維持 frequency-dependent（除以 N_b），與 socialmixr/SOCRATES 慣例一致。

【數值來源】
- 接觸矩陣 M：Wallinga, Teunis & Kretzschmar (2006) [提案參考文獻 6]，
  Appendix Table 2，Utrecht 1986，reciprocity-corrected（對稱）。
- 易感性 q：Zhao et al. (2026) [提案參考文獻 8]，<5歲=1.0，其餘=0.38。
- 恢復率 alpha：Boëlle et al. (2011) [提案參考文獻 37]，感染期~3天 -> alpha=1/3。
- 目標 R0：Boëlle [37] 中位數約 1.5；Wallinga [6] 流感 R0=1.73 佐證。
- N_BY_AGE：示意（最終用普查+UN資料[38]）。
"""

import numpy as np
import matplotlib.pyplot as plt


AGE_LABELS = ["0-5", "6-12", "13-19", "20-39", "40-59", "60+"]

M_WALLINGA = np.array([
    [169.14,  31.47,  17.76,  34.50,  15.83,  11.47],
    [ 31.47, 274.51,  32.31,  34.86,  20.61,  11.50],
    [ 17.76,  32.31, 224.25,  50.75,  37.52,  14.96],
    [ 34.50,  34.86,  50.75,  75.66,  49.45,  25.08],
    [ 15.83,  20.61,  37.52,  49.45,  61.26,  32.99],
    [ 11.47,  11.50,  14.96,  25.08,  32.99,  54.23],
])

Q_ZHAO = np.array([1.0, 0.38, 0.38, 0.38, 0.38, 0.38])
ALPHA = 1.0 / 3.0
TARGET_R0 = 1.5
N_BY_AGE = np.array([6000, 8000, 9000, 30000, 27000, 20000], dtype=np.int64)


def compute_R0(beta_c, q, alpha, M, N):
    """R0 = K 最大特徵值。K[a,b] = beta_c*q[a]*M[a,b]*(N[a]/N[b])/alpha。"""
    A = len(q)
    Nf = N.astype(float)
    K = np.empty((A, A))
    for a in range(A):
        for b in range(A):
            K[a, b] = beta_c * q[a] * M[a, b] * (Nf[a] / Nf[b]) / alpha
    return np.max(np.abs(np.linalg.eigvals(K)))


def beta_c_for_target_R0(target_R0, q, alpha, M, N):
    return target_R0 / compute_R0(1.0, q, alpha, M, N)


def seed_proportional(total_seed, N_by_age, rng):
    """按人口比例分配初始感染者（整數，總和=total_seed），最大餘數法。"""
    frac = total_seed * N_by_age / N_by_age.sum()
    base = np.floor(frac).astype(np.int64)
    remainder = int(total_seed - base.sum())
    order = np.argsort(-(frac - base))
    base[order[:remainder]] += 1
    return base


def tau_leap_sir_age(beta_c, q, alpha, M, N_by_age, I0_by_age,
                     tau=0.25, t_max=200.0, rng=None):
    """年齡結構版隨機 SIR，Poisson tau-leap（提案指定）。
    S/I/R/new_inf 形狀 (n_steps+1, A)，dtype int64。"""
    if rng is None:
        rng = np.random.default_rng()

    A = len(N_by_age)
    n_steps = int(round(t_max / tau))

    S = np.zeros((n_steps + 1, A), dtype=np.int64)
    I = np.zeros((n_steps + 1, A), dtype=np.int64)
    R = np.zeros((n_steps + 1, A), dtype=np.int64)
    new_inf = np.zeros((n_steps + 1, A), dtype=np.int64)
    t = np.arange(n_steps + 1) * tau

    S[0] = N_by_age - I0_by_age
    I[0] = I0_by_age

    for k in range(n_steps):
        s, i, r = S[k], I[k], R[k]

        # FOI（frequency-dependent；/N_b）：lambda_a = beta_c*q_a*sum_b M_ab*I_b/N_b
        prevalence = i / N_by_age
        lam = beta_c * q * (M @ prevalence)

        # Poisson tau-leap：每種事件數 ~ Poisson(rate * tau)（提案 §3.2）
        n_inf = rng.poisson(lam * s * tau) # 新增感染人数
        n_rec = rng.poisson(alpha * i * tau) # 新增康复人数

        # 安全截斷：新增感染不能超过现有易感人数
        n_inf = np.minimum(n_inf, s)
        n_rec = np.minimum(n_rec, i)

        S[k + 1] = s - n_inf
        I[k + 1] = i + n_inf - n_rec
        R[k + 1] = r + n_rec
        new_inf[k + 1] = n_inf

    return t, S, I, R, new_inf


if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)

    beta_c = beta_c_for_target_R0(TARGET_R0, Q_ZHAO, ALPHA, M_WALLINGA, N_BY_AGE)
    print(f"方法 = Poisson tau-leap (對齊 proposal §3.2),  tau = 0.25")
    print(f"alpha = {ALPHA:.4f}  (感染期 {1/ALPHA:.1f} 天, Boelle 2011)")
    print(f"beta_c = {beta_c:.5f},  檢查 R0 = {compute_R0(beta_c, Q_ZHAO, ALPHA, M_WALLINGA, N_BY_AGE):.3f}")

    I0_by_age = seed_proportional(30, N_BY_AGE, rng)
    print(f"按人口比例播種 I0 = {dict(zip(AGE_LABELS, I0_by_age.tolist()))}")

    t, S, I, R, new_inf = tau_leap_sir_age(
        beta_c, Q_ZHAO, ALPHA, M_WALLINGA, N_BY_AGE, I0_by_age,
        tau=0.25, t_max=200, rng=rng,
    )

    final_attack = R[-1] / N_BY_AGE
    print("\n各年齡層最終攻擊率：")
    for a, lab in enumerate(AGE_LABELS):
        print(f"  {lab:>6}: attack {final_attack[a]*100:5.1f}%   q={Q_ZHAO[a]:.2f}")

    colors = ["#1f78b4", "#33a02c", "#ff7f00", "#6a3d9a", "#e31a1c", "#666666"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    ax = axes[0]
    for a, lab in enumerate(AGE_LABELS):
        ax.plot(t, new_inf[:, a], color=colors[a], label=lab)
    ax.set_xlabel("time (days)"); ax.set_ylabel("new infections per step")
    ax.set_title("Incidence by age (Poisson tau-leap, matches proposal)\nM: Wallinga06, q: Zhao26, R0=1.5: Boelle11")
    ax.legend(title="age")

    ax = axes[1]
    bars = ax.bar(AGE_LABELS, final_attack * 100, color=colors)
    ax.set_ylabel("final attack rate (%)"); ax.set_xlabel("age group")
    ax.set_title("Final attack rate by age")
    for bar, q_ in zip(bars, Q_ZHAO):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f"q={q_:.2f}", ha="center", fontsize=8)

    plt.rcParams["axes.unicode_minus"] = False
    plt.tight_layout()
    plt.savefig("sir_age_poisson_demo.png", dpi=130)
    print("\n圖已存成 sir_age_poisson_demo.png")
