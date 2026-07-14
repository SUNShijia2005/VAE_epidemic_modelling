"""
第 1 步：最小核心 —— tau-leap 隨機 SIR 模擬器
===============================================

這是整個專案的地基。之後的年齡結構、家戶結構、觀測模型，
都是在這個「事件 -> 速率 -> tau-leap 一步更新」的骨架上往外長。

模型（隨機 SIR）：
    狀態 (S, I, R) = 易感、感染、康復 的人數
    事件 1  感染：S -> I，速率 = beta * S * I / N    
    事件 2  康復：I -> R，速率 = alpha * I

tau-leap 的想法（你提案 3.2 寫的）：
    不要一個事件一個事件地走（那是 Gillespie，精確但很慢），
    而是固定一個時間步 tau，假設這段時間內速率近似不變，
    把每種事件在這段時間發生的「次數」當作 Poisson 隨機數一次抽出來。
        感染次數 n_inf ~ Poisson(感染速率 * tau)
        康復次數 n_rec ~ Poisson(康復速率 * tau)
    然後一次更新狀態。這就是「用機率模型生成資料」。
"""

import numpy as np
import matplotlib.pyplot as plt


def tau_leap_sir(beta, alpha, N, I0, tau=0.5, t_max=120.0, rng=None):
    """
    模擬一條隨機 SIR 流行病軌跡。

    參數（這些就是之後要被推斷的 theta 的雛形）
    ----
    beta  : 傳播率（每次有效接觸的傳染強度）
    alpha : 恢復率（1/alpha = 平均感染期天數）
    N     : 總人口
    I0    : 一開始的感染人數
    tau   : 時間步長（天）。越小越接近精確，但越慢
    t_max : 模擬總天數
    rng   : numpy 亂數產生器（給定 seed 才能重現結果）

    回傳
    ----
    t : 時間點陣列
    S, I, R       : 各時間點的人數
    new_inf       : 每一步「新增感染數」—— 這就是流行病監測看到的 incidence
    """
    if rng is None:
        rng = np.random.default_rng()

    n_steps = int(round(t_max / tau))

    # 開一些陣列存軌跡
    S = np.zeros(n_steps + 1)
    I = np.zeros(n_steps + 1)
    R = np.zeros(n_steps + 1)
    new_inf = np.zeros(n_steps + 1)   # 每步的新感染（incidence 時間序列）
    t = np.arange(n_steps + 1) * tau

    # 初始狀態
    S[0] = N - I0
    I[0] = I0
    R[0] = 0.000

    for k in range(n_steps):
        s, i, r = S[k], I[k], R[k]

        # --- 1. 算每個事件的速率 ---
        rate_inf = beta * s * i / N   # 感染速率（質量作用）
        rate_rec = alpha * i          # 康復速率

        # --- 2. tau-leap：把事件次數當 Poisson 抽出來 ---
        n_inf = rng.poisson(rate_inf * tau)
        n_rec = rng.poisson(rate_rec * tau)

        # --- 3. 防止「抽超過實際可用人數」(Poisson 可能抽過頭) ---
        n_inf = min(n_inf, int(s))    # 感染不能超過現有易感者
        n_rec = min(n_rec, int(i))    # 康復不能超過現有感染者

        # --- 4. 一步更新狀態 ---
        S[k + 1] = s - n_inf
        I[k + 1] = i + n_inf - n_rec
        R[k + 1] = r + n_rec
        new_inf[k + 1] = n_inf

    return t, S, I, R, new_inf


if __name__ == "__main__":
    # ---- 跑一條軌跡看看 ----
    rng = np.random.default_rng(seed=42)   # 固定 seed 才能重現

    # 一組合理的流感樣參數：R0 = beta/alpha ~ 2.0
    beta = 0.4     # 傳播率
    alpha = 0.2    # 恢復率 -> 平均感染期 5 天
    N = 10_000     # 總人口
    I0 = 10        # 初始感染

    t, S, I, R, new_inf = tau_leap_sir(beta, alpha, N, I0, tau=0.5, t_max=120, rng=rng)

    print(f"R0 = beta/alpha = {beta/alpha:.2f}")
    print(f"流行高峰感染人數 = {I.max():.0f}，發生在第 {t[I.argmax()]:.1f} 天")
    print(f"流行結束時累計康復（總感染規模）= {R[-1]:.0f}（占人口 {R[-1]/N*100:.1f}%）")

    # ---- 畫多條軌跡，凸顯「隨機性」----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # 左圖：單條軌跡的 S/I/R
    ax = axes[0]
    ax.plot(t, S, label="S (susceptible)", color="#2c7fb8")
    ax.plot(t, I, label="I (infectious)", color="#d95f0e")
    ax.plot(t, R, label="R (recovered)", color="#31a354")
    ax.set_xlabel("time (days)")
    ax.set_ylabel("number of people")
    ax.set_title("A single stochastic SIR trajectory")
    ax.legend()

    # 右圖：20 條獨立軌跡的「每日新增感染」—— 看隨機波動
    ax = axes[1]
    for s in range(20):
        rng_s = np.random.default_rng(seed=s)
        ts, _, _, _, ni = tau_leap_sir(beta, alpha, N, I0, tau=0.5, t_max=120, rng=rng_s)
        ax.plot(ts, ni, color="#d95f0e", alpha=0.35, lw=1)
    ax.set_xlabel("time (days)")
    ax.set_ylabel("new infections per step (incidence)")
    ax.set_title("20 runs, same parameters, different outcomes\n(demographic stochasticity)")

    plt.rcParams["axes.unicode_minus"] = False
    plt.tight_layout()
    plt.savefig("sir_tauleap_demo.png", dpi=130)
    print("\n圖已存成 sir_tauleap_demo.png")
