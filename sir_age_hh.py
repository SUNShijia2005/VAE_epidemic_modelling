"""
第 3 步：年齡 + 家戶 雙層結構 Poisson tau-leap 隨機 SIR
==========================================================
在第 2 步(年齡)之上加入家戶層。對齊 proposal §1, §3.2。

【兩條感染管道】(proposal §1 的三個事件:社區感染、家戶內感染、康復)
  個體 j（年齡 a，家戶 k）的總感染風險 = 社區 + 家戶內：
    社區：  lambda^comm_a = beta_c * q_a * sum_b M_ab * I_b / N_b      （同第 2 步）
    家戶內：lambda^hh_k   = beta_h * i_k / (n_k - 1)                   （Barlow [3]）
  其中 i_k 是家戶 k 內的感染人數，n_k 家戶總人數，n_k-1 是家戶內可接觸人數。
  n_k = 1 的家戶沒有家戶內接觸，家戶內風險 = 0。

【方法】Poisson tau-leap（proposal §3.2），小 τ=0.25。
  每步：社區感染數（按年齡分組）、家戶內感染數（按家戶分組）、康復數，各抽 Poisson。
  先套社區感染、再在「剩下的易感者」上套家戶感染，杜絕同一人被兩管道重複感染。
  康復只從「本步開始時就已感染」者中抽，新感染者本步不康復。

【beta_h 錨定】家戶 secondary attack rate（SAR）。
  指數型感染期下，n=2 家戶中 1 個原發病例使唯一接觸者被感染的機率
  SAR = beta_h / (beta_h + alpha)  =>  beta_h = alpha * SAR / (1 - SAR)。
  本檔取 SAR=0.15（Martínez-Baz 等, Navarra [32]，14–19%）；
  另一端 MoSAIC [33] 約 2.3–7.6%（更小的 beta_h）。

【數值來源】（同前）M: Wallinga06 [6]；q: Zhao26 [8]；alpha: Boelle11 [37]；
  社區目標 R0=1.5: Boelle11 [37]。
【示意值】家戶大小分布（最終用 UN 家戶資料 [38]）；年齡在家戶內獨立指派
  （真實模型會讓家戶年齡組成相關，如親子同戶；此處簡化並標明）。
"""

import numpy as np
import matplotlib.pyplot as plt


AGE_LABELS = ["0-5", "6-12", "13-19", "20-39", "40-59", "60+"]

# a 年龄段的一个人,平均一天大概会接触多少个 b 年龄段的人【接触强度】
M_WALLINGA = np.array([
    [169.14,  31.47,  17.76,  34.50,  15.83,  11.47],
    [ 31.47, 274.51,  32.31,  34.86,  20.61,  11.50],
    [ 17.76,  32.31, 224.25,  50.75,  37.52,  14.96],
    [ 34.50,  34.86,  50.75,  75.66,  49.45,  25.08],
    [ 15.83,  20.61,  37.52,  49.45,  61.26,  32.99],
    [ 11.47,  11.50,  14.96,  25.08,  32.99,  54.23],
])
Q_ZHAO = np.array([1.0, 0.38, 0.38, 0.38, 0.38, 0.38]) #每个年龄段被传染的难易程度
ALPHA = 1.0 / 3.0 #平均感染期 3 天
TARGET_COMMUNITY_R0 = 1.5

# 年齡比例（示意；指派個體年齡用）
AGE_PROP = np.array([0.06, 0.08, 0.09, 0.30, 0.27, 0.20]) #人口里各年龄段各占多少比例

# 家戶大小分布（示意；最終用 UN [38]）。size 1..6
HH_SIZES = np.array([1, 2, 3, 4, 5, 6])
HH_SIZE_PROB = np.array([0.28, 0.34, 0.16, 0.13, 0.06, 0.03])


def compute_community_R0(beta_c, q, alpha, M, N):
    A = len(q); Nf = N.astype(float); K = np.empty((A, A))
    for a in range(A):
        for b in range(A):
            K[a, b] = beta_c * q[a] * M[a, b] * (Nf[a] / Nf[b]) / alpha
    return np.max(np.abs(np.linalg.eigvals(K)))


def beta_c_for_target_R0(target, q, alpha, M, N):
    return target / compute_community_R0(1.0, q, alpha, M, N)


def beta_h_from_SAR(SAR, alpha):
    """n=2 家戶 SAR = beta_h/(beta_h+alpha) -> beta_h = alpha*SAR/(1-SAR)。"""
    return alpha * SAR / (1.0 - SAR)


def build_population(target_N, rng):
    """造家戶與個體。回傳 age(個體年齡層), hh(個體家戶編號), n_k(各家戶人數)。同一户里的人年龄完全是随机凑的[可以改进]"""
    sizes = []
    total = 0
    while total < target_N:
        s = rng.choice(HH_SIZES, p=HH_SIZE_PROB)
        sizes.append(int(s)); total += int(s)
    n_k = np.array(sizes, dtype=np.int64)
    K = len(n_k)
    hh = np.repeat(np.arange(K), n_k)                       # 每個個體的家戶編號
    N = len(hh)
    age = rng.choice(len(AGE_LABELS), size=N, p=AGE_PROP)   # 個體年齡層（獨立指派）
    return age.astype(np.int64), hh.astype(np.int64), n_k   # age(每个人的年龄段)、hh(每个人的家户编号)、n_k(每户各有几人)


def choose_per_group(cand_idx, groups, k_per_group, rng):
    """在每個 group 內隨機選 k_per_group 個 candidate（向量化）。回傳被選中的個體索引。"""
    # 改状态(变成感染者)、更新历史记录(标记曾感染)、计入统计(记录这一步新增了几例)
    if len(cand_idx) == 0:
        return np.empty(0, dtype=np.int64)
    u = rng.random(len(cand_idx))
    order = np.lexsort((u, groups))            # 先按 group、再按隨機鍵排序
    g_sorted = groups[order]
    pos = np.arange(len(cand_idx))
    change = np.concatenate(([True], g_sorted[1:] != g_sorted[:-1]))
    group_start = np.maximum.accumulate(np.where(change, pos, 0))
    rank_in_group = pos - group_start          # 組內名次 0,1,2,...
    chosen = rank_in_group < k_per_group[g_sorted]
    return cand_idx[order][chosen]


def simulate(beta_c, beta_h, q, alpha, M, age, hh, n_k, N_by_age,
             I0_idx, tau=0.25, t_max=220.0, rng=None):
    """雙層 Poisson tau-leap。回傳 t, new_inf_by_age(每步各年齡新感染), ever_infected(個體是否曾感染)。"""
    if rng is None:
        rng = np.random.default_rng()
    N = len(age); K = len(n_k)
    A = len(N_by_age)
    n_steps = int(round(t_max / tau)) #总共要跑多少步

    st = np.zeros(N, dtype=np.int8)            # 每个人此刻的状态。0=S, 1=I, 2=R
    st[I0_idx] = 1                             #初始感染者
    ever_inf = np.zeros(N, dtype=bool)         #"曾经感染过没"的历史记录
    ever_inf[I0_idx] = True

    nk_minus1 = np.maximum(n_k - 1, 1)         # 防 n=1 除零（n=1 時家戶風險另設 0）
    is_size1 = (n_k == 1)

    t = np.arange(n_steps + 1) * tau
    new_inf_by_age = np.zeros((n_steps + 1, A), dtype=np.int64) #行:时间点,列:年龄段

    for step in range(1, n_steps + 1):
        S_mask = (st == 0); I_mask = (st == 1)
        S_idx = np.where(S_mask)[0]
        I_idx_start = np.where(I_mask)[0]
        if len(I_idx_start) == 0:
            break                              # 疫情結束

        # 本步開始時的全域與家戶內感染人數（tau-leap：率在 τ 內視為固定),统计这一刻"各年龄段、各家户各有多少感染者"
        I_by_age = np.bincount(age[I_idx_start], minlength=A)
        i_k = np.bincount(hh[I_idx_start], minlength=K)

        # --- 管道1：社區感染（按年齡分組）---
        comm_haz = beta_c * q * (M @ (I_by_age / N_by_age))   # 每個易感者的社區風險(長度A)
        S_age = age[S_idx]
        S_a = np.bincount(S_age, minlength=A)                  # 各年齡現有易感數
        mean_comm = comm_haz * S_a * tau                       # 各年齡社區新感染期望數
        n_comm = np.minimum(rng.poisson(mean_comm), S_a)       # 各年齡社區新感染數
        new_comm_idx = choose_per_group(S_idx, S_age, n_comm, rng)

        # 套用社區感染,把这批人标记为感染者,同时记入历史
        st[new_comm_idx] = 1
        ever_inf[new_comm_idx] = True

        # --- 管道2：家戶內感染（在「剩下的」易感者上，按家戶分組）---
        S_mask2 = (st == 0)
        S_idx2 = np.where(S_mask2)[0]
        hh_haz_k = np.where(is_size1, 0.0, beta_h * i_k / nk_minus1)  # 每個易感者的家戶風險(長度K)
        s_k_now = np.bincount(hh[S_idx2], minlength=K)               # 剩餘易感(按家戶)
        mean_hh = hh_haz_k * s_k_now * tau
        n_hh = np.minimum(rng.poisson(mean_hh), s_k_now)            # 各家戶內新感染數
        new_hh_idx = choose_per_group(S_idx2, hh[S_idx2], n_hh, rng)

        st[new_hh_idx] = 1
        ever_inf[new_hh_idx] = True

        # 記錄本步新感染（兩管道合計，按年齡）= 監測看到的 incidence
        all_new = np.concatenate([new_comm_idx, new_hh_idx])
        if len(all_new):
            new_inf_by_age[step] = np.bincount(age[all_new], minlength=A)

        # --- 康復：只從本步開始時已感染者抽，按泊松分布决定几人康复---
        mean_rec = alpha * len(I_idx_start) * tau
        n_rec = min(rng.poisson(mean_rec), len(I_idx_start))
        if n_rec > 0:
            rec_idx = rng.choice(I_idx_start, size=n_rec, replace=False)
            st[rec_idx] = 2

    return t[:step + 1], new_inf_by_age[:step + 1], ever_inf


def household_final_size_distribution(ever_inf, hh, n_k):
    """各家戶大小下，最終被感染人數的分布。回傳 dict: size -> 機率向量(0..size)。"""
    inf_per_hh = np.bincount(hh[ever_inf], minlength=len(n_k))   # 每戶最終感染人數
    dist = {}
    for size in sorted(set(n_k.tolist())): #对每一种户型大小,单独统计一次
        mask = (n_k == size)
        counts = np.bincount(inf_per_hh[mask], minlength=size + 1)[:size + 1]
        dist[size] = counts / counts.sum()
    return dist


if __name__ == "__main__":
    rng = np.random.default_rng(seed=7)

    age, hh, n_k = build_population(50000, rng)
    N = len(age); K = len(n_k)
    N_by_age = np.bincount(age, minlength=len(AGE_LABELS))

    beta_c = beta_c_for_target_R0(TARGET_COMMUNITY_R0, Q_ZHAO, ALPHA, M_WALLINGA, N_by_age)
    SAR_target = 0.15
    beta_h = beta_h_from_SAR(SAR_target, ALPHA)

    print(f"人口 N={N}，家戶數 K={K}，平均家戶人數={N/K:.2f}")
    print(f"方法 = 雙層 Poisson tau-leap (對齊 proposal §3.2), tau=0.25")
    print(f"社區 R0 = {compute_community_R0(beta_c, Q_ZHAO, ALPHA, M_WALLINGA, N_by_age):.3f}  (beta_c={beta_c:.5f})")
    print(f"beta_h = {beta_h:.4f}  (錨定 n=2 家戶 SAR={SAR_target:.0%}, Navarra [32])")

    # 按人口比例播種約 40 個初始感染者
    I0_idx = rng.choice(N, size=40, replace=False)

    t, new_inf_by_age, ever_inf = simulate(
        beta_c, beta_h, Q_ZHAO, ALPHA, M_WALLINGA, age, hh, n_k, N_by_age,
        I0_idx, tau=0.25, t_max=220, rng=rng,
    )

    # 各年齡最終攻擊率
    final_inf_by_age = np.array([ever_inf[age == a].sum() for a in range(len(AGE_LABELS))])
    print("\n各年齡層最終攻擊率：")
    for a, lab in enumerate(AGE_LABELS):
        print(f"  {lab:>6}: {final_inf_by_age[a]/N_by_age[a]*100:5.1f}%")

    # 家戶最終規模分布
    dist = household_final_size_distribution(ever_inf, hh, n_k)
    print("\n家戶最終規模分布（各家戶大小下，最終感染人數的比例）：")
    for size, pv in dist.items():
        s = "  ".join(f"{j}:{pv[j]*100:4.1f}%" for j in range(len(pv)))
        print(f"  size {size}: {s}")

    # ---- 畫圖：左=各年齡 incidence；右=家戶最終規模分布 ----
    colors = ["#1f78b4", "#33a02c", "#ff7f00", "#6a3d9a", "#e31a1c", "#666666"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for a, lab in enumerate(AGE_LABELS):
        ax.plot(t, new_inf_by_age[:, a], color=colors[a], label=lab)
    ax.set_xlabel("time (days)"); ax.set_ylabel("new infections per step")
    ax.set_title("Community incidence by age (data component 1)\ntwo-layer Poisson tau-leap")
    ax.legend(title="age", fontsize=8)

    ax = axes[1]
    sizes = sorted(dist.keys())
    width = 0.8 / max(len(sizes), 1)
    for si, size in enumerate(sizes):
        pv = dist[size]
        xs = np.arange(len(pv)) + (si - len(sizes)/2) * width
        ax.bar(xs, pv, width=width, label=f"size {size}")
    ax.set_xlabel("number ever infected in household")
    ax.set_ylabel("probability")
    ax.set_title("Household final-size distribution (data component 2)\n*the key info that resolves identifiability*")
    ax.legend(fontsize=8, ncol=2)

    plt.rcParams["axes.unicode_minus"] = False
    plt.tight_layout()
    plt.savefig("sir_age_hh_demo.png", dpi=130)
    print("\n圖已存成 sir_age_hh_demo.png")
