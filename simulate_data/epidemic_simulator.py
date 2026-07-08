"""
年齡 + 家戶 雙層結構 Poisson tau-leap 隨機 SIR ＋ 觀測模型 ＋ 先驗 ＋ simulate(θ)→x
================================================================================
本檔把原本分散在三個檔（sir_age_hh / observation_model / simulate_interface）的
內容合併成單一「模擬器」模組，對齊 proposal §1, §3.2, §3.3。它提供：

  1. 人口建構        build_population
  2. 雙層 tau-leap   simulate            （社區層 M/βc/q + 家戶層 βh/ψH）
  3. 觀測模型        observation_model   （通報率 ρ、通報延遲、過度離散）
  4. 先驗            sample_prior        （文獻錨定，抽完整 ϑ + 觀測 nuisance）
  5. 乾淨接口        simulate_theta      （θ → x = {incidence, hh_finalsize}）

================================================================================

  · ψH（家戶內接觸結構）—— 家戶內每個易感者的風險由
        λ^hh = βh · i_k / (n_k - 1)**ψH
    給出。ψH=1 → 純頻率依賴（frequency-dependent，舊版行為）；
    ψH=0 → 純密度依賴（density-dependent）；中間連續插值。
    ψH 由「家戶最終規模分布」識別：密度端會讓大家戶更容易整戶感染，
    這正是 proposal 說「ψH is identified from the household final-size data」的機制，
    且對應 House-Keeling [1] / Barlow [3] 對兩種家戶內混合機制的討論。

  · ψM（年齡接觸矩陣形狀）—— 由 1 維擴為 2 維，貼近 proposal「以一組
    低維參數表示接觸矩陣的形狀、連同不確定性一起推斷」[7]：
        ψM = (ψM_assort, ψM_inter)
        ψM_assort : 同齡聚集強度（縮放對角線）
        ψM_inter  : 代際/跨齡混合強度（縮放遠離對角線的元素）
    矩陣仍保持對稱，並把最大特徵值歸一化為 1（scale 鎖死，見 proposal §3.2
    對 βc 與 M 尺度共線的處理）。

  · 通報延遲 —— 由固定常數改為「可被推斷的 nuisance」：延遲用 Gamma PMF
    參數化，其平均 delay_mean 從先驗抽（proposal §3.2/§3.3 明列延遲為三個
    需推斷後邊緣化的觀測 nuisance 之一）。

  · R0 / SAR —— 仍記錄，但明確標記為「診斷/衍生量」，不是 VAE 回歸目標
    （它們與 βc / βh 一一對應、完全共線；回歸目標請用 βc/βh，見
     generate_training_data.py 的 THETA_TARGETS_* 定義）。

【數值來源】M: Wallinga06 [6]；q: Zhao26 [8]；α: Boelle11 [37]；
  社區 R0∈[1.2,2.3]: Boelle11 [37]；家戶 SAR: MoSAIC [33] ~ Navarra [32]；
  通報率 ρ~1%: Reed11 [39]。家戶大小分布（示意；最終用 UN [38]）。
"""

import numpy as np


# ============================================================================
# 固定輸入：年齡分層、參考接觸矩陣、人口結構（proposal §3.2：人口是固定輸入）
# ============================================================================

AGE_LABELS = ["0-5", "6-12", "13-19", "20-39", "40-59", "60+"]
A = len(AGE_LABELS)

# a 年齡段的一個人，平均一天大概接觸多少個 b 年齡段的人（接觸強度；Wallinga [6]）
M_WALLINGA = np.array([
    [169.14,  31.47,  17.76,  34.50,  15.83,  11.47],
    [ 31.47, 274.51,  32.31,  34.86,  20.61,  11.50],
    [ 17.76,  32.31, 224.25,  50.75,  37.52,  14.96],
    [ 34.50,  34.86,  50.75,  75.66,  49.45,  25.08],
    [ 15.83,  20.61,  37.52,  49.45,  61.26,  32.99],
    [ 11.47,  11.50,  14.96,  25.08,  32.99,  54.23],
])

ALPHA = 1.0 / 3.0                                    # 平均感染期 3 天（示意基準）
TARGET_COMMUNITY_R0 = 1.5                            # 社區 R0 基準（Boelle [37]）

AGE_PROP = np.array([0.06, 0.08, 0.09, 0.30, 0.27, 0.20])   # 各年齡人口比例（示意）

# 家戶大小分布（示意；最終用 UN [38]）。size 1..6
HH_SIZES_FIXED = [1, 2, 3, 4, 5, 6]
HH_SIZE_PROB = np.array([0.28, 0.34, 0.16, 0.13, 0.06, 0.03])

# x 形狀相關常數
T_FIXED = 180                                        # incidence 固定天數
HH_VEC_LEN = sum(s + 1 for s in HH_SIZES_FIXED)      # = 27


# ============================================================================
# 1. 社區層：R0 ↔ βc 換算（用於由先驗 R0 反推 βc）
# ============================================================================

def compute_community_R0(beta_c, q, alpha, M, N):
    """社區層 next-generation 矩陣的主特徵值。"""
    Nf = N.astype(float)
    K = np.empty((A, A))
    for a in range(A):
        for b in range(A):
            K[a, b] = beta_c * q[a] * M[a, b] * (Nf[a] / Nf[b]) / alpha
    return np.max(np.abs(np.linalg.eigvals(K)))


def beta_c_for_target_R0(target, q, alpha, M, N):
    return target / compute_community_R0(1.0, q, alpha, M, N)


# ============================================================================
# 2. 家戶層：SAR ↔ βh 換算
# ============================================================================

def beta_h_from_SAR(SAR, alpha):
    """n=2 家戶 SAR = βh/(βh+α)  ->  βh = α·SAR/(1-SAR)。
    註：此換算取 ψH=1（頻率依賴、n=2 時分母 (n-1)**ψH = 1），作為 βh 的
    錨定尺度；ψH≠1 只改變風險隨家戶大小的形狀，不改 n=2 的 SAR 定義。"""
    return alpha * SAR / (1.0 - SAR)


# ============================================================================
# 3. 年齡接觸矩陣形狀 M(ψM)  —— 2 維參數化
# ============================================================================

def build_M_shape(psi_M, M0):
    """2 維形狀參數 ψM = (assort, inter)：
        assort : 同齡聚集強度 —— 縮放對角線   diag *= exp(assort)
        inter  : 跨齡混合強度 —— 縮放非對角   offdiag *= exp(inter * d_norm)
                 其中 d_norm 是年齡塊距離（越遠受 inter 影響越大）
    之後對稱化並把最大特徵值歸一化為 1（scale 鎖死，proposal §3.2）。
    ψM = (0, 0) → 參考矩陣形狀。"""
    assort, inter = float(psi_M[0]), float(psi_M[1])
    M = M0.copy().astype(float)

    # 年齡塊距離矩陣（|a-b|，正規化到 0..1）
    idx = np.arange(A)
    dist = np.abs(idx[:, None] - idx[None, :]) / (A - 1)   # 對角=0，最遠=1

    # 非對角：距離越遠，inter 影響越大
    off_scale = np.exp(inter * dist)
    np.fill_diagonal(off_scale, 1.0)
    M = M * off_scale

    # 對角：同齡聚集
    np.fill_diagonal(M, np.diag(M0) * np.exp(assort))

    # 對稱化（尊重接觸互惠；proposal §3.2）＋ 主特徵值歸一
    M = 0.5 * (M + M.T)
    ev = np.max(np.abs(np.linalg.eigvals(M)))
    return M / ev


# ============================================================================
# 4. 人口建構
# ============================================================================

def build_population(target_N, rng):
    """造家戶與個體。回傳 age(個體年齡層), hh(個體家戶編號), n_k(各家戶人數)。
    同一戶內年齡獨立指派（簡化；真實模型家戶年齡組成相關，如親子同戶）。"""
    sizes, total = [], 0
    hh_sizes_arr = np.array(HH_SIZES_FIXED)
    while total < target_N:
        s = rng.choice(hh_sizes_arr, p=HH_SIZE_PROB)
        sizes.append(int(s)); total += int(s)
    n_k = np.array(sizes, dtype=np.int64)
    hh = np.repeat(np.arange(len(n_k)), n_k)
    N = len(hh)
    age = rng.choice(A, size=N, p=AGE_PROP)
    return age.astype(np.int64), hh.astype(np.int64), n_k


# ============================================================================
# 5. 雙層 Poisson tau-leap 模擬器
# ============================================================================

def _choose_per_group(cand_idx, groups, k_per_group, rng):
    """在每個 group 內隨機選 k_per_group 個 candidate（向量化）。"""
    if len(cand_idx) == 0:
        return np.empty(0, dtype=np.int64)
    u = rng.random(len(cand_idx))
    order = np.lexsort((u, groups))
    g_sorted = groups[order]
    pos = np.arange(len(cand_idx))
    change = np.concatenate(([True], g_sorted[1:] != g_sorted[:-1]))
    group_start = np.maximum.accumulate(np.where(change, pos, 0))
    rank_in_group = pos - group_start
    chosen = rank_in_group < k_per_group[g_sorted]
    return cand_idx[order][chosen]


def simulate(beta_c, beta_h, psi_H, q, alpha, M, age, hh, n_k, N_by_age,
             I0_idx, tau=0.25, t_max=220.0, rng=None):
    """雙層 Poisson tau-leap。

    社區層：λ^comm_a = βc · q_a · Σ_b M_ab · I_b / N_b
    家戶層：λ^hh_k   = βh · i_k / (n_k - 1)**ψH        ← ψH 控制頻率↔密度依賴

    回傳 t, new_inf_by_age(每步各年齡新感染), ever_infected(個體是否曾感染)。"""
    if rng is None:
        rng = np.random.default_rng()
    N = len(age); K = len(n_k)
    n_steps = int(round(t_max / tau))

    st = np.zeros(N, dtype=np.int8)            # 0=S, 1=I, 2=R
    st[I0_idx] = 1
    ever_inf = np.zeros(N, dtype=bool)
    ever_inf[I0_idx] = True

    # 家戶內接觸分母 (n_k-1)**ψH；n=1 家戶無家戶內接觸
    nk_minus1 = np.maximum(n_k - 1, 1)
    hh_denom = np.power(nk_minus1.astype(float), psi_H)
    is_size1 = (n_k == 1)

    t = np.arange(n_steps + 1) * tau
    new_inf_by_age = np.zeros((n_steps + 1, A), dtype=np.int64)

    step = 0
    for step in range(1, n_steps + 1):
        S_mask = (st == 0); I_mask = (st == 1)
        S_idx = np.where(S_mask)[0]
        I_idx_start = np.where(I_mask)[0]
        if len(I_idx_start) == 0:
            break

        I_by_age = np.bincount(age[I_idx_start], minlength=A)
        i_k = np.bincount(hh[I_idx_start], minlength=K)

        # --- 管道1：社區感染（按年齡分組）---
        comm_haz = beta_c * q * (M @ (I_by_age / N_by_age))
        S_age = age[S_idx]
        S_a = np.bincount(S_age, minlength=A)
        mean_comm = comm_haz * S_a * tau
        n_comm = np.minimum(rng.poisson(mean_comm), S_a)
        new_comm_idx = _choose_per_group(S_idx, S_age, n_comm, rng)
        st[new_comm_idx] = 1
        ever_inf[new_comm_idx] = True

        # --- 管道2：家戶內感染（在剩下的易感者上，按家戶分組）---
        S_idx2 = np.where(st == 0)[0]
        hh_haz_k = np.where(is_size1, 0.0, beta_h * i_k / hh_denom)
        s_k_now = np.bincount(hh[S_idx2], minlength=K)
        mean_hh = hh_haz_k * s_k_now * tau
        n_hh = np.minimum(rng.poisson(mean_hh), s_k_now)
        new_hh_idx = _choose_per_group(S_idx2, hh[S_idx2], n_hh, rng)
        st[new_hh_idx] = 1
        ever_inf[new_hh_idx] = True

        # 記錄本步新感染（兩管道合計，按年齡）
        all_new = np.concatenate([new_comm_idx, new_hh_idx])
        if len(all_new):
            new_inf_by_age[step] = np.bincount(age[all_new], minlength=A)

        # --- 康復：只從本步開始時已感染者抽 ---
        mean_rec = alpha * len(I_idx_start) * tau
        n_rec = min(rng.poisson(mean_rec), len(I_idx_start))
        if n_rec > 0:
            rec_idx = rng.choice(I_idx_start, size=n_rec, replace=False)
            st[rec_idx] = 2

    return t[:step + 1], new_inf_by_age[:step + 1], ever_inf


# ============================================================================
# 6. 觀測模型：潛在感染 → 通報病例（通報率 ρ、延遲、過度離散）
# ============================================================================

def _to_daily(new_inf_by_age, tau):
    """每步（tau）新感染 → 每日新感染。回傳 (n_days, A)。"""
    spd = int(round(1.0 / tau))
    rec = new_inf_by_age[1:]
    n_full = (len(rec) // spd) * spd
    return rec[:n_full].reshape(-1, spd, rec.shape[1]).sum(axis=1)


def gamma_delay_pmf(delay_mean, length=15, shape=2.0):
    """把通報延遲參數化成離散 Gamma PMF（平均 = delay_mean）。
    延遲是 proposal 要推斷的觀測 nuisance 之一，故其平均由先驗抽。"""
    scale = max(delay_mean, 1e-6) / shape
    xs = np.arange(length)
    # 用 Gamma CDF 差分得到離散 PMF（避免額外依賴 scipy，用簡單數值近似）
    from math import gamma as _gammafn
    # 未正規化的 Gamma pdf 值，再正規化為 PMF
    pdf = np.where(xs > 0,
                   (xs ** (shape - 1)) * np.exp(-xs / scale), 0.0)
    if pdf.sum() == 0:
        pdf[0] = 1.0
    return pdf / pdf.sum()


def _nb_sample(mean, size, rng):
    """負二項抽樣：平均=mean，方差=mean+mean^2/size。size 越小越離散。"""
    mean = np.asarray(mean, dtype=float)
    out = np.zeros(mean.shape, dtype=np.int64)
    pos = mean > 0
    p = size / (size + mean[pos])
    out[pos] = rng.negative_binomial(size, p)
    return out


def observation_model(new_inf_by_age, tau, rho, delay_pmf, nb_size, rng):
    """潛在每步感染 → 每日通報病例（按年齡）。
    回傳 true_daily, reported_daily，皆 (n_days, A)。"""
    true_daily = _to_daily(new_inf_by_age, tau)
    n_days, A_ = true_daily.shape

    delayed = np.zeros_like(true_daily, dtype=float)
    for a in range(A_):
        delayed[:, a] = np.convolve(true_daily[:, a], delay_pmf)[:n_days]

    expected_reported = rho * delayed
    reported_daily = np.zeros_like(true_daily, dtype=np.int64)
    for a in range(A_):
        reported_daily[:, a] = _nb_sample(expected_reported[:, a], nb_size, rng)
    return true_daily, reported_daily


# ============================================================================
# 7. 先驗 π(ϑ)：抽完整 ϑ = (q, α, βc, βh, ψM, ψH) + 觀測 nuisance (ρ, delay, nb)
# ============================================================================

def sample_prior(rng, M0, N_by_age):
    """從文獻錨定的先驗抽一組 θ。回傳 dict，含：
      推斷目標   : beta_c, beta_h, alpha, q(A,), psi_M(2,), psi_H
      觀測nuisance: rho, delay_mean, nb_size
      衍生/診斷  : R0, SAR   （與 βc/βh 一一對應，非回歸目標）
    """
    # --- 社區層 ---
    R0 = rng.uniform(1.2, 2.3)                              # Boelle [37]
    q = np.ones(A)
    q[1:] = np.exp(rng.normal(np.log(0.38), 0.25, A - 1))  # 中心 0.38 (Zhao [8])
    infectious_period = rng.lognormal(np.log(3.0), 0.18)   # ~3 天 (Boelle [37])
    alpha = 1.0 / infectious_period
    psi_M = np.array([rng.normal(0.0, 0.30),               # assort（同齡聚集）
                      rng.normal(0.0, 0.30)])              # inter （跨齡混合）
    M = build_M_shape(psi_M, M0)
    beta_c = beta_c_for_target_R0(R0, q, alpha, M, N_by_age)

    # --- 家戶層 ---
    SAR = rng.uniform(0.02, 0.19)                          # MoSAIC [33] ~ Navarra [32]
    beta_h = beta_h_from_SAR(SAR, alpha)
    psi_H = rng.uniform(0.0, 1.5)                          # 0=密度依賴, 1=頻率依賴, >1 更陡

    # --- 觀測 nuisance（proposal §3.2：三個都要推斷後邊緣化）---
    rho = np.exp(rng.uniform(np.log(0.005), np.log(0.05)))   # ~1% 量級 (Reed [39])
    delay_mean = rng.uniform(2.0, 6.0)                       # 幾天的通報延遲（弱信息）
    nb_size = np.exp(rng.uniform(np.log(2.0), np.log(40.0))) # 過度離散（Poisson-like ~ 明顯過離散）

    return dict(
        # 推斷目標 ϑ
        beta_c=beta_c, beta_h=beta_h, alpha=alpha, q=q, psi_M=psi_M, psi_H=psi_H,
        # 觀測 nuisance
        rho=rho, delay_mean=delay_mean, nb_size=nb_size,
        # 衍生/診斷量（非回歸目標）
        R0=R0, SAR=SAR,
    )


# ============================================================================
# 8. 乾淨接口 simulate_theta：θ → x = {incidence, hh_finalsize}
# ============================================================================

def hh_finalsize_vector(ever_inf, hh, n_k):
    """家戶大小 1..6 的最終規模分布，串接成固定長度向量（長度 27）。"""
    inf_per_hh = np.bincount(hh[ever_inf], minlength=len(n_k))
    parts = []
    for size in HH_SIZES_FIXED:
        mask = (n_k == size)
        if mask.sum() == 0:
            counts = np.zeros(size + 1)
        else:
            counts = np.bincount(inf_per_hh[mask], minlength=size + 1)[:size + 1].astype(float)
            counts = counts / counts.sum()
        parts.append(counts)
    return np.concatenate(parts)


def hh_index(size, k_infected, hh_sizes=HH_SIZES_FIXED):
    """在 hh_finalsize 向量裡定位「大小 size、恰好 k_infected 人感染」的索引。"""
    offset = sum(s + 1 for s in hh_sizes if s < size)
    return offset + k_infected


def simulate_theta(theta, pop, M0, rng, tau=0.25, t_max=180.0):
    """θ → x。建模型 → 雙層模擬 → 觀測模型 → 打包 x。"""
    age, hh, n_k, N_by_age = pop
    M = build_M_shape(theta['psi_M'], M0)
    I0_idx = rng.choice(len(age), size=40, replace=False)

    t, new_inf_by_age, ever_inf = simulate(
        theta['beta_c'], theta['beta_h'], theta['psi_H'], theta['q'],
        theta['alpha'], M, age, hh, n_k, N_by_age, I0_idx,
        tau=tau, t_max=t_max, rng=rng,
    )

    delay_pmf = gamma_delay_pmf(theta['delay_mean'])
    _, reported_daily = observation_model(
        new_inf_by_age, tau, theta['rho'], delay_pmf, theta['nb_size'], rng
    )

    incidence = np.zeros((T_FIXED, A))
    nd = min(len(reported_daily), T_FIXED)
    incidence[:nd] = reported_daily[:nd]
    hh_vec = hh_finalsize_vector(ever_inf, hh, n_k)
    return {'incidence': incidence, 'hh_finalsize': hh_vec}


# ============================================================================
# 自我測試：跑一組 θ，印形狀與關鍵值
# ============================================================================
if __name__ == "__main__":
    import time
    rng = np.random.default_rng(seed=11)

    age, hh, n_k = build_population(20000, rng)
    N_by_age = np.bincount(age, minlength=A)
    pop = (age, hh, n_k, N_by_age)
    print(f"固定人口 N={len(age)}, 家戶數 K={len(n_k)}, 平均家戶={len(age)/len(n_k):.2f}")

    th = sample_prior(rng, M_WALLINGA, N_by_age)
    print("\n抽到的 θ（推斷目標 + nuisance）：")
    print(f"  ϑ: R0={th['R0']:.2f}  SAR={th['SAR']:.1%}  alpha={th['alpha']:.3f}")
    print(f"     beta_c={th['beta_c']:.5f}  beta_h={th['beta_h']:.4f}")
    print(f"     q={np.round(th['q'],2)}")
    print(f"     psi_M={np.round(th['psi_M'],3)} (assort, inter)   psi_H={th['psi_H']:.3f}")
    print(f"  nuisance: rho={th['rho']:.4f}  delay_mean={th['delay_mean']:.2f}d  nb_size={th['nb_size']:.1f}")

    t0 = time.time()
    x = simulate_theta(th, pop, M_WALLINGA, rng)
    print(f"\n單次 simulate(θ)->x 耗時 ≈ {time.time()-t0:.2f}s")
    print(f"  x['incidence'].shape    = {x['incidence'].shape}  (T_FIXED={T_FIXED}, A={A})")
    print(f"  x['hh_finalsize'].shape = {x['hh_finalsize'].shape}  (HH_VEC_LEN={HH_VEC_LEN})")
    print(f"  通報總病例 = {int(x['incidence'].sum())}")
