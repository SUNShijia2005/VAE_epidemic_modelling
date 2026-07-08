"""
第5步 Simulation-Based Inference
simulate(θ) → x 干净接口 + 先验抽样 + 批量生成 (θ, x) 训练资料
================================================================
把第 3 步模拟器 + 第 4 步观测模型包成单一函数，并从文献锚定的先验抽 θ，
批量产生 (θ, x) 配对——VAE 训练集 / ABC 比对的原料。

对齐 proposal §1, §3.2, §3.3。

θ（推断目标 + 干扰参数）：
  beta_c  社区传播率（由先验 R0 反推）
  beta_h  家户内传播率（由先验 SAR 反推）
  alpha   恢复率（感染期 ~3 天）
  q       年龄易感性向量，参考组(0-5)固定 = 1
  psi_M   M 的形状参数（1 维同龄聚集强度；M 最大特征值固定为 1）
  rho     通报率（干扰参数）
  nb_size 过度离散（干扰参数）

x（资料，固定形状，可喂网路）：
  incidence    : (T_fixed, A) 各年龄每日通报病例
  hh_finalsize : 长度 27 向量，家户大小 1..6 的最终规模分布串接

依赖：sir_age_hh.py, observation_model.py 同资料夹。
"""

import numpy as np
import matplotlib.pyplot as plt

from sir_age_hh import (
    build_population, beta_c_for_target_R0, beta_h_from_SAR, simulate,
    M_WALLINGA, AGE_LABELS,
)
from observation_model import observation_model

A = len(AGE_LABELS)                                 # 年龄段数量
HH_SIZES_FIXED = [1, 2, 3, 4, 5, 6]                 # 固定家户大小集合（保证 x 长度固定）
DELAY_PMF = np.array([0.02, 0.10, 0.22, 0.26, 0.20, 0.12, 0.06, 0.02])
DELAY_PMF = DELAY_PMF / DELAY_PMF.sum()
T_FIXED = 180                                        # incidence 固定天数


def build_M_shape(psi_M, M0):
    """1 维形状参数：调同龄聚集强度（缩放对角线），再把最大特征值归一化为 1。
    psi_M=0 -> 参考矩阵形状；M 保持对称。"""
    M = M0.copy().astype(float)
    np.fill_diagonal(M, np.diag(M0) * np.exp(psi_M))
    ev = np.max(np.abs(np.linalg.eigvals(M)))
    return M / ev


def sample_prior(rng, M0, N_by_age):
    """从文献锚定的先验抽一组 θ。"""
    R0 = rng.uniform(1.2, 2.3)                        # Boelle [37]
    SAR = rng.uniform(0.02, 0.19)                     # MoSAIC [33] ~ Navarra [32]
    infectious_period = rng.lognormal(np.log(3.0), 0.18)   # ~3 天 (Boelle [37])
    alpha = 1.0 / infectious_period
    q = np.ones(A)
    q[1:] = np.exp(rng.normal(np.log(0.38), 0.25, A - 1))  # 中心 0.38 (Zhao [8])
    psi_M = rng.normal(0.0, 0.30)
    rho = np.exp(rng.uniform(np.log(0.005), np.log(0.05)))  # ~1% 量级 (Reed [39])
    nb_size = np.exp(rng.uniform(np.log(2.0), np.log(40.0)))

    M = build_M_shape(psi_M, M0)
    beta_c = beta_c_for_target_R0(R0, q, alpha, M, N_by_age)
    beta_h = beta_h_from_SAR(SAR, alpha)
    return dict(beta_c=beta_c, beta_h=beta_h, alpha=alpha, q=q, psi_M=psi_M,
                rho=rho, nb_size=nb_size, R0=R0, SAR=SAR)


def hh_finalsize_vector(ever_inf, hh, n_k):
    """从"字典"变成"一整条数字"，家户大小 1..6 的最终规模分布，串接成固定长度向量（长度 2+3+4+5+6+7=27）。"""
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
    """在 hh_finalsize 向量里，定位「大小为 size 的家户、恰好 k_infected 人感染」的索引。
    拼接结构：size s 占 (s+1) 格，起始索引 = 前面各段长度之和 sum_{s'<size}(s'+1)。
      size1: idx 0-1 | size2: idx 2-4 | size3: idx 5-8 | size4: 9-13 | size5: 14-19 | size6: 20-26
    例：hh_index(2, 2) -> size2 段起始 2 + 感染人数 2 = 4（size-2 家户两人都感染）。"""
    offset = sum(s + 1 for s in hh_sizes if s < size)
    return offset + k_infected


def simulate_theta(theta, pop, M0, rng, tau=0.25, t_max=180.0):
    """θ -> x。单一干净接口：建模型 -> 双层模拟 -> 观测模型 -> 打包资料 x。"""
    age, hh, n_k, N_by_age = pop
    M = build_M_shape(theta['psi_M'], M0)
    I0_idx = rng.choice(len(age), size=40, replace=False)

    t, new_inf_by_age, ever_inf = simulate(
        theta['beta_c'], theta['beta_h'], theta['q'], theta['alpha'], M,
        age, hh, n_k, N_by_age, I0_idx, tau=tau, t_max=t_max, rng=rng,
    )
    _, reported_daily = observation_model(
        new_inf_by_age, tau, theta['rho'], DELAY_PMF, theta['nb_size'], rng
    )

    incidence = np.zeros((T_FIXED, A))       
    nd = min(len(reported_daily), T_FIXED)
    incidence[:nd] = reported_daily[:nd]   # 180天里,每天新增了几例被通报的病例,按年龄分
    hh_vec = hh_finalsize_vector(ever_inf, hh, n_k)  # 疫情结束时,不同大小的家庭最终感染情况的分布
    return {'incidence': incidence, 'hh_finalsize': hh_vec} 

def generate_batch(n_samples, pop, M0, rng, verbose=True):
    """批量生成 (θ, x)。回传 thetas (传播参数组合 list of dict), X_inc(每日通报曲线 n,T,A), X_hh(家户最终感染分布 n,27)。"""
    thetas, X_inc, X_hh = [], [], []
    for i in range(n_samples):
        theta = sample_prior(rng, M0, pop[3])
        x = simulate_theta(theta, pop, M0, rng)
        thetas.append(theta)
        X_inc.append(x['incidence'])
        X_hh.append(x['hh_finalsize'])
        if verbose and (i + 1) % 10 == 0:
            print(f"  生成 {i+1}/{n_samples} ...")
    return thetas, np.array(X_inc), np.array(X_hh)


if __name__ == "__main__":
    import time
    rng = np.random.default_rng(seed=11)

    # 固定人口结构（家户大小分布是固定输入，proposal §3.2）；为批量速度用 2 万人
    age, hh, n_k = build_population(20000, rng)
    N_by_age = np.bincount(age, minlength=A)
    pop = (age, hh, n_k, N_by_age)
    print(f"固定人口 N={len(age)}, 家户数 K={len(n_k)}, 平均家户={len(age)/len(n_k):.2f}")

    # 单跑计时
    th = sample_prior(rng, M_WALLINGA, N_by_age)
    t0 = time.time()
    x = simulate_theta(th, pop, M_WALLINGA, rng)
    dt = time.time() - t0
    print(f"\n单次 simulate(θ)->x 耗时 ≈ {dt:.2f} 秒")
    print(f"  x['incidence'].shape = {x['incidence'].shape}")
    print(f"  x['hh_finalsize'].shape = {x['hh_finalsize'].shape}")
    print(f"  该 θ: R0={th['R0']:.2f}, SAR={th['SAR']:.1%}, alpha={th['alpha']:.3f}, "
          f"q={np.round(th['q'],2)}, psi_M={th['psi_M']:.2f}, rho={th['rho']:.3f}")

    # 小批量示范
    n_demo = 40
    print(f"\n批量生成 {n_demo} 组 (θ, x)（示范；正式训练需数千~数十万 + 平行化）...")
    t0 = time.time()
    thetas, X_inc, X_hh = generate_batch(n_demo, pop, M_WALLINGA, rng)
    print(f"完成，总耗时 {time.time()-t0:.1f} 秒")
    print(f"训练张量：X_inc {X_inc.shape}, X_hh {X_hh.shape}")

    np.savez("training_batch_demo.npz",
             X_inc=X_inc, X_hh=X_hh,
             R0=np.array([t['R0'] for t in thetas]),
             SAR=np.array([t['SAR'] for t in thetas]),
             beta_c=np.array([t['beta_c'] for t in thetas]),
             beta_h=np.array([t['beta_h'] for t in thetas]))
    print("训练资料已存成 training_batch_demo.npz")

    # ---- 视觉化：先验涵盖范围 ----
    R0s = np.array([t['R0'] for t in thetas])
    SARs = np.array([t['SAR'] for t in thetas])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    inc_tot = X_inc.sum(axis=2)                       # (n, T) 各样本每日总通报
    norm = plt.Normalize(R0s.min(), R0s.max())
    cmap = plt.cm.viridis
    for i in range(n_demo):
        ax.plot(inc_tot[i], color=cmap(norm(R0s[i])), alpha=0.6, lw=1)
    ax.set_xlabel("day"); ax.set_ylabel("total reported cases / day")
    ax.set_title(f"{n_demo} simulated reported curves\ncolour = R0 (prior 1.2-2.3)")
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="R0")

    ax = axes[1]
    # 家户 size=2 「两人都感染」的比例 vs SAR —— 应随 SAR 上升
    both2 = X_hh[:, hh_index(2, 2)]     # size-2 家户两人都感染（索引 4）
    ax.scatter(SARs * 100, both2 * 100, c=R0s, cmap=cmap, s=30)
    ax.set_xlabel("household SAR drawn from prior (%)")
    ax.set_ylabel("P(both infected | size-2 household) (%)")
    ax.set_title("Household clustering signal tracks SAR\n(this is what lets the net learn beta_h)")
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="R0")

    plt.rcParams["axes.unicode_minus"] = False
    plt.tight_layout()
    plt.savefig("training_batch_demo.png", dpi=130)
    print("图已存成 training_batch_demo.png")