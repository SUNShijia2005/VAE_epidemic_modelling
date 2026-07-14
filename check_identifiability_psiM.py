"""
第 1 步：ψM 的先验预测「可识别性」检查
================================================================================
问题：recovery 图里 psiM_inter r=0.22（几乎横云），psiM_assort r=0.50。
到底是「模型没学好」还是「数据里根本没有这个维度的信息」？

本脚本做根因诊断：把其他参数固定在一组 baseline，只扫 ψM 的一个分量，
每档跑多个随机种子，看输出数据随该分量变不变。
  · 参数引起的数据变化 >> 随机噪声  → 可识别，问题在模型/训练（去修过拟合）
  · 参数引起的数据变化 ~ 随机噪声    → 本质不可识别，调模型救不了
                                        （proposal 已 hedge；要救得给数据加通道）

关键设计（对齐训练分布逻辑，避免误判）：
  1. 扫 ψM 时 **保持 R0 固定**：M 改变后重算 beta_c，使社区 R0 不变。
     否则「总传播强度」变化会被误当成 inter 的信号。
  2. ψM_inter 是跨龄混合，主要改变 **年龄间传播剖面** 而非总量，
     所以看「各年龄组 attack-rate 剖面」，不是只看总曲线。
  3. ψM_assort 作对照：它 r=0.50 明显更可识别，用来校准「有信号」长什么样。

量化指标：对每个 summary 通道，
  separability = Var_between(跨档均值) / Mean(档内跨种子方差)
  >> 1 表示信号盖过噪声（可识别）；~1 或更小表示淹没在噪声里（不可识别）。

用法：
  python check_identifiability_psiM.py            # 默认 N=20000, 7 档, 每档 6 种子
  python check_identifiability_psiM.py --n_pop 30000 --levels 9 --reps 8
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "simulate_data"))
import epidemic_simulator as es


# ---------------------------------------------------------------------------
# 从一次模拟里同时取「机制信号」（潜在，无观测噪声）与「观测信号」（含 ρ/delay/nb）
# ---------------------------------------------------------------------------
def hh_age_coinfection(ever_inf, age, hh, A):
    """候选新通道：家户内感染者的『年龄共现矩阵』(6×6, 上三角+对角=21维)。
    同一户里每一对被感染个体 (a,b) 贡献 C[a,b]+=1。对角=同龄共感染，
    非对角=跨龄共感染——正是 ψM_inter(跨龄混合) 的直接信号，且潜在层无观测噪声。
    归一化为频率后取上三角展平。"""
    C = np.zeros((A, A))
    inf_idx = np.where(ever_inf)[0]
    if len(inf_idx) >= 2:
        h = hh[inf_idx]; a = age[inf_idx]
        order = np.argsort(h, kind="stable")
        h = h[order]; a = a[order]
        bounds = np.where(np.diff(h))[0] + 1
        for g in np.split(a, bounds):
            if len(g) < 2:
                continue
            for i in range(len(g)):
                for j in range(i + 1, len(g)):
                    C[g[i], g[j]] += 1; C[g[j], g[i]] += 1
    s = C.sum()
    if s > 0:
        C = C / s
    iu = np.triu_indices(A)
    return C[iu]                                        # 21 维


def simulate_summaries(theta, pop, M0, rng, tau=0.25, t_max=180.0):
    age, hh, n_k, N_by_age = pop
    M = es.build_M_shape(theta["psi_M"], M0)
    I0_idx = rng.choice(len(age), size=40, replace=False)

    t, new_inf_by_age, ever_inf = es.simulate(
        theta["beta_c"], theta["beta_h"], theta["psi_H"], theta["q"],
        theta["alpha"], M, age, hh, n_k, N_by_age, I0_idx,
        tau=tau, t_max=t_max, rng=rng,
    )

    # (1) 机制信号：各年龄组最终 attack rate（潜在 ever_infected / 该龄人口）
    age_attack = np.bincount(age[ever_inf], minlength=es.A).astype(float) / np.maximum(N_by_age, 1)

    # (2) 观测信号：走完整观测模型，得到 reported incidence 的年龄剖面
    delay_pmf = es.gamma_delay_pmf(theta["delay_mean"])
    _, reported_daily = es.observation_model(
        new_inf_by_age, tau, theta["rho"], delay_pmf, theta["nb_size"], rng
    )
    tot = reported_daily.sum()
    obs_age_profile = reported_daily.sum(axis=0).astype(float) / max(tot, 1)  # 各年龄占报告总数比例（6维）

    # (3) 世帯 final-size 分布（潜在，27维）——现有通道，只按家户大小分层，无年龄信息
    hh_vec = es.hh_finalsize_vector(ever_inf, hh, n_k)

    # (4) 候选A：家户内年龄共现矩阵（潜在，21维）
    hh_coinf = hh_age_coinfection(ever_inf, age, hh, es.A)

    # (5) 候选B：各年龄组潜在 incidence 曲线的『峰值时间 + 归一化形状』——
    #     ψM_inter 改变社区跨龄传播 -> 各龄疫情的相对时序(谁先谁后/同步性)。
    #     daily 潜在 incidence (n_days, A)。取每龄峰值时间(6) + 每龄曲线的时间重心(6)。
    daily_latent = es._to_daily(new_inf_by_age, tau).astype(float)   # (n_days, A)
    nd = daily_latent.shape[0]
    peak_t = daily_latent.argmax(axis=0).astype(float) / max(nd, 1)          # 6 维峰值时间
    tgrid = np.arange(nd)[:, None]
    col = daily_latent.sum(axis=0); col[col == 0] = 1.0
    centroid = (daily_latent * tgrid).sum(axis=0) / col / max(nd, 1)         # 6 维时间重心
    age_timing = np.concatenate([peak_t, centroid])                         # 12 维

    return dict(age_attack=age_attack, obs_age_profile=obs_age_profile,
                hh_final=hh_vec, hh_coinf=hh_coinf, age_timing=age_timing)


def separability(stacked):
    """stacked: (n_levels, n_reps, dim)。回传每维 between/within 方差比，以及整体标量。"""
    level_means = stacked.mean(axis=1)                 # (n_levels, dim)
    var_between = level_means.var(axis=0)              # (dim,) 跨档均值的方差
    var_within = stacked.var(axis=1).mean(axis=0)      # (dim,) 档内跨种子方差（对档平均）
    ratio = var_between / (var_within + 1e-12)
    # 整体：用总平方和口径聚合（对通道的所有维度）
    overall = var_between.sum() / (var_within.sum() + 1e-12)
    return ratio, overall


def scan_component(comp, baseline, pop, M0, levels, reps, spread, seed0):
    """扫 psi_M[comp] 在 baseline±spread 的 levels 档，每档 reps 个种子。
    保持 R0 固定：M 变了就重算 beta_c。回传各 summary 的 (n_levels,n_reps,dim)。"""
    age, hh, n_k, N_by_age = pop
    vals = np.linspace(baseline["psi_M"][comp] - spread,
                       baseline["psi_M"][comp] + spread, levels)
    R0_fixed = baseline["R0"]

    out = None
    for i, v in enumerate(vals):
        th = dict(baseline)
        pm = baseline["psi_M"].copy(); pm[comp] = v
        th["psi_M"] = pm
        # 复现训练分布：M 变 -> 重算 beta_c 使 R0 = R0_fixed
        M = es.build_M_shape(pm, M0)
        th["beta_c"] = es.beta_c_for_target_R0(R0_fixed, th["q"], th["alpha"], M, N_by_age)

        reps_summ = [simulate_summaries(th, pop, M0,
                                        np.random.default_rng(seed0 + 1000 * i + r))
                     for r in range(reps)]
        if out is None:
            out = {k: [] for k in reps_summ[0]}
        for k in out:
            out[k].append(np.array([s[k] for s in reps_summ]))
    return vals, {k: np.array(v) for k, v in out.items()}   # each (levels,reps,dim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_pop", type=int, default=20000)
    ap.add_argument("--levels", type=int, default=7)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--spread", type=float, default=0.60, help="扫描半宽（先验σ=0.30，默认±2σ）")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=str, default="identifiability_psiM.png")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    age, hh, n_k = es.build_population(args.n_pop, rng)
    N_by_age = np.bincount(age, minlength=es.A)
    pop = (age, hh, n_k, N_by_age)
    print(f"[pop] N={len(age)}  K={len(n_k)}  平均家戶={len(age)/len(n_k):.2f}")

    # baseline：从先验抽一组稳定的 θ，psi_M 置中(0,0)，R0 取中值
    base = es.sample_prior(rng, es.M_WALLINGA, N_by_age)
    base["psi_M"] = np.array([0.0, 0.0])
    base["R0"] = 1.7
    base["beta_c"] = es.beta_c_for_target_R0(
        base["R0"], base["q"], base["alpha"],
        es.build_M_shape(base["psi_M"], es.M_WALLINGA), N_by_age)
    print(f"[baseline] R0={base['R0']}  SAR≈{base['SAR']:.1%}  psi_H={base['psi_H']:.2f}  "
          f"q={np.round(base['q'],2)}")
    print(f"[scan] 每个分量 {args.levels} 档 × {args.reps} 种子，半宽 ±{args.spread}（先验σ=0.30）\n")

    comps = {0: "psiM_assort（同龄聚集/对照）", 1: "psiM_inter（跨龄混合/被质疑）"}
    results = {}
    for c, name in comps.items():
        print(f"[run] 扫 {name} ...")
        vals, summ = scan_component(c, base, pop, es.M_WALLINGA,
                                    args.levels, args.reps, args.spread, args.seed + c * 99991)
        results[c] = (vals, summ)

    # ---- 量化：separability 表 ----
    print("\n" + "=" * 72)
    print("可识别性指标 separability = Var_between(跨档均值) / Mean(档内跨种子方差)")
    print("  >>1 = 信号盖过噪声(可识别)   ~1 或更小 = 淹没在噪声里(不可识别)")
    print("=" * 72)
    chan_names = {"age_attack": "潜在年龄attack剖面",
                  "obs_age_profile": "观测年龄剖面(含噪)",
                  "hh_final": "世帯final-size(现有,27维)",
                  "hh_coinf": "★候选A家户年龄共现(21维)",
                  "age_timing": "★候选B各龄峰值时序(12维)"}
    print(f"\n  {'分量':28s} {'通道':28s} {'separability':>14s}")
    print("  " + "-" * 72)
    overall_tab = {}
    for c, name in comps.items():
        _, summ = results[c]
        for ch in ("age_attack", "obs_age_profile", "hh_final", "hh_coinf", "age_timing"):
            _, ov = separability(summ[ch])
            overall_tab[(c, ch)] = ov
            print(f"  {name:28s} {chan_names[ch]:28s} {ov:14.2f}")
        print()

    # ---- 画图：两列(assort/inter) × 三行(三个通道)，每档一条均值线 ----
    fig, axes = plt.subplots(4, 2, figsize=(13, 15))
    plt.rcParams["axes.unicode_minus"] = False
    row_chan = [("age_attack", "latent age attack-rate profile"),
                ("obs_age_profile", "observed age profile (reported share)"),
                ("hh_final", "household final-size (EXISTING channel)"),
                ("hh_coinf", "household age co-infection (NEW candidate)")]
    xlabels = {0: "age group", 1: "age group",
               2: "hh final-size bin (size1..6)", 3: "age-pair index (upper-tri)"}

    for col, (c, name) in enumerate(comps.items()):
        vals, summ = results[c]
        for row, (ch, title) in enumerate(row_chan):
            ax = axes[row][col]
            data = summ[ch]                              # (levels,reps,dim)
            level_mean = data.mean(axis=1)               # (levels,dim)
            dim = level_mean.shape[1]
            xs = np.arange(dim)
            cmap = plt.cm.viridis(np.linspace(0, 1, len(vals)))
            for i, v in enumerate(vals):
                ax.plot(xs, level_mean[i], color=cmap[i], lw=1.6,
                        label=f"{v:+.2f}" if (row == 0) else None)
            ov = overall_tab[(c, ch)]
            ax.set_title(f"{name.split('（')[0]} → {title}\nseparability={ov:.2f}", fontsize=9)
            ax.set_xlabel(xlabels[row]); ax.set_ylabel(ch)
            if row == 0:
                ax.legend(title="ψ value", fontsize=7, ncol=2)
    fig.suptitle("ψM prior-predictive identifiability:  does the data move when the parameter moves?\n"
                 "(left assort = the more-identifiable control r≈0.50;  right inter = the questioned dim r≈0.22)",
                 fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(args.out, dpi=120); plt.close()
    print(f"[done] 图已存：{args.out}")

    # ---- 结论提示 ----
    print("\n[读图指引]")
    print("  · 看右列(inter)三行的线是否『挤成一团』：若不同 ψ 值的均值线基本重合，")
    print("    且 separability≈1，则 inter 本质不可识别——调模型/修过拟合救不了。")
    print("  · 对照左列(assort)：若它的线明显分得开、separability 明显更大，")
    print("    说明『有信号』长这样，反衬 inter 确实缺信息。")
    print("  · 若 inter 的线其实也分得开(separability>>1)，则问题在模型端，")
    print("    回到第2步修过拟合(weight decay/dropout/降 z_dim)即可。")


if __name__ == "__main__":
    main()
