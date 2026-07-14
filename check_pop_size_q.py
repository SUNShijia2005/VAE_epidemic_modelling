"""
验证：把单次模拟的『人口规模』放大，能否提升高龄 q 的可识别性？
================================================================================
背景：BASE2 里 q[60+] corr=0.29、q[40-59]=0.41，弱。诊断认为根因是
「高龄组感染事件少 -> 泊松噪声相对大 -> 数据里高龄信号弱」。

若真如此，则加『训练样本数』没用（可识别性天花板），但加『单次模拟人口规模』
可能有用：高龄绝对病例数增多 -> 随机噪声相对变小 -> 高龄 attack 信号增强。

本脚本不重训 VAE，只用先验预测的 signal-to-noise 验证：
  对每个 q[k]，扫它在先验范围内取几档值，每档多种子，看
    separability = Var_between(跨档均值) / Mean(档内跨种子方差)
  在 N=20000 vs N=80000 两档人口下的变化。
  · 大人口下 separability 明显上升 -> 加人口有用，值得重新生成数据。
  · 基本不动               -> 高龄就是推不准，死心。

保持训练分布逻辑：扫 q[k] 时 R0 固定（重算 beta_c）。
用法：
  python check_pop_size_q.py                       # 默认 q[20-39],q[40-59],q[60+]，20k vs 80k
  python check_pop_size_q.py --pops 20000 120000 --levels 7 --reps 6
"""

import argparse
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "simulate_data"))
import epidemic_simulator as es


def summaries(theta, pop, M0, rng, tau=0.25, t_max=180.0):
    age, hh, n_k, N_by_age = pop
    M = es.build_M_shape(theta["psi_M"], M0)
    I0_idx = rng.choice(len(age), size=40, replace=False)
    t, new_inf_by_age, ever_inf = es.simulate(
        theta["beta_c"], theta["beta_h"], theta["psi_H"], theta["q"],
        theta["alpha"], M, age, hh, n_k, N_by_age, I0_idx, tau=tau, t_max=t_max, rng=rng)
    # 潜在各年龄 attack rate（6维）——q[k] 的主战场
    age_attack = np.bincount(age[ever_inf], minlength=es.A).astype(float) / np.maximum(N_by_age, 1)
    # 观测各年龄剖面（含噪，6维）
    delay_pmf = es.gamma_delay_pmf(theta["delay_mean"])
    _, rep = es.observation_model(new_inf_by_age, tau, theta["rho"], delay_pmf, theta["nb_size"], rng)
    tot = rep.sum()
    obs = rep.sum(axis=0).astype(float) / max(tot, 1)
    return age_attack, obs


def sep_scalar(stacked):
    """stacked (levels,reps,dim) -> 单个 overall separability。"""
    lvl_mean = stacked.mean(axis=1)
    vb = lvl_mean.var(axis=0).sum()
    vw = stacked.var(axis=1).mean(axis=0).sum()
    return vb / (vw + 1e-12)


def sep_dim(stacked, d):
    """只看第 d 维（改 q[k] 主要动第 k 组 attack）。"""
    lvl_mean = stacked.mean(axis=1)[:, d]
    vb = lvl_mean.var()
    vw = stacked.var(axis=1)[:, d].mean()
    return vb / (vw + 1e-12)


def scan_q(k, baseline, pop, M0, qlo, qhi, levels, reps, seed0):
    age, hh, n_k, N_by_age = pop
    vals = np.linspace(qlo, qhi, levels)
    R0 = baseline["R0"]
    attk, obsv = [], []
    for i, v in enumerate(vals):
        th = dict(baseline); q = baseline["q"].copy(); q[k] = v; th["q"] = q
        M = es.build_M_shape(th["psi_M"], M0)
        th["beta_c"] = es.beta_c_for_target_R0(R0, q, th["alpha"], M, N_by_age)
        pa, po = [], []
        for r in range(reps):
            rng = np.random.default_rng(seed0 + 1000 * i + r)
            a, o = summaries(th, pop, M0, rng)
            pa.append(a); po.append(o)
        attk.append(pa); obsv.append(po)
    return np.array(attk), np.array(obsv)      # (levels,reps,6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pops", type=int, nargs="+", default=[20000, 80000])
    ap.add_argument("--kidx", type=int, nargs="+", default=[3, 4, 5],
                    help="扫哪些 q 分量(0=0-5参考..5=60+)；默认 q[20-39],q[40-59],q[60+]")
    ap.add_argument("--levels", type=int, default=6)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--qlo", type=float, default=0.20)
    ap.add_argument("--qhi", type=float, default=0.70)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    labels = es.AGE_LABELS
    print(f"[setup] pops={args.pops}  扫 q{[labels[k] for k in args.kidx]}  "
          f"{args.levels}档×{args.reps}种子  q∈[{args.qlo},{args.qhi}]  R0固定\n")

    # 每个人口规模建一次人口 + 一组稳定 baseline
    results = {}   # (n_pop, k) -> (sep_attack_kdim, sep_obs_kdim, sep_attack_all)
    for n_pop in args.pops:
        rng = np.random.default_rng(args.seed)
        age, hh, n_k = es.build_population(n_pop, rng)
        N_by_age = np.bincount(age, minlength=es.A)
        pop = (age, hh, n_k, N_by_age)
        base = es.sample_prior(rng, es.M_WALLINGA, N_by_age)
        base["psi_M"] = np.array([0.0, 0.0]); base["R0"] = 1.7
        base["beta_c"] = es.beta_c_for_target_R0(
            base["R0"], base["q"], base["alpha"], es.build_M_shape(base["psi_M"], es.M_WALLINGA), N_by_age)
        print(f"[pop N={len(age)}] 各年龄人口数 = {dict(zip(labels, N_by_age))}")
        for k in args.kidx:
            attk, obsv = scan_q(k, base, pop, es.M_WALLINGA, args.qlo, args.qhi,
                                args.levels, args.reps, args.seed + k * 131)
            results[(n_pop, k)] = (sep_dim(attk, k), sep_dim(obsv, k), sep_scalar(attk))
        print()

    # 对比表
    print("=" * 74)
    print("separability（越大=信号越盖过噪声=越可识别）。看大人口下高龄 q 是否上升")
    print("=" * 74)
    p0, p1 = args.pops[0], args.pops[-1]
    print(f"\n  {'q 分量':12s} | {'潜在attack(该组维)':>22s} | {'观测剖面(该组维,含噪)':>24s}")
    print(f"  {'':12s} | N={p0:<7d} N={p1:<7d} 倍 | N={p0:<7d} N={p1:<7d} 倍")
    print("  " + "-" * 70)
    for k in args.kidx:
        a0, o0, _ = results[(p0, k)]
        a1, o1, _ = results[(p1, k)]
        print(f"  q[{labels[k]:8s}] | {a0:8.2f} {a1:8.2f} {a1/max(a0,1e-9):4.1f}x | "
              f"{o0:8.2f} {o1:8.2f} {o1/max(o0,1e-9):4.1f}x")
    print("\n[读表] 关注 q[40-59]/q[60+] 的『潜在attack』列：")
    print("  · 若 N 从 %d→%d 使 separability 明显放大(>2x)且绝对值进入可识别区(>~3)," % (p0, p1))
    print("    则加人口有效 -> 值得用大人口重新生成训练数据重训。")
    print("  · 若基本不动或仍<~2，则高龄 q 是硬限制，加人口也救不了。")
    print("  · 观测列通常远低于潜在列(报告噪声压制)，这是 VAE 实际面对的上限。")


if __name__ == "__main__":
    main()
