"""
金标准对照：VAE 的摊销后验 vs 网格精确后验
================================================================================
最简 SIR 只有 3 个未知量 (β, α, ρ)，可以在 z 空间上铺网格把**精确后验**算出来
（似然与造数据用的生成模型完全一致，所以这是真后验，不是近似）。有了它才能
区分覆盖率不足的两个原因：

    (a) q 本身过窄        -> 网格后验比 VAE 后验宽    -> 改目标(IWAE)/加正则
    (b) encoder 有摊销偏差 -> 网格后验的均值准、VAE 的均值偏 -> 加 encoder 容量/数据

关键效率点：网格上的 λ(β,α,ρ) 曲线与观测数据无关，**算一次全体测试样本共用**。
64000 个网格点 × 120 天一次性展开，之后每条曲线只是一次 NB log-pmf 求和。

用法：
  python check_posterior_vs_grid.py --ckpt results_hybrid/sir_final2_model.pt
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

import sir_hybrid_vae as H


def build_grid(n_beta, n_alpha, n_rho, span, device):
    """在标准化 z 空间铺网格，返回 (z_grid, log_prior, λ 曲线)。"""
    g = [torch.linspace(-span, span, n, device=device)
         for n in (n_beta, n_alpha, n_rho)]
    zz = torch.stack(torch.meshgrid(*g, indexing="ij"), -1).reshape(-1, 3)
    log_prior = (-0.5 * zz ** 2).sum(1)                       # N(0,1)，常数项省略
    with torch.no_grad():
        beta, alpha, rho = H.phys_from_z(zz)
        lam = H.sir_incidence(beta, alpha) * rho.unsqueeze(1)  # (G, T)
    return zz, log_prior, lam.clamp(min=1e-6), [x.cpu().numpy() for x in g]


def grid_posterior(y, lam, log_prior, k):
    """给一条观测 y (T,)，返回网格上的归一化后验权重 (G,)。"""
    ll = H.nb_log_prob(y.unsqueeze(0).expand(lam.shape[0], -1), lam, k).sum(1)
    logw = ll + log_prior
    return torch.softmax(logw, 0)


def marginal_stats(w, shape, axis, values):
    """把权重按某一坐标轴边缘化，返回该轴上的 (均值, 标准差, 5%, 95%) —— z 尺度。"""
    wm = w.reshape(shape).sum(dim=[d for d in range(3) if d != axis]).cpu().numpy()
    wm = wm / wm.sum()
    mean = (wm * values).sum()
    sd = np.sqrt((wm * (values - mean) ** 2).sum())
    c = np.cumsum(wm)
    lo, hi = np.interp([0.05, 0.95], c, values)
    return mean, sd, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str,
                    default="results_hybrid/sir_final2_model.pt")
    ap.add_argument("--n_eval", type=int, default=200)
    ap.add_argument("--grid", type=int, default=40)
    ap.add_argument("--span", type=float, default=3.5)
    ap.add_argument("--out", type=str,
                    default="results_hybrid/posterior_vs_grid_final")
    args = ap.parse_args()

    device = "cpu"
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    targs = ck["args"]
    H.N_POP = targs["n_pop"]
    H.I0 = max(2.0, round(H.N_POP * 2e-4))
    print(f"[ckpt] {args.ckpt}  n_pop={H.N_POP} iwae={targs['iwae']} "
          f"nn_scale={targs['nn_scale']}")

    # ---- 用与训练时同一套种子/切分重建测试集 ----
    n_va = targs["n_train"] // 8
    n_all = targs["n_train"] + targs["n_test"] + n_va
    y, _, phys_true, _ = H.make_data(n_all, seed=targs["seed"],
                                     misspec=targs["misspec"])
    n_eval = min(args.n_eval, targs["n_test"])
    y_te = y[:n_eval].to(device)
    ph_te = phys_true[:n_eval].numpy()

    model = H.HybridVAE(targs["z_res_dim"], targs["nn_scale"],
                        targs["residual_mode"],
                        full_cov=bool(targs.get("full_cov", 1))).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    k = model.log_k.exp().detach()

    m, s = ck["x_scaler"]
    x_te = (torch.log1p(y_te) - m) / s

    # ---- 网格 ----
    shape = (args.grid, args.grid, args.grid)
    print(f"[grid] {args.grid}^3 = {args.grid ** 3} 点，展开 SIR ...")
    zz, log_prior, lam, axes_z = build_grid(*shape, args.span, device)

    # ---- VAE 后验（z 空间高斯；满协方差时取边缘 sd = sqrt(diag(LLᵀ)) ）----
    with torch.no_grad():
        mu_p, L_p, _, _ = model.enc(x_te)
        sd_p = torch.sqrt(torch.diagonal(L_p @ L_p.transpose(1, 2), dim1=1, dim2=2))
    mu_p, sd_p = mu_p.numpy(), sd_p.numpy()

    # ---- 逐条比较 ----
    stat = {n: {kk: [] for kk in ("g_mean", "g_sd", "g_cov", "v_mean", "v_sd", "v_cov")}
            for n in H.PHYS_NAMES}
    zt = np.stack([(np.log(ph_te[:, j]) - mu) / sg for j, (mu, sg) in enumerate(
        [(H.LOG_BETA_M, H.LOG_BETA_S), (H.LOG_ALPHA_M, H.LOG_ALPHA_S),
         (H.LOG_RHO_M, H.LOG_RHO_S)])], 1)                      # 真值的 z 坐标

    examples = []
    for i in range(n_eval):
        w = grid_posterior(y_te[i], lam, log_prior, k)
        for j, name in enumerate(H.PHYS_NAMES):
            gm, gs, glo, ghi = marginal_stats(w, shape, j, axes_z[j])
            stat[name]["g_mean"].append(gm)
            stat[name]["g_sd"].append(gs)
            stat[name]["g_cov"].append(glo <= zt[i, j] <= ghi)
            vm, vs = mu_p[i, j], sd_p[i, j]
            stat[name]["v_mean"].append(vm)
            stat[name]["v_sd"].append(vs)
            stat[name]["v_cov"].append(abs(zt[i, j] - vm) <= 1.645 * vs)
        if i < 3:
            examples.append((i, w.reshape(shape)))

    print("\n  === z 尺度上的比较（网格 = 真后验，VAE = 摊销后验）===")
    print("  参数     网格sd   VAE sd   网格cov  VAE cov   |VAE均值-网格均值|  真值-网格均值")
    for name in H.PHYS_NAMES:
        d = {kk: np.array(v) for kk, v in stat[name].items()}
        print(f"  {name:7s} {d['g_sd'].mean():7.4f}  {d['v_sd'].mean():7.4f}  "
              f"{d['g_cov'].mean():7.1%}  {d['v_cov'].mean():7.1%}  "
              f"{np.abs(d['v_mean'] - d['g_mean']).mean():14.4f}  "
              f"{np.sqrt(((zt[:len(d['g_mean']), H.PHYS_NAMES.index(name)] - d['g_mean']) ** 2).mean()):13.4f}")
    print("\n  读法：")
    print("   · 网格 cov 应 ≈90%（这是真后验，做校验用）")
    print("   · VAE sd 明显 < 网格 sd  => q 过窄")
    print("   · |VAE均值-网格均值| 与网格 sd 同量级或更大 => 摊销偏差是主因")

    # ---- 图：三条测试曲线上，网格边缘后验 vs VAE 高斯 ----
    fig, axes = plt.subplots(len(examples), 3, figsize=(13, 3.2 * len(examples)))
    axes = np.atleast_2d(axes)
    for r, (i, W) in enumerate(examples):
        for j, name in enumerate(H.PHYS_NAMES):
            ax = axes[r, j]
            wm = W.sum(dim=tuple(d for d in range(3) if d != j)).numpy()
            wm = wm / wm.sum() / (axes_z[j][1] - axes_z[j][0])
            ax.plot(axes_z[j], wm, color="#2c7fb8", lw=2, label="grid (exact)")
            xs = axes_z[j]
            ax.plot(xs, np.exp(-0.5 * ((xs - mu_p[i, j]) / sd_p[i, j]) ** 2) /
                    (sd_p[i, j] * np.sqrt(2 * np.pi)), color="#d95f0e", lw=2,
                    ls="--", label="VAE q")
            ax.axvline(zt[i, j], color="k", ls=":", lw=1.2, label="truth")
            ax.set_xlabel(f"z_{name}")
            if j == 0:
                ax.set_ylabel(f"test #{i}\ndensity")
            if r == 0 and j == 0:
                ax.legend(fontsize=7)
    plt.rcParams["axes.unicode_minus"] = False
    plt.tight_layout()
    plt.savefig(f"{args.out}.png", dpi=120)
    print(f"\n[done] {args.out}.png")


if __name__ == "__main__":
    main()
