"""
看见数据里的 pattern：hybrid VAE 的可解释性诊断
================================================================================
老师要求「要能看到 data 的 pattern」。hybrid 架构正好把 pattern 拆成两半，
这个脚本把两半都画出来：

  图 1  patterns_atlas.png      潜空间地图：真实数据按推断出的 (z_β, z_α) 排布
        ------------------------------------------------------------------
        每个格子放一条**真实观测曲线**，位置由 encoder 推断的潜坐标决定。
        看这张图就知道「潜空间的某个位置 = 数据长什么样」——潜变量不是黑箱
        编码，而是直接对应流行曲线的形状。

  图 2  patterns_traversal.png  潜变量遍历：每个可解释维度控制哪种 pattern
        ------------------------------------------------------------------
        固定其余维度，单独扫 z_β / z_α / z_ρ，画 decoder 生成的曲线。
        β 管上升速度与峰高，α 管峰宽与时点，ρ 管整体幅度——机理项让这三个
        方向天然可解释（纯黑箱 VAE 的潜维度没有这个性质）。

  图 3  patterns_blackbox.png   黑箱那一半到底抓到了什么 pattern
        ------------------------------------------------------------------
        (a) 扫 z_res，看 NN 能表达的修正曲线族；
        (b) **误设数据上的关键检验**：真数据里 β 在第 55 天前后被「干预」压低，
            NN 学到的修正曲线能不能把这个 pattern 还原出来（真值 vs 还原值）；
        (c) 后验预测检查：从后验抽样重新生成数据，band 能不能盖住实际观测。

用法：
  python inspect_patterns.py --ckpt results_hybrid/sir_final2_model.pt \
      --mis_ckpt results_hybrid/mis_fb_model.pt
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

import sir_hybrid_vae as H


def load(ckpt_path, device="cpu"):
    """载入 checkpoint，并按训练时的设定重建测试数据。"""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ck["args"]
    H.N_POP = a["n_pop"]
    H.I0 = max(2.0, round(H.N_POP * 2e-4))
    n_va = a["n_train"] // 8
    y, _, phys, beta_mod = H.make_data(a["n_train"] + a["n_test"] + n_va,
                                       seed=a["seed"], misspec=a["misspec"])
    y_te = y[:a["n_test"]]
    model = H.HybridVAE(a["z_res_dim"], a["nn_scale"], a["residual_mode"],
                        full_cov=bool(a.get("full_cov", 1))).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    m, s = ck["x_scaler"]
    x_te = (torch.log1p(y_te) - m) / s
    bm_te = beta_mod[:a["n_test"]] if beta_mod is not None else None
    return model, a, y_te, x_te, phys[:a["n_test"]].numpy(), bm_te


# ============================================================================
# 图 1：潜空间地图——真实数据按推断潜坐标排布
# ============================================================================
def fig_atlas(model, y_te, x_te, phys_te, out, n=5, span=1.6):
    with torch.no_grad():
        mu_p, _, mu_r, _ = model.enc(x_te)
        lam = model.decode(mu_p, mu_r)
    mu = mu_p.numpy()
    centers = np.linspace(-span, span, n)

    fig, axes = plt.subplots(n, n, figsize=(2.2 * n, 1.9 * n), sharex=True)
    for i, za in enumerate(centers[::-1]):                 # 行：z_alpha（上大下小）
        for j, zb in enumerate(centers):                   # 列：z_beta
            ax = axes[i, j]
            d = np.hypot(mu[:, 0] - zb, mu[:, 1] - za)
            k = int(d.argmin())
            if d[k] > 0.55:                                # 这个区域没有数据
                ax.set_facecolor("#f4f4f4")
                ax.set_xticks([]); ax.set_yticks([])
                continue
            ax.plot(y_te[k].numpy(), color="#999", lw=0.8)
            ax.plot(lam[k].numpy(), color="#d95f0e", lw=1.3)
            ax.set_title(f"β={phys_te[k,0]:.2f} α={phys_te[k,1]:.2f}", fontsize=7)
            ax.tick_params(labelsize=6)
            if i < n - 1:
                ax.set_xticks([])
    for j, zb in enumerate(centers):
        axes[-1, j].set_xlabel(f"$z_\\beta$={zb:+.1f}", fontsize=8)
    for i, za in enumerate(centers[::-1]):
        axes[i, 0].set_ylabel(f"$z_\\alpha$={za:+.1f}", fontsize=8)
    fig.suptitle("潜空间地图：每格是一条真实观测曲线(灰) + decoder 拟合(橙)，"
                 "位置 = encoder 推断的潜坐标", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[fig] {out}")


# ============================================================================
# 图 2：潜变量遍历——每个可解释维度控制哪种 pattern
# ============================================================================
def fig_traversal(model, out, n_step=7, span=2.0):
    vals = np.linspace(-span, span, n_step)
    names = [r"$z_\beta$ (传播率)", r"$z_\alpha$ (恢复率)", r"$z_\rho$ (上报率)"]
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
    for j, ax in enumerate(axes):
        z = torch.zeros(n_step, 3)
        z[:, j] = torch.tensor(vals, dtype=torch.float32)
        with torch.no_grad():
            lam = model.decode(z, torch.zeros(n_step, model.z_res_dim))
        for i in range(n_step):
            ax.plot(lam[i].numpy(), color=cmap(i / (n_step - 1)), lw=1.8,
                    label=f"{vals[i]:+.1f}" if i % 2 == 0 else None)
        ax.set_title(f"只动 {names[j]}，其余固定在先验均值", fontsize=10)
        ax.set_xlabel("day")
        ax.legend(fontsize=7, title="z 值", title_fontsize=7)
    axes[0].set_ylabel("每日报告病例")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[fig] {out}")


# ============================================================================
# 图 3：黑箱那一半抓到了什么
# ============================================================================
def fig_blackbox(model, y_te, x_te, out, mis=None, n_show=25):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))

    # (a) 黑箱在两种数据上各自做了多少事：数据合模型时应当「保持沉默」
    ax = axes[0]
    dR = model.z_res_dim

    def inferred_residual(mdl, xs, n):
        if mdl.z_res_dim == 0:
            return None
        with torch.no_grad():
            _, _, mu_r, _ = mdl.enc(xs[:n])
            return mdl.res(mu_r).numpy() * mdl.nn_scale

    f_ok = inferred_residual(model, x_te, n_show)
    if f_ok is not None:
        for i in range(len(f_ok)):
            ax.plot(f_ok[i], color="#2c7fb8", lw=0.8, alpha=0.7,
                    label="数据符合 SIR" if i == 0 else None)
    if mis is not None:
        f_ms = inferred_residual(mis[0], mis[2], n_show)
        for i in range(len(f_ms)):
            ax.plot(f_ms[i], color="#d95f0e", lw=0.8, alpha=0.7,
                    label="数据被误设(β 有干预)" if i == 0 else None)
        rms_ok = np.abs(f_ok).max() if f_ok is not None else 0
        ax.set_title(f"(a) 黑箱在两种数据上实际做了多少修正\n"
                     f"合模型时最大 {rms_ok:.3f}，误设时最大 "
                     f"{np.abs(f_ms).max():.3f}（相差 "
                     f"{np.abs(f_ms).max() / max(rms_ok, 1e-9):.0f} 倍）", fontsize=10)
    else:
        ax.set_title("(a) 黑箱实际做的修正（合模型数据）", fontsize=10)
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.legend(fontsize=7)
    ax.set_xlabel("day")
    ax.set_ylabel("log 修正量")

    # (b) 误设数据：NN 学到的 pattern vs 真实 pattern
    ax = axes[1]
    if mis is not None:
        mmodel, my, mx, mbm = mis
        with torch.no_grad():
            _, _, mu_r, _ = mmodel.enc(mx[:n_show])
            f = mmodel.res(mu_r).numpy() * mmodel.nn_scale       # 学到的 log β 修正
        truth = np.log(mbm[:n_show].numpy())                     # 真实 log β 修正
        f = f - f[:, :1]                                         # 常数偏移会被 β 吸收，
        truth = truth - truth[:, :1]                             # 对齐起点后比较
        for i in range(n_show):
            ax.plot(truth[i], color="#999", lw=0.7)
            ax.plot(f[i], color="#d95f0e", lw=0.7, alpha=0.7)
        ax.plot(truth.mean(0), color="k", lw=2.5, label="真实 log β(t) 变化(均值)")
        ax.plot(f.mean(0), color="#d95f0e", lw=2.5, ls="--",
                label="NN 学到的修正(均值)")
        r = np.corrcoef(truth.ravel(), f.ravel())[0, 1]
        ax.set_title(f"(b) 误设数据上：黑箱还原出了干预 pattern\n"
                     f"逐点相关 r={r:.3f}", fontsize=10)
        ax.legend(fontsize=7)
    else:
        ax.set_title("(b) 需要 --mis_ckpt", fontsize=10)
    ax.set_xlabel("day")
    ax.set_ylabel("log β 相对变化")

    # (c) 后验预测检查
    ax = axes[2]
    i = 0
    with torch.no_grad():
        mu_p, L_p, mu_r, lv_r = model.enc(x_te[i:i + 1])
        q = model.q_phys(mu_p.expand(300, -1), L_p.expand(300, -1, -1))
        z_p = q.sample()
        z_r = (mu_r + torch.exp(0.5 * lv_r) * torch.randn(300, dR)) if dR > 0 \
            else torch.zeros(300, 0)
        lam = model.decode(z_p, z_r)
        k = model.log_k.exp()
        rep = torch.distributions.NegativeBinomial(
            total_count=k, logits=torch.log(lam) - torch.log(k)).sample()
    lo, hi = np.percentile(rep.numpy(), [5, 95], axis=0)
    ax.fill_between(range(len(lo)), lo, hi, color="#d95f0e", alpha=0.3,
                    label="后验预测 90% band")
    ax.plot(lam.mean(0).numpy(), color="#d95f0e", lw=1.5, label="后验预测均值")
    ax.plot(y_te[i].numpy(), color="#333", lw=1.0, label="实际观测")
    cov = float(((lo <= y_te[i].numpy()) & (y_te[i].numpy() <= hi)).mean())
    ax.set_title(f"(c) 后验预测检查 (test #0)\nband 盖住 {cov:.0%} 的观测点",
                 fontsize=10)
    ax.set_xlabel("day")
    ax.set_ylabel("每日报告病例")
    ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[fig] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str,
                    default="results_hybrid/sir_final2_model.pt")
    ap.add_argument("--mis_ckpt", type=str, default="",
                    help="误设情形下训练的模型；用来检验黑箱有没有还原出真 pattern")
    ap.add_argument("--prefix", type=str, default="results_hybrid/patterns")
    args = ap.parse_args()

    plt.rcParams["font.sans-serif"] = ["PingFang HK", "Heiti TC", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    model, targs, y_te, x_te, phys_te, _ = load(args.ckpt)
    print(f"[ckpt] {args.ckpt}  n_test={len(y_te)}  nn_scale={targs['nn_scale']}")

    mis = None
    if args.mis_ckpt:
        mm, ma, my, mx, _, mbm = load(args.mis_ckpt)
        print(f"[ckpt] {args.mis_ckpt}  misspec={ma['misspec']} "
              f"mode={ma['residual_mode']}")
        mis = (mm, my, mx, mbm)
        H.N_POP = targs["n_pop"]          # load() 会改全局，用完切回主模型的设定
        H.I0 = max(2.0, round(H.N_POP * 2e-4))

    fig_atlas(model, y_te, x_te, phys_te, f"{args.prefix}_atlas.png")
    fig_traversal(model, f"{args.prefix}_traversal.png")
    fig_blackbox(model, y_te, x_te, f"{args.prefix}_blackbox.png", mis=mis)


if __name__ == "__main__":
    main()
