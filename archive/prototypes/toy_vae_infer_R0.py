"""
用 conditional VAE 从 toy SIR 曲线估计 R0
================================================================================
这是老师说的「toy SIR + VAE 估 R0」最小实验。目的不是做完整研究，而是
**验证方法本身能不能 work**：给一条疫情曲线，VAE 能不能把生成它的 R0 推回来。

对齐 proposal §3.5 与 Nautiyal 等 [26] 的 conditional VAE for SBI，但砍到最小：
  - 推断目标 θ = 只有 R0 一个数
  - 数据 x     = 一条总感染曲线 new_inf（toy 版不分年龄、不分家户）
  - cVAE 三件套：encoder q(z|θ,x) -> 潜变量 z -> decoder p(θ|x,z)
  - 损失 = 重构项 + KL 项（proposal §3.5 的 L(φ)，这里用最常见的 β-VAE 写法）
  - 推断：给一条 x_obs，从 p(z) 抽多个 z，decoder 吐出多个 R0 -> 近似 posterior

流程（对齐 proposal Fig.2 的「训练一次、推断一次前向」）：
  1. 从先验抽 R0            -> sample_prior
  2. toy SIR 模拟成曲线 x   -> 你的 sir_tauleap.tau_leap_sir
  3. 大量 (R0, x) 配对当训练集
  4. 训练 cVAE
  5. 在「已知真值」的测试曲线上，看 posterior 能不能罩住真值

依赖：sir_tauleap.py（放同资料夹）、numpy、torch、matplotlib。
纯 CPU 即可，网络很小，几分钟训完。

用法：
  python vae_infer_R0.py                 # 用默认设定跑完整流程
  python vae_infer_R0.py --n 4000 --epochs 60
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sir_tauleap import tau_leap_sir


# ============================================================================
# 固定设定（toy 版：这些当固定输入，不推断）
# ============================================================================
N_POP = 10_000          # 总人口
I0 = 10                 # 初始感染
TAU = 0.5               # tau-leap 步长
T_MAX = 120.0           # 模拟总天数
ALPHA_FIXED = 0.2       # 恢复率固定（平均感染期 5 天）；只推 R0 -> beta = R0 * alpha
T_FIXED = int(round(T_MAX / TAU)) + 1   # 曲线长度（固定，才能喂网络）

# R0 先验范围（对齐 proposal：流感样，Boëlle [37] 多数在 1.2–2.3，这里放宽一点点）
R0_MIN, R0_MAX = 1.2, 3.0


# ============================================================================
# 1) 先验 + 模拟：把一个 R0 变成一条曲线 x
# ============================================================================
def sample_R0(rng):
    """从先验（均匀）抽一个 R0。"""
    return float(rng.uniform(R0_MIN, R0_MAX))


def simulate_curve(R0, rng):
    """R0 -> 一条总感染曲线 x（长度 T_FIXED）。
    只推 R0，所以 alpha 固定、beta = R0 * alpha。"""
    beta = R0 * ALPHA_FIXED
    t, S, I, R, new_inf = tau_leap_sir(
        beta, ALPHA_FIXED, N_POP, I0, tau=TAU, t_max=T_MAX, rng=rng
    )
    # new_inf 长度应为 T_FIXED；保险起见对齐长度
    x = np.zeros(T_FIXED, dtype=np.float32)
    nd = min(len(new_inf), T_FIXED)
    x[:nd] = new_inf[:nd]
    return x


def make_dataset(n_samples, seed):
    """生成 n_samples 组 (R0, x)。回传 R0 array (n,1)、x array (n, T_FIXED)。"""
    rng = np.random.default_rng(seed)
    R0s = np.zeros((n_samples, 1), dtype=np.float32)
    X = np.zeros((n_samples, T_FIXED), dtype=np.float32)
    for i in range(n_samples):
        r0 = sample_R0(rng)
        R0s[i, 0] = r0
        X[i] = simulate_curve(r0, rng)
        if (i + 1) % max(1, n_samples // 10) == 0:
            print(f"  模拟 {i+1}/{n_samples} ...")
    return R0s, X


# ============================================================================
# 2) 数据标准化：曲线值可能上千，R0 在 1~3，两者都归一化后网络才好训
# ============================================================================
class Scaler:
    """记住训练集的均值/标准差，之后对所有数据做同样的标准化。"""
    def __init__(self, arr):
        self.mean = arr.mean(axis=0, keepdims=True)
        self.std = arr.std(axis=0, keepdims=True) + 1e-6

    def fwd(self, arr):     # 原始 -> 标准化
        return (arr - self.mean) / self.std

    def inv(self, arr):     # 标准化 -> 原始
        return arr * self.std + self.mean


# ============================================================================
# 3) conditional VAE
#    encoder: (θ, x) -> z 的 (mu, logvar)          == q_φ(z | θ, x)
#    decoder: (x, z) -> θ 的 (mu, logvar)          == p(θ | x, z)
#    x 先经一个小 embedding 压成低维（对齐 proposal 的 shared embedding）
# ============================================================================
class Embedding(nn.Module):
    """把长度 T_FIXED 的曲线压成 emb_dim 维（proposal 的 shared embedding，简化成 MLP）。"""
    def __init__(self, t_len, emb_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(t_len, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, emb_dim), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class CVAE(nn.Module):
    def __init__(self, theta_dim=1, emb_dim=32, z_dim=8, hidden=64):
        super().__init__()
        self.theta_dim = theta_dim
        self.z_dim = z_dim
        self.embed = Embedding(T_FIXED, emb_dim)

        # encoder q(z | θ, x)：吃 [θ, embed(x)] -> z 的 mu/logvar
        self.enc = nn.Sequential(
            nn.Linear(theta_dim + emb_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.enc_mu = nn.Linear(hidden, z_dim)
        self.enc_logvar = nn.Linear(hidden, z_dim)

        # decoder p(θ | x, z)：吃 [embed(x), z] -> θ 的 mu/logvar
        self.dec = nn.Sequential(
            nn.Linear(emb_dim + z_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.dec_mu = nn.Linear(hidden, theta_dim)
        self.dec_logvar = nn.Linear(hidden, theta_dim)

    def encode(self, theta, emb):
        h = self.enc(torch.cat([theta, emb], dim=1))
        return self.enc_mu(h), self.enc_logvar(h)

    def decode(self, emb, z):
        h = self.dec(torch.cat([emb, z], dim=1))
        return self.dec_mu(h), self.dec_logvar(h)

    @staticmethod
    def reparam(mu, logvar):
        """reparameterization trick：z = mu + sigma * eps，让梯度能穿过采样。"""
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, theta, x):
        emb = self.embed(x)
        z_mu, z_logvar = self.encode(theta, emb)
        z = self.reparam(z_mu, z_logvar)
        th_mu, th_logvar = self.decode(emb, z)
        return th_mu, th_logvar, z_mu, z_logvar


def gaussian_nll(target, mu, logvar):
    """高斯负对数似然（重构项）：-log p(θ | x, z)，让 decoder 的高斯罩住真值。"""
    return 0.5 * (logvar + (target - mu) ** 2 / torch.exp(logvar)).sum(dim=1)


def kl_standard_normal(mu, logvar):
    """KL( q(z|·) ‖ N(0,I) )：把潜变量拉回标准正态先验。"""
    return -0.5 * (1 + logvar - mu ** 2 - torch.exp(logvar)).sum(dim=1)


# ============================================================================
# 4) 训练
# ============================================================================
def train(model, th_tr, x_tr, th_va, x_va, epochs, batch, lr, beta_kl):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = th_tr.shape[0]
    hist = {"train": [], "val": []}

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for b in range(0, n, batch):
            idx = perm[b:b + batch]
            th_b, x_b = th_tr[idx], x_tr[idx]
            th_mu, th_logvar, z_mu, z_logvar = model(th_b, x_b)
            rec = gaussian_nll(th_b, th_mu, th_logvar)
            kl = kl_standard_normal(z_mu, z_logvar)
            loss = (rec + beta_kl * kl).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        tr_loss = tot / n

        # 验证
        model.eval()
        with torch.no_grad():
            th_mu, th_logvar, z_mu, z_logvar = model(th_va, x_va)
            rec = gaussian_nll(th_va, th_mu, th_logvar)
            kl = kl_standard_normal(z_mu, z_logvar)
            va_loss = (rec + beta_kl * kl).mean().item()

        hist["train"].append(tr_loss); hist["val"].append(va_loss)
        if ep % max(1, epochs // 10) == 0 or ep == 1:
            print(f"  epoch {ep:3d}/{epochs}  train {tr_loss:8.3f}  val {va_loss:8.3f}")
    return hist


# ============================================================================
# 5) 推断：给一条 x_obs，抽多个 z -> decoder 吐出多个 R0 -> posterior
#    对齐 proposal：latent draws from p(z)，single forward pass per draw
# ============================================================================
@torch.no_grad()
def posterior_samples(model, x_row, n_draws, th_scaler):
    """x_row: 已标准化的单条曲线 (1, T_FIXED) tensor。回传 n_draws 个 R0（原始尺度）。"""
    model.eval()
    emb = model.embed(x_row).repeat(n_draws, 1)          # 同一条 x 复制 n_draws 份
    z = torch.randn(n_draws, model.z_dim)                # z ~ p(z) = N(0, I)
    th_mu, th_logvar = model.decode(emb, z)              # 每个 z 一个 θ 的高斯
    std = torch.exp(0.5 * th_logvar)
    draws = th_mu + std * torch.randn_like(std)          # 再从高斯抽一次，得完整 posterior
    draws = draws.numpy()                                # 标准化尺度
    return th_scaler.inv(draws)[:, 0]                    # 反标准化 -> 原始 R0


# ============================================================================
def main():
    ap = argparse.ArgumentParser(description="toy SIR + cVAE 估计 R0（最小验证）")
    ap.add_argument("--n", type=int, default=3000, help="训练+验证总样本数")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta_kl", type=float, default=1.0, help="KL 项权重（β-VAE）")
    ap.add_argument("--z_dim", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_test", type=int, default=8, help="画几条测试曲线的 posterior")
    ap.add_argument("--n_draws", type=int, default=2000, help="每条 x 抽几个 posterior 样本")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    # ---- 生成资料 ----
    print(f"[1/4] 生成 {args.n} 组 (R0, 曲线) ...")
    R0s, X = make_dataset(args.n, seed=args.seed)

    # 切训练/验证（9:1）
    n_val = max(1, args.n // 10)
    R0_tr, X_tr = R0s[:-n_val], X[:-n_val]
    R0_va, X_va = R0s[-n_val:], X[-n_val:]

    # ---- 标准化（用训练集统计量）----
    xs = Scaler(X_tr); ths = Scaler(R0_tr)
    Xtr_s = xs.fwd(X_tr); Xva_s = xs.fwd(X_va)
    Rtr_s = ths.fwd(R0_tr); Rva_s = ths.fwd(R0_va)

    to_t = lambda a: torch.tensor(a, dtype=torch.float32)
    th_tr, x_tr = to_t(Rtr_s), to_t(Xtr_s)
    th_va, x_va = to_t(Rva_s), to_t(Xva_s)

    # ---- 训练 ----
    print(f"[2/4] 训练 cVAE（epochs={args.epochs}, z_dim={args.z_dim}, "
          f"beta_kl={args.beta_kl}）...")
    model = CVAE(theta_dim=1, z_dim=args.z_dim)
    hist = train(model, th_tr, x_tr, th_va, x_va,
                 epochs=args.epochs, batch=args.batch, lr=args.lr,
                 beta_kl=args.beta_kl)

    # ---- 在验证集上评估：posterior mean 对真值的散点 + 覆盖率 ----
    print(f"[3/4] 评估：验证集上 posterior mean vs 真值 ...")
    post_means = np.zeros(n_val)
    inside_90 = 0
    for j in range(n_val):
        draws = posterior_samples(model, x_va[j:j+1], args.n_draws, ths)
        post_means[j] = draws.mean()
        lo, hi = np.percentile(draws, [5, 95])
        if lo <= R0_va[j, 0] <= hi:
            inside_90 += 1
    truth = R0_va[:, 0]
    rmse = float(np.sqrt(np.mean((post_means - truth) ** 2)))
    corr = float(np.corrcoef(post_means, truth)[0, 1])
    cover90 = inside_90 / n_val
    print(f"    RMSE(posterior mean vs 真值) = {rmse:.3f}")
    print(f"    相关系数 corr               = {corr:.3f}")
    print(f"    90% 可信区间覆盖率          = {cover90:.2%}  (理想 ~90%)")

    # ---- 画图 ----
    print(f"[4/4] 画图 ...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # (a) 训练曲线
    ax = axes[0]
    ax.plot(hist["train"], label="train")
    ax.plot(hist["val"], label="val")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss (rec + KL)")
    ax.set_title("Training curve"); ax.legend()

    # (b) 验证集 posterior mean vs 真值：点贴近对角线 = 推得准
    ax = axes[1]
    ax.scatter(truth, post_means, s=14, alpha=0.5, color="#d95f0e")
    lim = [R0_MIN - 0.1, R0_MAX + 0.1]
    ax.plot(lim, lim, "k--", lw=1, label="y = x (perfect)")
    ax.set_xlabel("true R0"); ax.set_ylabel("posterior mean R0")
    ax.set_title(f"Recovery on validation set\nRMSE={rmse:.3f}, corr={corr:.3f}")
    ax.set_xlim(lim); ax.set_ylim(lim); ax.legend()

    # (c) 几条测试曲线的完整 posterior：竖线是真值，落在分布里 = 罩住了
    ax = axes[2]
    colors = plt.cm.viridis(np.linspace(0, 0.9, args.n_test))
    for k in range(min(args.n_test, n_val)):
        draws = posterior_samples(model, x_va[k:k+1], args.n_draws, ths)
        ax.hist(draws, bins=40, density=True, alpha=0.35,
                color=colors[k], histtype="stepfilled")
        ax.axvline(R0_va[k, 0], color=colors[k], lw=1.5, ls="--")
    ax.set_xlabel("R0"); ax.set_ylabel("posterior density")
    ax.set_title(f"{args.n_test} test posteriors\n(dashed line = true R0)")

    plt.rcParams["axes.unicode_minus"] = False
    plt.tight_layout()
    plt.savefig("vae_infer_R0_demo.png", dpi=130)
    print("图已存成 vae_infer_R0_demo.png")

    print("\n[解读]")
    print("  - 中间图的点若贴着对角线、右图真值竖线若落在各自 posterior 里，")
    print("    就说明『VAE 能把 R0 推回来』—— 老师要的最小验证通过。")
    print("  - 覆盖率接近 90% 表示不确定性也标定得合理（不过度自信/保守）。")
    print("  - 通过后再往上加：年龄结构 -> 家户 + hh_finalsize -> ψ_H -> 真实数据。")


if __name__ == "__main__":
    main()
