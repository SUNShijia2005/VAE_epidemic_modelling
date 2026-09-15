"""
Hybrid SIR-VAE: aligned with the runnable Colab model.
Default decoder: beta(t) = beta * exp(0.5 * f(t)), anchored at day zero.
Full-covariance physical posterior; four residual coordinates; ordinary ELBO.
Report dataset: 16000 training / 2000 validation / 500 test curves, 120 days.
Train for 120 epochs without KL warmup or best-epoch selection by default.

python sir_hybrid_vae.py
Original Colab main experiment:
python sir_hybrid_vae.py --n_train 8000 --n_val 600 --n_test 600 --out results_hybrid/colab_original

Historical configurations and results are documented in archive/pre_colab_alignment/.
Neural correction and learned dispersion mean this is not a strictly matched
constant-SIR control. Parameter-recovery limitations still require diagnosis.
"""

import argparse
import copy
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# ============================================================================
# 0) 先验与固定设定（人口、初始感染者、观测天数都是固定输入）
# ============================================================================
LOG_BETA_M, LOG_BETA_S = float(np.log(0.35)), 0.25      # β  中位 0.35 /天
LOG_ALPHA_M, LOG_ALPHA_S = float(np.log(0.20)), 0.20    # α  中位 0.20 (5 天病程)
LOG_RHO_M, LOG_RHO_S = float(np.log(0.10)), 0.30        # ρ  中位 10% 上报率
NB_K = 10.0                                              # 造数据用的过度离散
N_POP = 100_000.0
I0 = 20.0
T_DAYS = 120
SUBSTEP = 2                                              # 每天 2 个 Euler 子步
PHYS_NAMES = ["beta", "alpha", "rho"]


def phys_from_z(z_phys):
    """标准化潜变量 (B,3) -> (β, α, ρ)。lognormal 先验的一一映射，可导。"""
    beta = torch.exp(LOG_BETA_M + LOG_BETA_S * z_phys[:, 0])
    alpha = torch.exp(LOG_ALPHA_M + LOG_ALPHA_S * z_phys[:, 1])
    rho = torch.exp(LOG_RHO_M + LOG_RHO_S * z_phys[:, 2])
    return beta, alpha, rho


# ============================================================================
# 1) 可微 SIR：decoder 的「机理」那一半
# ============================================================================
def sir_incidence(beta, alpha, beta_mod=None, T=T_DAYS, substep=SUBSTEP):
    """展开 Euler 解 SIR，返回每日新增感染 (B, T)。对 beta/alpha/beta_mod 全程可导。

    beta_mod: (B, T) 逐日乘性修正，用于 --residual_mode beta；None 表示 β 恒定。
    """
    B = beta.shape[0]
    dev, dt = beta.device, 1.0 / substep
    S = torch.full((B,), N_POP - I0, device=dev)
    I = torch.full((B,), I0, device=dev)
    daily = []
    for t in range(T):
        b_t = beta if beta_mod is None else beta * beta_mod[:, t]
        acc = torch.zeros(B, device=dev)
        for _ in range(substep):
            new_inf = (b_t * S * I / N_POP * dt).clamp(min=0.0)
            new_inf = torch.minimum(new_inf, S)          # 不能感染超过剩余易感者
            S = S - new_inf
            I = (I + new_inf - alpha * I * dt).clamp(min=0.0)
            acc = acc + new_inf
        daily.append(acc)
    return torch.stack(daily, dim=1)


def nb_log_prob(y, mean, k):
    """NegBin(mean, dispersion k) 的 log pmf。k -> ∞ 退化为 Poisson。"""
    mean = mean.clamp(min=1e-6)
    return torch.distributions.NegativeBinomial(
        total_count=k, logits=torch.log(mean) - torch.log(k)).log_prob(y)


# ============================================================================
# 2) 造训练数据：从先验抽 z -> SIR -> 观测模型
#    先验与模型内先验完全一致，所以 KL 项是精确的（不是近似）
# ============================================================================
def make_data(n, seed=0, misspec=0.0, noise_seed_offset=999):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, 3, generator=g)
    beta, alpha, rho = phys_from_z(z)

    beta_mod = None
    if misspec > 0:                                      # 误设情形：真数据里 β 随时间下降
        t = torch.arange(T_DAYS).float()
        drop = torch.sigmoid((t - 55.0) / 8.0)           # 第 55 天前后一次「干预」
        amp = misspec * (0.5 + 0.5 * torch.rand(n, 1, generator=g))
        beta_mod = torch.exp(-amp * drop.unsqueeze(0))   # (n, T)

    with torch.no_grad():
        lam = sir_incidence(beta, alpha, beta_mod) * rho.unsqueeze(1)
        k = torch.tensor(NB_K)
        dist = torch.distributions.NegativeBinomial(
            total_count=k, logits=torch.log(lam.clamp(min=1e-6)) - torch.log(k))
        # NegativeBinomial 不吃 generator，用固定种子的全局流保证可复现
        torch.manual_seed(seed + noise_seed_offset)
        y = dist.sample()
    # beta_mod 一并返回：inspect_patterns.py 要拿它当「真 pattern」跟 NN 学到的对照
    return y, z, torch.stack([beta, alpha, rho], 1), beta_mod


# ============================================================================
# 3) encoder：x -> q(z_phys | x) 与 q(z_res | x)
# ============================================================================
class Encoder(nn.Module):
    """Full-covariance Gaussian approximation in physical latent coordinates.

    Correlated rates motivate a flexible covariance; calibration must be evaluated.
    """

    def __init__(self, z_res_dim, hidden=128, full_cov=True):
        super().__init__()
        self.full_cov = full_cov
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),        # 保留 8 个时间段的特征：α 靠曲线「形状」识别，
        )                                   # 池成 1 个点会把形状信息抹掉
        self.mlp = nn.Sequential(nn.Linear(64 * 8, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        # mu(3) + Cholesky 下三角(对角 3 + 非对角 3)；对角模式只用前 6 个
        self.head_phys = nn.Linear(hidden, 9 if full_cov else 6)
        self.z_res_dim = z_res_dim
        if z_res_dim > 0:
            self.head_res = nn.Linear(hidden, 2 * z_res_dim)
        tril = torch.tril_indices(3, 3, offset=-1)
        self.register_buffer("tril_i", tril[0])
        self.register_buffer("tril_j", tril[1])

    def forward(self, x_std):
        h = self.conv(x_std.unsqueeze(1)).flatten(1)
        h = self.mlp(h)
        o = self.head_phys(h)
        mu_p = o[:, :3]
        L = torch.diag_embed(torch.exp(0.5 * o[:, 3:6].clamp(-8, 4)))
        if self.full_cov:
            L = L.clone()
            L[:, self.tril_i, self.tril_j] = o[:, 6:9]
        if self.z_res_dim > 0:
            r = self.head_res(h)
            mu_r, lv_r = r[:, :self.z_res_dim], r[:, self.z_res_dim:].clamp(-8, 4)
        else:
            mu_r = lv_r = torch.zeros(h.shape[0], 0, device=h.device)
        return mu_p, L, mu_r, lv_r


# ============================================================================
# 4) decoder：机理 SIR + 黑箱 NN 残差
# ============================================================================
class ResidualNet(nn.Module):
    """z_res -> 一条平滑的时间修正曲线 f_t。用低阶余弦基保证平滑（不让它去拟合噪声）。"""

    def __init__(self, z_res_dim, n_basis=8, T=T_DAYS, anchor=True):
        super().__init__()
        # anchor: 强制 f(0)=0。否则「修正曲线的常数部分」与 β 本身完全共线
        # （β·exp(c) 和 β 无法区分），会把 β 的估计整体拉偏。
        self.anchor = anchor
        self.net = nn.Sequential(nn.Linear(z_res_dim, 64), nn.ReLU(),
                                 nn.Linear(64, n_basis))
        t = torch.linspace(0, np.pi, T)
        basis = torch.stack([torch.cos(k * t) for k in range(n_basis)], 0)  # (K, T)
        self.register_buffer("basis", basis)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)                # 从「无修正」出发

    def forward(self, z_res):
        f = torch.tanh(self.net(z_res) @ self.basis)     # (B, T), 值域 (-1,1)
        return f - f[:, :1] if self.anchor else f


class HybridVAE(nn.Module):
    def __init__(self, z_res_dim=4, nn_scale=0.5, residual_mode="beta",
                 full_cov=True, anchor=True):
        super().__init__()
        self.z_res_dim = z_res_dim if nn_scale > 0 else 0
        self.nn_scale = nn_scale
        self.residual_mode = residual_mode
        self.enc = Encoder(self.z_res_dim, full_cov=full_cov)
        self.res = ResidualNet(self.z_res_dim, anchor=anchor) \
            if self.z_res_dim > 0 else None
        self.log_k = nn.Parameter(torch.tensor(float(np.log(NB_K))))  # 全局过度离散

    @staticmethod
    def reparam(mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    @staticmethod
    def q_phys(mu, L):
        return torch.distributions.MultivariateNormal(mu, scale_tril=L)

    def decode(self, z_phys, z_res):
        """两个 model 在这里合流：SIR 出机理曲线，NN 出修正，ρ 缩放到观测尺度。"""
        beta, alpha, rho = phys_from_z(z_phys)
        f = self.res(z_res) if self.res is not None else None
        if f is not None and self.residual_mode == "beta":
            lam = sir_incidence(beta, alpha, beta_mod=torch.exp(self.nn_scale * f))
        else:
            lam = sir_incidence(beta, alpha)
            if f is not None:
                lam = lam * torch.exp(self.nn_scale * f)
        return (lam * rho.unsqueeze(1)).clamp(min=1e-6)


def kl_std_normal(mu, logvar):
    if mu.shape[1] == 0:
        return torch.zeros(mu.shape[0], device=mu.device)
    return (-0.5 * (1 + logvar - mu ** 2 - logvar.exp())).sum(1)


def kl_per_dim(mu, logvar):
    """逐潜维 KL( q ‖ N(0,1) )，(B, d)。free-bits 用。"""
    return -0.5 * (1 + logvar - mu ** 2 - logvar.exp())


def kl_mvn_std(mu, L):
    """KL( N(mu, LLᵀ) ‖ N(0, I) )，满协方差解析式。"""
    d = mu.shape[1]
    tr = (L ** 2).sum([1, 2])
    logdet = 2.0 * torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(1)
    return 0.5 * (tr + (mu ** 2).sum(1) - d - logdet)


def log_normal(x, mu, logvar):
    if x.shape[-1] == 0:
        return torch.zeros(x.shape[:-1], device=x.device)
    return (-0.5 * (np.log(2 * np.pi) + logvar + (x - mu) ** 2 / logvar.exp())).sum(-1)


# ============================================================================
# 5) 训练目标：K=1 -> ELBO；K>1 -> IWAE
# ============================================================================
def objective(model, x_b, y_b, beta_kl, K, free_bits=0.0):
    """返回 (loss, 每样本 log 重建) 。IWAE: -log(1/K Σ w_i)，w_i = p(y,z)/q(z|x)。"""
    B = x_b.shape[0]
    mu_p, L_p, mu_r, lv_r = model.enc(x_b)
    dR = model.z_res_dim

    def rep(v):
        return v.unsqueeze(0).expand(K, *v.shape).reshape(K * B, *v.shape[1:])

    # z_res 每个数据点只采一个、K 个 z_phys 样本共用：这样 z_res 那部分保持解析
    # KL，可以挂 free-bits（防塌缩），z_phys 那部分照常做重要性加权。由 Jensen
    # 不等式，这仍是 log p(y|x) 的合法下界。
    z_r = model.reparam(mu_r, lv_r) if dR > 0 else mu_r
    q_p = model.q_phys(rep(mu_p), rep(L_p))
    z_p = q_p.rsample()
    lam = model.decode(z_p, z_r.repeat(K, 1))
    k = model.log_k.exp()
    log_lik = nb_log_prob(y_b.repeat(K, 1), lam, k).sum(1)          # (K*B,)

    # free-bits：每个残差潜维保底留 free_bits 个 nat 的配额，配额内不罚。
    # 不加这个，z_res 会直接塌回先验，黑箱只学到一条「平均修正曲线」，
    # 无法随样本变化（实测：误设数据上各序列干预强度不同，却全被拟合成同一条）。
    kl_r = torch.clamp(kl_per_dim(mu_r, lv_r).mean(0), min=free_bits).sum() \
        if dR > 0 else torch.zeros((), device=x_b.device)

    if K == 1:                                                       # 标准 ELBO
        kl_p = kl_mvn_std(mu_p, L_p).mean()
        return -log_lik.mean() + beta_kl * (kl_p + kl_r), log_lik.mean()

    log_prior = log_normal(z_p, torch.zeros_like(z_p), torch.zeros_like(z_p))
    log_w = (log_lik + beta_kl * (log_prior - q_p.log_prob(z_p))).reshape(K, B)
    iwae = (torch.logsumexp(log_w, 0) - np.log(K)).mean()
    return -iwae + beta_kl * kl_r, log_lik.mean()


def train(model, x_tr, y_tr, x_va, y_va, args, device):
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # 余弦退火：现在瓶颈是 encoder 的摊销偏差（不是 q 的宽度），末期小学习率能压偏差
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                       eta_min=args.lr * 0.02)
    n = x_tr.shape[0]
    hist = {"train": [], "val": []}
    best_val, best_state, best_ep, since = float("inf"), None, -1, 0

    for ep in range(1, args.epochs + 1):
        beta_kl = args.beta_kl * min(1.0, ep / max(1, args.kl_warmup))
        model.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for b in range(0, n, args.batch):
            idx = perm[b:b + args.batch]
            loss, _ = objective(model, x_tr[idx], y_tr[idx], beta_kl, args.iwae,
                                 args.free_bits)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(idx)
        tr_loss = tot / n
        sched.step()

        model.eval()
        va_loss = float("nan")
        validate = args.patience > 0 or ep == 1 or ep % 30 == 0
        if validate:
            with torch.no_grad():
                va_loss = objective(model, x_va, y_va, beta_kl, args.iwae,
                                    args.free_bits)[0].item()
        hist["train"].append(tr_loss)
        hist["val"].append(va_loss)

        tag = ""
        if args.patience > 0 and ep >= args.kl_warmup:                          # warmup 期目标在变，不判早停
            if va_loss < best_val:
                best_val, best_ep, since = va_loss, ep, 0
                best_state = copy.deepcopy(model.state_dict())
                tag = "  <- best"
            else:
                since += 1
        if validate:
            print(f"  epoch {ep:3d}/{args.epochs}  beta={beta_kl:.2f}  "
                  f"train {tr_loss:10.1f}  val {va_loss:10.1f}  "
                  f"k={model.log_k.exp().item():5.1f}"
                  f"  (patience {since}/{args.patience}){tag}")
        if args.patience > 0 and ep >= args.kl_warmup and since >= args.patience:
            print(f"[early-stop] epoch {ep} 停止，回退到 epoch {best_ep}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[restore] 载入最优权重 epoch {best_ep} (val={best_val:.1f})")
    return hist, best_ep if args.patience > 0 else -1


# ============================================================================
# 6) 推断：encoder 输出即 q(β,α,ρ|x) 的后验
# ============================================================================
@torch.no_grad()
def posterior_phys(model, x_std, n_draws=400):
    mu_p, L_p, _, _ = model.enc(x_std)
    B = mu_p.shape[0]
    z = model.q_phys(mu_p, L_p).sample((n_draws,))           # (n_draws, B, 3)
    phys = torch.stack(phys_from_z(z.reshape(-1, 3)), 1)
    return phys.reshape(n_draws, B, 3).cpu().numpy()


def summarize(draws, truth, name):
    pm = draws.mean(0)
    lo, hi = np.percentile(draws, [5, 95], axis=0)
    return dict(name=name, post_mean=pm, truth=truth,
                corr=float(np.corrcoef(pm, truth)[0, 1]),
                rmse=float(np.sqrt(((pm - truth) ** 2).mean())),
                post_sd=float(draws.std(0).mean()),      # 后验宽度，用来诊断覆盖率
                bias=float((pm - truth).mean()),
                cover=float(((lo <= truth) & (truth <= hi)).mean()),
                sbc=(draws < truth[None, :]).mean(0))


def main():
    global N_POP, I0
    ap = argparse.ArgumentParser(description="Hybrid VAE：decoder = 可微 SIR + NN 残差")
    ap.add_argument("--n_train", type=int, default=16000)
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--n_val", type=int, default=2000)
    ap.add_argument("--noise_seed_offset", type=int, default=999)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--beta_kl", type=float, default=1.0)
    ap.add_argument("--kl_warmup", type=int, default=0)
    ap.add_argument("--patience", type=int, default=0, help="0: fixed epochs, use last weights (Colab)")
    ap.add_argument("--iwae", type=int, default=1,
                    help="重要性样本数 K；1 = 普通 ELBO，>1 = IWAE（收紧摊销 gap）")
    ap.add_argument("--full_cov", type=int, default=1,
                    help="q(z_phys|x) 用满协方差高斯；0 = 对角(均场，会显著低估边缘方差)")
    ap.add_argument("--free_bits", type=float, default=0.0,
                    help="每个残差潜维的最小 KL 配额(nat)，防 z_res 塌缩。默认 0："
                         "数据符合 SIR 时不该强迫黑箱去编码噪声。确认有系统性误设"
                         "(见 inspect_patterns.py 图3a 的修正幅度)时开到 ~0.5")
    ap.add_argument("--anchor", type=int, default=1,
                    help="强制残差曲线 f(0)=0，消掉与 β 的共线性")
    ap.add_argument("--z_res_dim", type=int, default=4)
    ap.add_argument("--nn_scale", type=float, default=0.5,
                    help="NN 残差强度；0 = 纯机理 VAE（对照组）")
    ap.add_argument("--residual_mode", choices=["obs", "beta"], default="beta")
    ap.add_argument("--misspec", type=float, default=0.0,
                    help=">0 则造数据时让真 β 随时间下降（模型误设），测 hybrid 的价值")
    ap.add_argument("--n_pop", type=float, default=N_POP)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="results_hybrid/colab_aligned",
                    help="输出前缀（含目录，自动创建）")
    args = ap.parse_args()

    if os.path.dirname(args.out):
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
    N_POP = args.n_pop
    I0 = max(2.0, round(N_POP * 2e-4))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] device={device}  residual_mode={args.residual_mode} "
          f"nn_scale={args.nn_scale} iwae_K={args.iwae} misspec={args.misspec}")

    # ---- 数据 ----
    n_va = args.n_val
    n_all = args.n_train + args.n_test + n_va
    y, z_true, phys_true, _ = make_data(n_all, seed=args.seed, misspec=args.misspec,
                                             noise_seed_offset=args.noise_seed_offset)
    y_te, y_va, y_tr = y[:args.n_test], y[args.n_test:args.n_test + n_va], \
        y[args.n_test + n_va:]
    ph_te = phys_true[:args.n_test].numpy()
    print(f"[data] train={len(y_tr)} val={len(y_va)} test={len(y_te)}  T={T_DAYS}  "
          f"观测峰值(中位)={y_tr.max(1).values.median():.0f}  "
          f"观测总数(中位)={y_tr.sum(1).median():.0f}")

    log_y = torch.log1p(y_tr)
    m, s = log_y.mean(), log_y.std()

    def std(v):
        return ((torch.log1p(v) - m) / s).to(device)

    x_tr, x_va, x_te = std(y_tr), std(y_va), std(y_te)
    y_tr, y_va, y_te = y_tr.to(device), y_va.to(device), y_te.to(device)

    # ---- 训练 ----
    torch.manual_seed(args.seed)  # Colab resets immediately before model construction.
    model = HybridVAE(args.z_res_dim, args.nn_scale, args.residual_mode,
                      full_cov=bool(args.full_cov),
                      anchor=bool(args.anchor)).to(device)
    hist, best_ep = train(model, x_tr, y_tr, x_va, y_va, args, device)

    # ---- 评估 ----
    model.eval()
    draws = posterior_phys(model, x_te)                       # (n_draws, B, 3)
    res = [summarize(draws[:, :, j], ph_te[:, j], PHYS_NAMES[j]) for j in range(3)]
    r0_draws = draws[:, :, 0] / draws[:, :, 1]
    res.append(summarize(r0_draws, ph_te[:, 0] / ph_te[:, 1], "R0"))

    print(f"\n  学到的过度离散 k = {model.log_k.exp().item():.2f} (真值 {NB_K})")
    print("\n  参数        RMSE     post_sd     bias     corr    cover90")
    for r in res:
        print(f"  {r['name']:8s} {r['rmse']:8.4f} {r['post_sd']:8.4f} "
              f"{r['bias']:+8.4f}  {r['corr']:6.3f}   {r['cover']:6.1%}")
    print("  [读法] cover90 目标 90%。联合检查 RMSE、bias 和覆盖率；"
          "误差来源需同模型对照验证。")

    # ---- 图 1：训练曲线 + recovery ----
    fig, axes = plt.subplots(1, 5, figsize=(21, 3.8))
    axes[0].plot(hist["train"], label="train")
    axes[0].plot(hist["val"], label="val")
    if best_ep > 0:
        axes[0].axvline(best_ep - 1, color="green", ls=":", label=f"best ep {best_ep}")
    axes[0].set_title("Training curve (-ELBO/-IWAE)")
    axes[0].set_xlabel("epoch")
    axes[0].set_yscale("symlog")
    axes[0].legend()
    for ax, r in zip(axes[1:], res):
        ax.scatter(r["truth"], r["post_mean"], s=8, alpha=0.4, color="#d95f0e")
        lim = [r["truth"].min(), r["truth"].max()]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_title(f"{r['name']}  RMSE={r['rmse']:.3f}  r={r['corr']:.2f}  "
                     f"cov={r['cover']:.0%}", fontsize=10)
        ax.set_xlabel("true")
        ax.set_ylabel("posterior mean")
    plt.rcParams["axes.unicode_minus"] = False
    plt.tight_layout()
    plt.savefig(f"{args.out}_recovery.png", dpi=120)
    plt.close()

    # ---- 图 2：decoder 拆解 + SBC 校准直方图 ----
    with torch.no_grad():
        mu_p, _, mu_r, _ = model.enc(x_te[:3])
        lam_full = model.decode(mu_p, mu_r)
        b, a, rho = phys_from_z(mu_p)
        lam_mech = sir_incidence(b, a) * rho.unsqueeze(1)
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.6))
    for j in range(3):
        ax = axes[j]
        ax.plot(y_te[j].cpu().numpy(), color="#666", lw=1, label="observed y")
        ax.plot(lam_mech[j].cpu().numpy(), color="#2c7fb8", lw=1.6,
                label="SIR only (mechanistic)")
        if model.res is not None:
            ax.plot(lam_full[j].cpu().numpy(), color="#d95f0e", lw=1.6, ls="--",
                    label="SIR + NN residual")
        ax.set_title(f"test #{j}  true β={ph_te[j,0]:.2f} α={ph_te[j,1]:.2f} "
                     f"ρ={ph_te[j,2]:.2f}", fontsize=9)
        ax.set_xlabel("day")
        if j == 0:
            ax.set_ylabel("reported cases / day")
            ax.legend(fontsize=7)
    ax = axes[3]
    for r in res[:3]:
        ax.hist(r["sbc"], bins=20, histtype="step", lw=1.5, density=True,
                label=r["name"])
    ax.axhline(1.0, color="k", ls="--", lw=1, label="uniform = calibrated")
    ax.set_xlabel("SBC rank  P(draw < truth)")
    ax.set_title("SBC calibration (flat = good)", fontsize=10)
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{args.out}_fit.png", dpi=120)
    plt.close()

    metrics = [{key: r[key] for key in ("name", "corr", "rmse", "post_sd", "bias", "cover")} for r in res]
    with open(f"{args.out}_metrics.json", "w") as handle:
        json.dump({"experiment": "colab_aligned_v1", "args": vars(args),
                   "posterior_draws": 400, "selected_epoch": best_ep if best_ep > 0 else len(hist["train"]),
                   "learned_k": model.log_k.exp().item(), "metrics": metrics}, handle, indent=2)
    torch.save({"experiment": "colab_aligned_v1", "model": model.state_dict(), "args": vars(args),
                "x_scaler": (float(m), float(s))}, f"{args.out}_model.pt")
    print(f"\n[done] {args.out}_model.pt / {args.out}_recovery.png / {args.out}_fit.png")


if __name__ == "__main__":
    main()
