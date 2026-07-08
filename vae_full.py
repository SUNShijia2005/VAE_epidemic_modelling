"""
全量 cVAE（自回归 decoder + free-bits）：专治 q 的 posterior collapse
================================================================================
基线 vae_full.py 已验证方法 work（平均 corr(ϑ)≈0.57、覆盖率≈89%），但 recovery
图显示 q[20-39]/q[40-59]/q[60+] 出现「横云」——decoder 对这些弱信息维度直接吐先验
均值（posterior collapse）。实验也证明：调低 beta_kl 会发散，不是解法。

本版做两处结构性升级，其余（数据、embedding、encoder、评估、画图）与基线完全一致：

  1. 自回归 decoder（对齐 proposal 的链式分解 p(q,α,βc,βh|x,z)·p(ψM|x,z)·p(ψH|ψM,·)）
     基线：14 维一次性从一个高斯头吐出，各维条件独立 → 弱维度塌回先验。
     本版：按固定顺序逐维生成，第 k 维 p(θ_k | x, z, θ_{<k}) 能看到前面已定的维度。
           顺序刻意设计为：强识别参数(beta_c,beta_h,alpha,psi_H) → q 链(每个 q 看得到
           前面的 q 和强参数) → psi_M → nuisance(rho,delay,nb) 垫底。
           年龄易感性 q 相邻组本就强相关，让难的 q[60+] 参考已推准的邻居，
           就能把它从横云里拽起来。训练用 teacher forcing（条件于真值 θ_{<k}），
           推断用逐维采样（条件于已采样的 θ_{<k}）。

  2. free-bits KL（替代「调 beta_kl」这条错路）
     ------------------------------------------------------------------------
     给每个潜维度留一个最小 KL 配额（free_bits，单位 nat）：配额内不惩罚。
     既防潜变量塌缩、又不像削 beta_kl 那样引发发散。beta_kl 仍固定 1.0。

用法（默认存成 vae_ar_*，不覆盖基线 vae_full_*）：
  python vae_full_ar.py --data simulate_data/training_data/training_data.npz --epochs 120
  # 想调 free-bits 强度： --free_bits 0.2
"""

import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ============================================================================
# 0) 读数据 + 组装 14 维目标（剔除固定的 q 参考组）——与基线一致
# ============================================================================
def load_data(path):
    d = np.load(path, allow_pickle=True)
    inc = d["incidence"].astype(np.float32)
    hh  = d["hh_finalsize"].astype(np.float32)
    scal = d["theta_targets_scalar"].astype(np.float32)
    q    = d["q"].astype(np.float32)
    psiM = d["psi_M"].astype(np.float32)
    scal_names = list(map(str, d["theta_targets_scalar_names"]))
    age_labels = list(map(str, d["age_labels"]))

    keep = q.std(0) > 1e-8
    ref_idx = int(np.where(~keep)[0][0]) if (~keep).any() else -1
    q_free = q[:, keep]
    q_free_names = [f"q[{age_labels[i]}]" for i in range(len(age_labels)) if keep[i]]

    theta = np.concatenate([scal, q_free, psiM], axis=1)
    names = scal_names + q_free_names + ["psiM_assort", "psiM_inter"]
    nuisance = np.array([n in ("rho", "delay_mean", "nb_size") for n in names])

    print(f"[data] N={inc.shape[0]}  incidence{inc.shape}  hh{hh.shape}  theta{theta.shape}")
    if ref_idx >= 0:
        print(f"[data] 剔除固定参考组 q[{age_labels[ref_idx]}]=1.0 -> 目标维度 {theta.shape[1]}")
    return inc, hh, theta, names, nuisance


# ============================================================================
# 1) 标准化——与基线一致
# ============================================================================
class Scaler:
    def __init__(self, arr, axis, log1p=False):
        self.log1p = log1p
        a = np.log1p(arr) if log1p else arr
        self.mean = a.mean(axis=axis, keepdims=True)
        self.std = a.std(axis=axis, keepdims=True) + 1e-6

    def fwd(self, arr):
        a = np.log1p(arr) if self.log1p else arr
        return ((a - self.mean) / self.std).astype(np.float32)

    def inv(self, arr):
        a = arr * self.std + self.mean
        return np.expm1(a) if self.log1p else a


# ============================================================================
# 2) shared embedding——与基线一致
# ============================================================================
class Embedding(nn.Module):
    def __init__(self, t_len=180, n_age=6, hh_len=27, emb_dim=64):
        super().__init__()
        self.inc_conv = nn.Sequential(
            nn.Conv1d(n_age, 32, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.inc_head = nn.Linear(64, 64)
        self.hh_mlp = nn.Sequential(
            nn.Linear(hh_len, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
        )
        self.fuse = nn.Sequential(nn.Linear(64 + 32, emb_dim), nn.ReLU())

    def forward(self, inc, hh):
        c = self.inc_conv(inc.transpose(1, 2)).squeeze(-1)
        c = torch.relu(self.inc_head(c))
        h = self.hh_mlp(hh)
        return self.fuse(torch.cat([c, h], dim=1))


# ============================================================================
# 3) 自回归 decoder：p(θ_k | x, z, θ_{<k})，逐维生成
#    共享 context = base([emb, z])；每个维度一个小头，吃 [context, θ_{<k}]。
# ============================================================================
class ARDecoder(nn.Module):
    def __init__(self, theta_dim, emb_dim, z_dim, hidden=128, head_hidden=64):
        super().__init__()
        self.theta_dim = theta_dim
        self.base = nn.Sequential(
            nn.Linear(emb_dim + z_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        # 第 k 个头：输入 [context(hidden), θ_{<k}(k 维)] -> (mu_k, logvar_k)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden + k, head_hidden), nn.ReLU(),
                          nn.Linear(head_hidden, 2))
            for k in range(theta_dim)
        ])

    def forward_teacher(self, emb, z, theta_ar):
        """训练：teacher forcing，第 k 维条件于 AR 顺序下的真值 theta_ar[:, :k]。"""
        h = self.base(torch.cat([emb, z], dim=1))
        mus, lvs = [], []
        for k, head in enumerate(self.heads):
            inp = torch.cat([h, theta_ar[:, :k]], dim=1) if k > 0 else h
            out = head(inp)
            mus.append(out[:, 0:1]); lvs.append(out[:, 1:2])
        mu = torch.cat(mus, dim=1)
        lv = torch.cat(lvs, dim=1).clamp(-8, 6)
        return mu, lv                                    # AR 顺序

    @torch.no_grad()
    def sample(self, emb, z):
        """推断：逐维采样，第 k 维条件于已采样的 θ_{<k}（AR 顺序，标准化尺度）。"""
        h = self.base(torch.cat([emb, z], dim=1))
        theta = torch.zeros(h.shape[0], 0, device=h.device)
        for k, head in enumerate(self.heads):
            inp = torch.cat([h, theta], dim=1) if k > 0 else h
            out = head(inp)
            mu, lv = out[:, 0:1], out[:, 1:2].clamp(-8, 6)
            xk = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
            theta = torch.cat([theta, xk], dim=1)
        return theta                                     # AR 顺序


class CVAE(nn.Module):
    def __init__(self, theta_dim, ar_order, emb_dim=64, z_dim=16, hidden=128,
                 inc_shape=(180, 6), hh_len=27, use_data_dec=True):
        super().__init__()
        self.theta_dim, self.z_dim = theta_dim, z_dim
        self.ar_order = list(ar_order)
        inv = [0] * theta_dim
        for k, i in enumerate(self.ar_order):
            inv[i] = k
        self.inv_order = inv                             # AR 顺序 -> 原始顺序
        self.inc_shape = inc_shape
        self.hh_len = hh_len
        self.use_data_dec = use_data_dec

        self.embed = Embedding(emb_dim=emb_dim)
        self.enc = nn.Sequential(
            nn.Linear(theta_dim + emb_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.enc_mu = nn.Linear(hidden, z_dim)
        self.enc_logvar = nn.Linear(hidden, z_dim)
        self.dec = ARDecoder(theta_dim, emb_dim, z_dim, hidden)

        # ---- 数据 decoder p(x|z)：proposal 目标的第三项(辅助，keeps z informative) ----
        # 从 z 重建标准化后的 x = [incidence 展平, hh_finalsize]，对角高斯。
        self.x_dim = inc_shape[0] * inc_shape[1] + hh_len
        if use_data_dec:
            self.data_dec = nn.Sequential(
                nn.Linear(z_dim, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
            )
            self.data_mu = nn.Linear(256, self.x_dim)
            self.data_logvar = nn.Linear(256, self.x_dim)

    def decode_data(self, z):
        h = self.data_dec(z)
        return self.data_mu(h), self.data_logvar(h).clamp(-8, 6)

    def encode(self, theta, emb):
        h = self.enc(torch.cat([theta, emb], dim=1))
        return self.enc_mu(h), self.enc_logvar(h).clamp(-8, 6)

    @staticmethod
    def reparam(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, theta, inc, hh):
        emb = self.embed(inc, hh)
        z_mu, z_logvar = self.encode(theta, emb)
        z = self.reparam(z_mu, z_logvar)
        theta_ar = theta[:, self.ar_order]               # teacher forcing 用 AR 顺序真值
        th_mu, th_lv = self.dec.forward_teacher(emb, z, theta_ar)
        if self.use_data_dec:                            # p(x|z) 辅助重建
            x_mu, x_lv = self.decode_data(z)
        else:
            x_mu, x_lv = None, None
        return th_mu, th_lv, z_mu, z_logvar, x_mu, x_lv  # th_* 为 AR 顺序


def gaussian_nll(target, mu, logvar):
    return 0.5 * (logvar + (target - mu) ** 2 / torch.exp(logvar)).sum(dim=1)


def kl_per_dim(mu, logvar):
    """逐潜维 KL( q(z_i|·) ‖ N(0,1) )，回传 (B, z_dim)。"""
    return -0.5 * (1 + logvar - mu ** 2 - torch.exp(logvar))


# ============================================================================
# 4) 训练：free-bits KL（每潜维保底配额）+ beta warmup
# ============================================================================
def train(model, tr, va, epochs, batch, lr, beta_kl, kl_warmup, free_bits,
          patience, min_delta, recon_x, device):
    """训练 + 早停。

    早停照搬地震 NPP 代码那三条：盯 val loss、patience 轮无改善即停、回退到最优权重。
    关键区别：本脚本有 KL warmup，前 kl_warmup 轮 beta 在变，val loss 在变目标下
    不可比（会假性冲到 -6~-10）。所以早停只在 warmup 结束（beta 到满）后才开始判定，
    否则会把 warmup 期的假最优存下来。

    目标对齐 proposal：L = KL(free-bits) - E[log p(ϑ|x,z)] - recon_x·E[log p(x|z)]
    第三项是数据 decoder（辅助项，proposal 中标注 optional）；recon_x=0 可关闭。
    """
    th_tr, inc_tr, hh_tr = tr
    th_va, inc_va, hh_va = va
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = th_tr.shape[0]
    hist = {"train": [], "val": []}
    ar = model.ar_order

    def batch_loss(th_b, inc_b, hh_b, beta):
        th_mu, th_lv, z_mu, z_lv, x_mu, x_lv = model(th_b, inc_b, hh_b)
        target_ar = th_b[:, ar]                                  # 目标也换成 AR 顺序
        rec = gaussian_nll(target_ar, th_mu, th_lv).mean()      # -E[log p(ϑ|x,z)]
        kld = kl_per_dim(z_mu, z_lv).mean(0)                    # (z_dim,) 批平均
        kld_fb = torch.clamp(kld, min=free_bits).sum()         # free-bits
        loss = rec + beta * kld_fb
        if recon_x > 0 and x_mu is not None:                   # -recon_x·E[log p(x|z)]
            B = th_b.shape[0]
            x_target = torch.cat([inc_b.reshape(B, -1), hh_b], dim=1)  # 标准化尺度
            data_nll = gaussian_nll(x_target, x_mu, x_lv).mean()
            loss = loss + recon_x * data_nll
        return loss

    best_val = float("inf")
    best_state = None
    best_epoch = -1
    since_improve = 0

    for ep in range(1, epochs + 1):
        beta = beta_kl * min(1.0, ep / max(1, kl_warmup))
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for b in range(0, n, batch):
            idx = perm[b:b + batch]
            loss = batch_loss(th_tr[idx], inc_tr[idx], hh_tr[idx], beta)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        tr_loss = tot / n

        model.eval()
        with torch.no_grad():
            va_loss = batch_loss(th_va, inc_va, hh_va, beta).item()
        hist["train"].append(tr_loss); hist["val"].append(va_loss)

        # ---- 早停判定：只在 warmup 结束后开始（beta 已固定为满值）----
        tag = ""
        if ep >= kl_warmup:
            if va_loss < best_val - min_delta:
                best_val = va_loss
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = ep
                since_improve = 0
                tag = "  <- best"
            else:
                since_improve += 1

        if ep % 10 == 0 or ep == 1 or tag:
            print(f"  epoch {ep:3d}/{epochs}  beta={beta:.2f}  "
                  f"train {tr_loss:8.3f}  val {va_loss:8.3f}"
                  f"  (patience {since_improve}/{patience}){tag}")

        if ep >= kl_warmup and since_improve >= patience:
            print(f"[early-stop] val loss 连续 {patience} 轮无改善，"
                  f"在 epoch {ep} 停止；回退到最优 epoch {best_epoch}(val={best_val:.3f})")
            break

    # ---- 回退到最优权重（restore_best_weights）----
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[restore] 已载入最优权重 epoch {best_epoch}(val={best_val:.3f})")
    else:
        print("[warn] 没触发早停判定（epochs <= kl_warmup？），用最后一版权重")
    return hist, best_epoch


# ============================================================================
# 5) 推断：逐维自回归采样 -> 换回原始顺序 -> 反标准化
# ============================================================================
@torch.no_grad()
def posterior_samples(model, inc_row, hh_row, n_draws, th_scaler, device):
    model.eval()
    emb = model.embed(inc_row.to(device), hh_row.to(device)).repeat(n_draws, 1)
    z = torch.randn(n_draws, model.z_dim, device=device)
    draws_ar = model.dec.sample(emb, z)                        # (n_draws, D) AR 顺序
    draws = draws_ar[:, model.inv_order].cpu().numpy()         # -> 原始顺序
    return th_scaler.inv(draws)


# ============================================================================
def main():
    ap = argparse.ArgumentParser(description="全量 cVAE（自回归 decoder + free-bits）")
    ap.add_argument("--data", type=str, default="training_data.npz")
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=300, help="上限；早停通常会提前停")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta_kl", type=float, default=1.0)
    ap.add_argument("--kl_warmup", type=int, default=30)
    ap.add_argument("--free_bits", type=float, default=0.1, help="每潜维最小 KL 配额(nat)")
    ap.add_argument("--patience", type=int, default=10, help="val 连续几轮无改善即早停")
    ap.add_argument("--min_delta", type=float, default=0.0, help="视为改善的最小降幅")
    ap.add_argument("--recon_x", type=float, default=0.1,
                    help="数据 decoder p(x|z) 的权重(proposal 第三项，optional)；0 关闭")
    ap.add_argument("--z_dim", type=int, default=16)
    ap.add_argument("--n_eval", type=int, default=1000)
    ap.add_argument("--n_draws", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="vae_ar")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] device={device}")

    inc, hh, theta, names, nuisance = load_data(args.data)
    if args.subset and args.subset < len(inc):
        inc, hh, theta = inc[:args.subset], hh[:args.subset], theta[:args.subset]
        print(f"[data] subset -> {len(inc)}")

    # ---- 构造自回归生成顺序：严格对齐 proposal 的分解 ----
    #   p(ϑ|x,z) = p(q,α,βc,βh | x,z) · p(ψM | x,z) · p(ψH | ψM, x,z)
    #   即：block1(βc,βh,α + q链) 先出 -> ψM(assort,inter) -> ψH 条件于 ψM(放 ψM 之后)
    #       -> 观测 nuisance(rho,delay,nb) 垫底(不污染上面的条件链)
    #   注意：psi_H 必须排在 psi_M 之后，这样 decoder 生成 ψH 时能看到已生成的 ψM，
    #   才是 proposal 要的 p(ψH|ψM,·)；之前把 psi_H 放前面是错的，方向反了。
    block1 = ["beta_c", "beta_h", "alpha"] + [nm for nm in names if nm.startswith("q[")]
    psim = ["psiM_assort", "psiM_inter"]
    psih = ["psi_H"]
    nuis = ["rho", "delay_mean", "nb_size"]
    desired = [nm for nm in (block1 + psim + psih + nuis) if nm in names]
    for nm in names:                                     # 兜底：漏掉的补到最后
        if nm not in desired:
            desired.append(nm)
    ar_order = [names.index(nm) for nm in desired]
    print(f"[AR] decoder 生成顺序(对齐 proposal p(ψH|ψM)): {desired}")

    # ---- 切分 ----
    n = len(inc)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(n)
    n_test = min(args.n_eval, n // 10)
    n_cal, n_val = n // 10, n // 10
    te, ca, va, tr = idx[:n_test], idx[n_test:n_test+n_cal], \
                     idx[n_test+n_cal:n_test+n_cal+n_val], idx[n_test+n_cal+n_val:]
    print(f"[split] train={len(tr)} val={len(va)} cal={len(ca)} test={len(te)}")

    # ---- 标准化 ----
    inc_sc = Scaler(inc[tr], axis=(0, 1), log1p=True)
    hh_sc  = Scaler(hh[tr],  axis=0,      log1p=True)
    th_sc  = Scaler(theta[tr], axis=0,    log1p=False)

    def pack(ix):
        return (torch.tensor(th_sc.fwd(theta[ix])).to(device),
                torch.tensor(inc_sc.fwd(inc[ix])).to(device),
                torch.tensor(hh_sc.fwd(hh[ix])).to(device))
    tr_t, va_t = pack(tr), pack(va)

    # ---- 训练 ----
    print(f"[train] epochs<={args.epochs} z_dim={args.z_dim} "
          f"beta_kl={args.beta_kl} free_bits={args.free_bits} "
          f"recon_x={args.recon_x} patience={args.patience}(warmup 后生效)")
    model = CVAE(theta_dim=theta.shape[1], ar_order=ar_order, z_dim=args.z_dim,
                 inc_shape=inc.shape[1:], hh_len=hh.shape[1],
                 use_data_dec=(args.recon_x > 0)).to(device)
    hist, best_epoch = train(model, tr_t, va_t, args.epochs, args.batch, args.lr,
                             args.beta_kl, args.kl_warmup, args.free_bits,
                             args.patience, args.min_delta, args.recon_x, device)

    # ---- 评估（与基线完全一致）----
    print(f"[eval] test={len(te)} 条, 每条抽 {args.n_draws} 个 posterior 样本 ...")
    inc_te = torch.tensor(inc_sc.fwd(inc[te]))
    hh_te  = torch.tensor(hh_sc.fwd(hh[te]))
    truth = theta[te]
    D = theta.shape[1]
    post_mean = np.zeros((len(te), D)); cover90 = np.zeros(D)
    sbc_ranks = np.zeros((len(te), D))
    for j in range(len(te)):
        draws = posterior_samples(model, inc_te[j:j+1], hh_te[j:j+1],
                                  args.n_draws, th_sc, device)
        post_mean[j] = draws.mean(0)
        lo, hi = np.percentile(draws, [5, 95], axis=0)
        cover90 += ((lo <= truth[j]) & (truth[j] <= hi)).astype(float)
        sbc_ranks[j] = (draws < truth[j]).mean(0)
    cover90 /= len(te)

    rmse = np.sqrt(((post_mean - truth) ** 2).mean(0))
    corr = np.array([np.corrcoef(post_mean[:, k], truth[:, k])[0, 1] for k in range(D)])

    print("\n  参数            RMSE     corr    cover90   类型")
    for k in range(D):
        tag = "nuisance" if nuisance[k] else "ϑ target"
        print(f"  {names[k]:14s} {rmse[k]:7.4f}  {corr[k]:5.3f}   {cover90[k]:5.1%}   {tag}")
    print(f"\n  [汇总] 平均 corr(ϑ target only) = {corr[~nuisance].mean():.3f}   "
          f"平均 cover90(全部) = {cover90.mean():.1%}  (理想 ~90%)")

    # ---- 画图 1 ----
    ncol = 4; nrow = int(np.ceil((D + 1) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.4 * nrow)); axes = axes.ravel()
    ax = axes[0]; ax.plot(hist["train"], label="train"); ax.plot(hist["val"], label="val")
    if best_epoch > 0:
        ax.axvline(best_epoch - 1, color="green", ls=":", lw=1.5,
                   label=f"best (ep {best_epoch})")
    ax.set_title("Training curve"); ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend()
    for k in range(D):
        ax = axes[k + 1]
        col = "#7f7f7f" if nuisance[k] else "#d95f0e"
        ax.scatter(truth[:, k], post_mean[:, k], s=8, alpha=0.4, color=col)
        lim = [truth[:, k].min(), truth[:, k].max()]; ax.plot(lim, lim, "k--", lw=1)
        ax.set_title(f"{names[k]}\nRMSE={rmse[k]:.3f} r={corr[k]:.2f}", fontsize=9)
        ax.set_xlabel("true"); ax.set_ylabel("post mean")
    for k in range(D + 1, len(axes)):
        axes[k].axis("off")
    plt.rcParams["axes.unicode_minus"] = False
    plt.tight_layout(); plt.savefig(f"{args.out}_recovery.png", dpi=120); plt.close()

    # ---- 画图 2 ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5))
    colors = ["#7f7f7f" if nuisance[k] else "#d95f0e" for k in range(D)]
    a1.bar(range(D), cover90, color=colors)
    a1.axhline(0.9, color="k", ls="--", lw=1, label="target 0.90")
    a1.set_xticks(range(D)); a1.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    a1.set_ylabel("90% CI coverage"); a1.set_title("Per-parameter coverage"); a1.legend()
    a2.hist(sbc_ranks.ravel(), bins=20, density=True, color="#2c7fb8", alpha=0.8)
    a2.axhline(1.0, color="k", ls="--", lw=1, label="uniform (calibrated)")
    a2.set_xlabel("SBC rank (P(draw < truth))"); a2.set_ylabel("density")
    a2.set_title("SBC rank histogram (flat = calibrated)"); a2.legend()
    plt.tight_layout(); plt.savefig(f"{args.out}_coverage_sbc.png", dpi=120); plt.close()

    torch.save({"model": model.state_dict(), "ar_order": ar_order,
                "names": names, "nuisance": nuisance,
                "inc_sc": vars(inc_sc), "hh_sc": vars(hh_sc), "th_sc": vars(th_sc),
                "args": vars(args)}, f"{args.out}_model.pt")
    print(f"\n[done] 存档：{args.out}_model.pt / {args.out}_recovery.png / {args.out}_coverage_sbc.png")
    print("[比较] 对着 vae_full_recovery.png 看 q[20-39]~q[60+]：横云若立起来、r 上升 => AR 起作用。")


if __name__ == "__main__":
    main()