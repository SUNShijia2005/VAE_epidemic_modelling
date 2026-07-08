"""
数据体检 check_training_data.py
================================================================================
几秒钟跑完，确认生成的 (θ, x) 训练集健不健康。重点查四件事：
  1. 空疫情样本：初始 40 个感染者有时直接熄火，incidence 全 0 或极小 ——
     这种样本对 VAE 是纯噪声，占比过高要处理（加大 I0、或过滤）。
  2. 先验覆盖：R0、SAR、rho 等有没有铺满先验范围（没有断层/塌缩）。
  3. θ 和 x 有没有 NaN / Inf（数值炸掉）。
  4. 家户信号：SAR 越高，size-2 家户「两人都感染」的比例应越高
     —— 这是 proposal 说的、让网路能学到 beta_h 的关键信号，画出来确认它真的存在。

用法：
  python check_training_data.py --dir ./training_data
"""

import argparse
import glob
import os

import numpy as np
import matplotlib.pyplot as plt


def load_all(data_dir):
    paths = sorted(glob.glob(os.path.join(data_dir, "shard_*.npz")))
    if not paths:
        raise SystemExit(f"在 {data_dir} 找不到 shard_*.npz")
    scal, q, inc, hh = [], [], [], []
    names = None
    for p in paths:
        d = np.load(p, allow_pickle=True)
        scal.append(d["theta_scalars"]); q.append(d["q"])
        inc.append(d["incidence"]); hh.append(d["hh_finalsize"])
        if names is None:
            names = list(d["theta_scalar_names"])
    return (np.concatenate(scal), np.concatenate(q),
            np.concatenate(inc), np.concatenate(hh), names, len(paths))


def hh_index(size, k, sizes=(1, 2, 3, 4, 5, 6)):
    return sum(s + 1 for s in sizes if s < size) + k


def main(data_dir):
    scal, q, inc, hh, names, n_shards = load_all(data_dir)
    n = len(scal)
    idx = {nm: i for i, nm in enumerate(names)}
    R0 = scal[:, idx["R0"]]; SAR = scal[:, idx["SAR"]]
    rho = scal[:, idx["rho"]]; alpha = scal[:, idx["alpha"]]

    print(f"===== 数据体检：{data_dir} =====")
    print(f"分片数 {n_shards}，样本数 {n}")
    print(f"incidence 形状 {inc.shape}，hh_finalsize 形状 {hh.shape}，q 形状 {q.shape}")

    # 1. NaN / Inf
    bad_theta = (~np.isfinite(scal)).any(1) | (~np.isfinite(q)).any(1)
    bad_x = (~np.isfinite(inc)).any((1, 2)) | (~np.isfinite(hh)).any(1)
    print(f"\n[1] 数值检查：θ 含 NaN/Inf 的样本 {bad_theta.sum()}，"
          f"x 含 NaN/Inf 的样本 {bad_x.sum()}  "
          f"{'✅ 干净' if (bad_theta.sum()+bad_x.sum())==0 else '⚠️ 有问题'}")

    # 2. 空疫情：整条 incidence 的总通报数
    total_reported = inc.sum((1, 2))       # 每个样本一生的总通报病例
    dead = total_reported == 0
    tiny = total_reported < 5              # 几乎没起来
    print(f"\n[2] 疫情规模（总通报病例 / 样本）：")
    print(f"    中位数 {np.median(total_reported):.0f}，"
          f"5%分位 {np.percentile(total_reported,5):.0f}，"
          f"95%分位 {np.percentile(total_reported,95):.0f}")
    print(f"    完全没通报 (=0) 的样本：{dead.sum()} ({dead.mean()*100:.1f}%)")
    print(f"    几乎没起来 (<5)  的样本：{tiny.sum()} ({tiny.mean()*100:.1f}%)")
    if tiny.mean() > 0.15:
        print("    ⚠️ 空/微疫情占比偏高(>15%)。这些样本对 VAE 信息量低。")
        print("       可考虑：把 simulate 的初始感染者从 40 调高，或训练时下采样这些样本。")
    else:
        print("    ✅ 空疫情占比可接受。")

    # 3. 先验覆盖
    print(f"\n[3] 先验覆盖（应铺满、无断层）：")
    for nm, v, lo, hi in [("R0", R0, 1.2, 2.3), ("SAR", SAR, 0.02, 0.19),
                          ("rho", rho, 0.005, 0.05)]:
        print(f"    {nm:>4}: [{v.min():.3f}, {v.max():.3f}]  "
              f"(先验区间约 [{lo}, {hi}])  均值 {v.mean():.3f}")

    # 4. 家户信号：SAR vs size-2 两人都感染的比例
    both2 = hh[:, hh_index(2, 2)]
    # 只看有起疫情的样本，空样本的家户比例没意义
    live = ~tiny
    if live.sum() > 10:
        corr = np.corrcoef(SAR[live], both2[live])[0, 1]
        print(f"\n[4] 家户识别信号：corr(SAR, P(both|size-2)) = {corr:.3f}  "
              f"{'✅ 正相关，信号在' if corr > 0.2 else '⚠️ 相关性弱，需检查'}")
    else:
        corr = np.nan
        print("\n[4] 有效样本太少，跳过家户信号检查")

    # ---- 画 4 张图 ----
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    ax[0, 0].hist(np.log10(np.maximum(total_reported, 0.5)), bins=40, color="#4c72b0")
    ax[0, 0].set_title(f"total reported per sample (log10)\n"
                       f"dead={dead.mean()*100:.1f}%, tiny(<5)={tiny.mean()*100:.1f}%")
    ax[0, 0].set_xlabel("log10(total reported cases)")

    ax[0, 1].hist(R0, bins=40, color="#55a868")
    ax[0, 1].set_title("R0 prior coverage"); ax[0, 1].set_xlabel("R0")

    ax[1, 0].hist(SAR * 100, bins=40, color="#c44e52")
    ax[1, 0].set_title("SAR prior coverage"); ax[1, 0].set_xlabel("SAR (%)")

    if live.sum() > 10:
        ax[1, 1].scatter(SAR[live] * 100, both2[live] * 100, s=6, alpha=0.3,
                         c=R0[live], cmap="viridis")
        ax[1, 1].set_title(f"household signal (corr={corr:.2f})\n"
                           "P(both infected | size-2) vs SAR")
        ax[1, 1].set_xlabel("SAR (%)")
        ax[1, 1].set_ylabel("P(both | size-2) (%)")

    plt.tight_layout()
    out = os.path.join(data_dir, "data_health_check.png")
    plt.savefig(out, dpi=120)
    print(f"\n图已存成 {out}")

    # ---- 一句话总结 ----
    ok = (bad_theta.sum() + bad_x.sum() == 0) and tiny.mean() <= 0.15 \
         and (np.isnan(corr) or corr > 0.2)
    print("\n" + ("✅ 总体：数据健康，可以拿去训练 VAE。"
                  if ok else
                  "⚠️ 总体：有需要留意的地方，见上面 ⚠️ 标记。"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default="./training_data")
    main(ap.parse_args().dir)