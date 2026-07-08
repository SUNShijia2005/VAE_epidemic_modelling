"""
大規模訓練資料生成：generate (θ, x) at scale
================================================================================
從文獻錨定的先驗 π(θ) 抽 θ，用共享的 tau-leap 模擬器 + 觀測模型生成
x = {incidence, hh_finalsize}，批量產生 (θ, x) 配對，作為 conditional VAE 的
訓練集 / ABC-MCMC 的比對原料。對齊 proposal §3.2 / §3.3 / §3.5。

依賴：epidemic_simulator.py（同資料夾；已合併原 sir_age_hh / observation_model /
simulate_interface 三檔）。

================================================================================
【θ 的記錄規格——對齊 proposal 的推斷目標 ϑ = (q, α, βc, βh, ψM, ψH)】
================================================================================
proposal 的回歸目標是 ϑ 本身加上觀測 nuisance；R0/SAR 是 βc/βh 的一一對應
衍生量（完全共線），只作診斷，不進回歸目標。故本檔把 θ 分三類分開存：

  THETA_TARGETS_SCALAR : 純量推斷目標 + 觀測 nuisance
      beta_c, beta_h, alpha, psi_H,          ← ϑ 的純量部分
      rho, delay_mean, nb_size               ← 觀測 nuisance（推斷後邊緣化）
  THETA_TARGETS_VECTOR : 向量推斷目標
      q     (長度 A)                          ← 年齡易感性
      psi_M (長度 2)                          ← 年齡矩陣形狀 (assort, inter)
  THETA_DIAGNOSTICS    : 衍生量，非回歸目標（訓練時務必排除在 loss 外）
      R0, SAR

x 形狀固定：incidence (T_FIXED, A)，hh_finalsize (HH_VEC_LEN,)，可直接堆成張量。

================================================================================
【三點工程升級（把 40 組推到 1e4~1e5 組）】
================================================================================
  1. 多進程平行 (multiprocessing.Pool)
  2. 分片存盤 (sharded .npz)
  3. 可斷點續跑 (--resume)

【複現性】人口是固定輸入（proposal §3.2）：所有 worker 用「同一顆」由
master_seed 決定的人口種子建人口，故給定 master_seed 完全可複現（不再用
os.getpid()，避免換次執行人口就變）。每個樣本用 SeedSequence.spawn 派生獨立
隨機流，平行下不重號、可完全重現。

用法示例：
  python generate_training_data.py --n 20000 --workers 16 --shard 2000 \
      --pop 20000 --out ./training_data --seed 20240601
  python generate_training_data.py --merge --out ./training_data
"""

import argparse
import glob
import os
import time

import numpy as np

from epidemic_simulator import (
    build_population, sample_prior, simulate_theta,
    M_WALLINGA, AGE_LABELS, A, T_FIXED, HH_VEC_LEN,
)

# ---- θ 記錄規格（見檔頭說明）----
THETA_TARGETS_SCALAR = ["beta_c", "beta_h", "alpha", "psi_H",
                        "rho", "delay_mean", "nb_size"]
PSI_M_LEN = 2
THETA_DIAGNOSTICS = ["R0", "SAR"]


# ============================================================================
# worker：每進程只建一次人口，再連續生成樣本
# ============================================================================

def _worker_init(pop_target, pop_seed_int):
    """每個 worker 起來時跑一次：用「共同的」pop_seed 建同一份固定人口。"""
    global _POP, _M0
    rng_pop = np.random.default_rng(pop_seed_int)     # 不含 PID：所有 worker 人口一致、可複現
    age, hh, n_k = build_population(pop_target, rng_pop)
    N_by_age = np.bincount(age, minlength=A)
    _POP = (age, hh, n_k, N_by_age)
    _M0 = M_WALLINGA


def _simulate_one(seed_seq):
    """單一樣本：給獨立 SeedSequence，抽 θ、跑 simulate、攤平回傳。"""
    rng = np.random.default_rng(seed_seq)
    theta = sample_prior(rng, _M0, _POP[3])
    x = simulate_theta(theta, _POP, _M0, rng)

    tgt_scal = np.array([float(theta[k]) for k in THETA_TARGETS_SCALAR], np.float64)
    qv = np.asarray(theta["q"], np.float64)                # (A,)
    psi_m = np.asarray(theta["psi_M"], np.float64)         # (2,)
    diag = np.array([float(theta[k]) for k in THETA_DIAGNOSTICS], np.float64)

    inc = x["incidence"].astype(np.float32)                # (T_FIXED, A)
    hh = x["hh_finalsize"].astype(np.float32)              # (HH_VEC_LEN,)
    return tgt_scal, qv, psi_m, diag, inc, hh


# ============================================================================
# 分片生成
# ============================================================================

def _shard_path(out_dir, shard_id):
    return os.path.join(out_dir, f"shard_{shard_id:05d}.npz")


def generate_sharded(n_total, shard_size, pop_target, out_dir, master_seed,
                     workers, resume):
    os.makedirs(out_dir, exist_ok=True)
    n_shards = (n_total + shard_size - 1) // shard_size

    root_ss = np.random.SeedSequence(master_seed)
    all_child = root_ss.spawn(n_total)                       # 每樣本一顆獨立種子
    pop_seed_int = int(root_ss.generate_state(1)[0])         # 人口固定種子（所有 worker 共用）

    print(f"[plan] n_total={n_total}  shard_size={shard_size}  n_shards={n_shards}")
    print(f"[plan] pop={pop_target}  workers={workers}  seed={master_seed}")
    print(f"[plan] targets_scalar={THETA_TARGETS_SCALAR}")
    print(f"[plan] targets_vector: q(A={A}), psi_M({PSI_M_LEN})   diagnostics={THETA_DIAGNOSTICS}")
    print(f"[plan] out={out_dir}  resume={resume}")

    import multiprocessing as mp
    ctx = mp.get_context("spawn")

    t_start = time.time()
    done = 0

    for shard_id in range(n_shards):
        path = _shard_path(out_dir, shard_id)
        lo = shard_id * shard_size
        hi = min(lo + shard_size, n_total)
        n_local = hi - lo

        if resume and os.path.exists(path):
            print(f"[skip] shard {shard_id} 已存在（{n_local} 樣本），跳過")
            done += n_local
            continue

        seeds_local = all_child[lo:hi]

        tgt_buf = np.empty((n_local, len(THETA_TARGETS_SCALAR)), np.float64)
        q_buf   = np.empty((n_local, A), np.float64)
        psim_buf= np.empty((n_local, PSI_M_LEN), np.float64)
        diag_buf= np.empty((n_local, len(THETA_DIAGNOSTICS)), np.float64)
        inc_buf = np.empty((n_local, T_FIXED, A), np.float32)
        hh_buf  = np.empty((n_local, HH_VEC_LEN), np.float32)

        t0 = time.time()
        with ctx.Pool(processes=workers, initializer=_worker_init,
                      initargs=(pop_target, pop_seed_int)) as pool:
            for i, (tgt, qv, psim, diag, inc, hh) in enumerate(
                    pool.imap(_simulate_one, seeds_local, chunksize=4)):
                tgt_buf[i] = tgt; q_buf[i] = qv; psim_buf[i] = psim
                diag_buf[i] = diag; inc_buf[i] = inc; hh_buf[i] = hh
                if (i + 1) % max(1, n_local // 10) == 0:
                    rate = (i + 1) / (time.time() - t0)
                    print(f"  shard {shard_id}: {i+1}/{n_local}  ({rate:.1f} samp/s)")

        np.savez_compressed(
            path,
            theta_targets_scalar=tgt_buf,
            theta_targets_scalar_names=np.array(THETA_TARGETS_SCALAR),
            q=q_buf,
            psi_M=psim_buf,
            theta_diagnostics=diag_buf,
            theta_diagnostics_names=np.array(THETA_DIAGNOSTICS),
            incidence=inc_buf,
            hh_finalsize=hh_buf,
            age_labels=np.array(AGE_LABELS),
        )
        done += n_local
        dt = time.time() - t0
        print(f"[save] shard {shard_id} -> {path}  "
              f"({n_local} 樣本, {dt:.1f}s, {n_local/dt:.1f} samp/s)  累計 {done}/{n_total}")

    total_dt = time.time() - t_start
    print(f"\n[done] {done} 樣本, 總耗時 {total_dt/60:.1f} 分, "
          f"平均 {done/max(total_dt,1e-9):.1f} samp/s")


# ============================================================================
# 合併分片（可選）
# ============================================================================

def merge_shards(out_dir):
    paths = sorted(glob.glob(os.path.join(out_dir, "shard_*.npz")))
    if not paths:
        print("沒有找到分片可合併。"); return
    keys = ["theta_targets_scalar", "q", "psi_M", "theta_diagnostics",
            "incidence", "hh_finalsize"]
    acc = {k: [] for k in keys}
    for p in paths:
        d = np.load(p, allow_pickle=True)
        for k in keys:
            acc[k].append(d[k])
    merged = {k: np.concatenate(v) for k, v in acc.items()}
    out = os.path.join(out_dir, "training_data.npz")
    np.savez_compressed(
        out,
        theta_targets_scalar_names=np.array(THETA_TARGETS_SCALAR),
        theta_diagnostics_names=np.array(THETA_DIAGNOSTICS),
        age_labels=np.array(AGE_LABELS),
        **merged,
    )
    n = len(merged["theta_targets_scalar"])
    print(f"[merge] {len(paths)} 片 -> {out}  共 {n} 樣本  "
          f"incidence {merged['incidence'].shape}, hh {merged['hh_finalsize'].shape}")


def iter_shards(out_dir):
    """訓練時逐片 yield，省記憶體。回傳 (tgt_scalar, q, psi_M, incidence, hh)。"""
    for p in sorted(glob.glob(os.path.join(out_dir, "shard_*.npz"))):
        d = np.load(p, allow_pickle=True)
        yield (d["theta_targets_scalar"], d["q"], d["psi_M"],
               d["incidence"], d["hh_finalsize"])


# ============================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="大規模生成 (θ, x) 訓練資料")
    ap.add_argument("--n", type=int, default=20000, help="總樣本數")
    ap.add_argument("--shard", type=int, default=2000, help="每片樣本數")
    ap.add_argument("--pop", type=int, default=20000, help="人口規模 N")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", type=str, default="./training_data")
    ap.add_argument("--seed", type=int, default=20240601)
    ap.add_argument("--resume", action="store_true", help="跳過已存在的分片")
    ap.add_argument("--merge", action="store_true", help="只合併已生成的分片後退出")
    args = ap.parse_args()

    if args.merge:
        merge_shards(args.out)
    else:
        generate_sharded(
            n_total=args.n, shard_size=args.shard, pop_target=args.pop,
            out_dir=args.out, master_seed=args.seed,
            workers=args.workers, resume=args.resume,
        )
