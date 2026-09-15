# Epidemic VAE — 两套架构

仓库里有**两套方向相反**的 VAE，不要混用。

| | 旧：conditional VAE | 新：hybrid VAE（当前主线） |
|---|---|---|
| encoder | q(z \| θ, x) | q(β, α, ρ \| x) — **潜空间就是参数** |
| decoder | NN 吐参数 θ | **可微 SIR + NN 残差** — 吐数据 x |
| 训练数据 | 必须有 (θ, x) 配对（离线模拟 ~10h） | 只要观测 x，无需标签 |
| 主脚本 | `vae_full.py` | `sir_hybrid_vae.py` |

新架构来自 2026-07 与老师的讨论：参数估计只做 β 和 α，decoder 里装两个 model
（一个 SIR、一个 neural network），低维潜空间直接包含 β、α。现阶段用最简 SIR
（单一均质人群，无年龄/家户结构），PINN 暂不做。

---

## 新架构（hybrid VAE）

### 文件

| 文件 | 作用 |
|---|---|
| `sir_hybrid_vae.py` | 模型 + 训练 + 评估。改这个 |
| `check_posterior_vs_grid.py` | 3 参数网格数值参考；需验证网格收敛与模型一致性，再比较 VAE 后验 |
| `inspect_patterns.py` | 看数据里的 pattern：潜空间地图、潜维度遍历、黑箱诊断 |
| `results_hybrid/` | 当前基线模型与图 |

### 实验口径与当前文档

`SIR_VAE_report_revised.ipynb` 说明生成过程、推断过程与下一步诊断。
生成器和 VAE 共用 SIR 与负二项形式，但当前 VAE 还包含神经修正并学习 k；
严格同模型对照应关闭修正并固定 k=10。当前网格脚本忽略修正，不能直接视为完整
hybrid 模型的精确后验。下列指标为历史记录，需统一模型、数据划分及评估方式复核。

### 两个基线模型

| checkpoint | 情形 | cover90 | 备注 |
|---|---|---|---|
| `results_hybrid/sir_final2_model.pt` | 数据符合 SIR | 89% | corr: β 0.968 / ρ 0.946 / R₀ 0.964 / α 0.771 |
| `results_hybrid/mis_fb_model.pt` | 误设（真 β 有干预） | 69% | 靠黑箱残差补，需 `--free_bits 0.5` |

### 复现

```bash
PY=/opt/anaconda3/envs/epidemic/bin/python

# 合模型基线（16 分钟 CPU）
$PY sir_hybrid_vae.py --n_train 16000 --epochs 200 --iwae 5 --patience 30 \
    --out results_hybrid/sir_final2

# 误设基线（8 分钟）
$PY sir_hybrid_vae.py --n_train 8000 --epochs 150 --iwae 5 --patience 25 \
    --misspec 1 --residual_mode beta --free_bits 0.5 --out results_hybrid/mis_fb

# 诊断（默认路径已指向上面两个 checkpoint）
$PY check_posterior_vs_grid.py
$PY inspect_patterns.py --mis_ckpt results_hybrid/mis_fb_model.pt
```

### 历史观察与待验证解释

1. **满协方差 q 是覆盖率的关键**。(β,α,ρ) 后验沿 R₀ 方向强相关，对角高斯给的是
   条件方差，比真后验窄 3.5 倍 → cover90 只有 6%。换 Cholesky 满协方差后直接到 78%。
   IWAE 是次要因素（单独用只能 6%→23%）。
2. **α 恢复较弱，原因待验证**：历史网格比较记录了标准化 α 的后验 sd 约 0.554、
   点估计相关约 0.77。这些数值不能证明理论上限，也不能排除实现或训练问题。
   先完成严格同模型对照，再区分数据的信息量与 VAE 推断误差。
3. **误设时必须开 `--free_bits 0.5 --anchor 1`**：否则 z_res 塌缩（黑箱只学一条固定
   平均曲线），且残差常数项与 β 共线 → β 偏差 −0.053、R₀ 覆盖率 1.8%。开了之后
   β 偏差 +0.010、R₀ RMSE 0.50→0.17。
4. **`--free_bits` 默认 0**：数据符合 SIR 时不该强迫黑箱编码噪声。先看
   `inspect_patterns.py` 图 3(a) 的修正幅度，确认真有误设再开。
5. 黑箱行为正常的判据：合模型数据上修正幅度 ~0.003（沉默），误设数据上 ~0.600
   （启动），相差百倍量级；还原的 log β(t) 与真干预曲线逐点相关 r≈0.96。

---

## 旧架构（conditional VAE）

`vae_full.py` + `simulate_data/`（年龄×家户 tau-leap 模拟器）+ `check_*.py` 诊断脚本。
基线结果 `vae_pop80k_*`。结论见 proposal 与诊断脚本注释；现阶段不再推进。

`archive/` 存历史产物：`old_vae_runs/`（旧 cVAE 各轮）、`prototypes/`（早期原型）、
`demo_figures/`、`hybrid_dev_runs/`（hybrid 架构的中间实验，共 29 个文件，
按 `sir_hybrid_* → sir_obs_* → sir_fullcov_* → sir_final_*` 的顺序反映调试过程）。
