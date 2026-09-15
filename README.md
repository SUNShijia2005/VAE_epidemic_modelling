# Hybrid SIR-VAE：统一实验

## 当前入口

- `sir_hybrid_vae.py`：训练、评估和保存模型。
- `Hybrid_SIR_VAE_report.ipynb`：可独立运行的 Colab，同一模型和主实验配置。
- `SIR_VAE_report_revised.ipynb`：配套项目说明。
- `tests/check_colab_alignment.py`：与保存的原 Colab 比较模拟器、decoder、ELBO 和梯度。

## 模型与参数

模型依据有代码的 Colab：encoder 近似推断 β、α、ρ 的联合后验；decoder 先令
β(t)=β·exp(0.5 f(t))，再求解 SIR。f(0)=0，残差潜变量为 4 维，物理后验为满协方差高斯。

| 项目 | 统一设置 |
|---|---|
| 人口 / 初始感染 / 观测天数 | 100,000 / 20 / 120 |
| 训练 / 验证 / 测试 | 16,000 / 2,000 / 500（项目报告的数据规模） |
| β、α、ρ 先验中位数 | 0.35、0.20、0.10 |
| 对数先验标准差 | 0.25、0.20、0.30 |
| 观测 | 负二项；生成时 k=10，推断时学习全局 k |
| 目标 / 轮数 | 普通 ELBO / 120 |
| batch / 学习率 | 128 / 0.002，余弦衰减 |
| KL 预热 / 早停 | 关闭 / 关闭，使用最后一轮权重 |
| 种子 / 观测种子偏移 | 0 / 999，与原 Colab 一致 |
| 参数后验抽样 | 400 次 |

ρ 是正的报告倍数，当前对数正态先验没有把它限制在 1 以下。
参数点估计是后验样本均值；重建图是将潜变量均值送入 decoder 的拟合曲线。

## 运行

```bash
/opt/anaconda3/envs/epidemic/bin/python sir_hybrid_vae.py
```

新文件前缀为 `results_hybrid/colab_aligned`；输出模型、图和 `_metrics.json`，后者包含完整配置及每个参数的指标。默认不会使用历史模型继续训练。

复现原 Colab 主实验的数据规模：

```bash
/opt/anaconda3/envs/epidemic/bin/python sir_hybrid_vae.py --n_train 8000 --n_val 600 --n_test 600 --out results_hybrid/colab_original
```

Notebook 中的干预强度扫描是独立实验：每组 3,000/400/400 条曲线、50 轮，保留原 Colab 设置。

## 结果状态与历史

统一配置尚未完成全量训练；Notebook 已清除旧输出和不能沿用的结果结论。
历史 `sir_final2` 是观测曲线修正 0.3、IWAE=5 的实验，其相关性 0.968/0.771/0.946/0.964
不能作为新配置的结果。`results_hybrid/` 中历史权重和图保留；修改前代码、说明和 Notebook
位于 `archive/pre_colab_alignment/`。桌面原始两份 Colab 未修改。

## 后验诊断的边界

生成器没有神经修正且固定 k=10，因此当前 hybrid 还不是严格同模型对照。
`check_posterior_vs_grid.py` 的网格忽略神经修正，仅作 constant-SIR 参考；默认拒绝
把启用残差的 checkpoint 当作完整模型的精确后验。若明确研究这个不同模型的参考，
可加 `--allow_reduced_reference`，并自行检查网格分辨率与范围收敛。
α 恢复较弱的原因尚未确定，不能宣称已达到理论上限。

## 其他文件

`vae_full.py` 和 `simulate_data/` 属于旧 conditional VAE 项目，没有在本次调整。
历史 checkpoint 的诊断加载保留旧观测种子偏移 12345 和旧验证集划分，避免用新规则重建旧数据。
