# Hybrid SIR-VAE 项目


## 当前结果：只用观测数据和 ELBO，仍有欠覆盖

最新完成[训练抽样数对照](results_elbo/sampling_ablation/evaluation/REPORT.md)：
在新一批 2,000 条曲线上，默认 Hybrid 的 α RMSE 为 0.01909、覆盖率为 88.15%；
继续训练的 8 次抽样版本为 0.01898、88.40%，32 次抽样版本为 0.01896、88.75%。
增加抽样确实降低了训练梯度波动，但相对 8 次抽样的精度收益很小；不能称为彻底解决。
验证集选中 32 次抽样候选，权重单独保存在 `results_elbo/sampling_ablation/mc32/model.pt`。
默认推断权重仍未替换，全部训练继续只用观测 ELBO。

上一轮完成[每轮新训练曲线的单因素对照](results_elbo/fresh_ablation/evaluation/REPORT.md)：
在另外 2,000 条新曲线上，原 Hybrid 的 α RMSE 为 0.01923、90% 覆盖率为 88.30%；
验证集选中的新数据训练版为 0.01912、88.35%。同预算继续使用固定数据的对照为
0.01918、88.70%。新数据略微改善点估计，没有改善覆盖率，**并未彻底解决**。
实验候选保存在 `results_elbo/fresh_ablation/hybrid_fresh/model.pt`，可显式传入
`infer_elbo_only.py --checkpoint`；这次没有替换默认权重，也没有改动模型结构。

当前推断入口是 `infer_elbo_only.py`，默认读取验证集选定的
`results_elbo/curved_hybrid/model.pt`。只估计 β、α，固定已知 ρ=0.10、k=10；
保留神经网络 + SIR decoder。训练仅用 ELBO，测试只运行一次编码器，
不提供真实参数/参考后验，也不做逐曲线优化。

第三批全新 1,000 条测试曲线上，α 相关性 **0.881**、RMSE **0.0192**、
90% 区间覆盖率 **86.9%**。点估计有改善，但后验仍欠覆盖，**不能称为彻底解决**。
参考 [最终完整报告](results_elbo/final_holdout/REPORT.md) 和
[实验约束与数据隔离](results_elbo/PROTOCOL.md)。

训练入口：`train_elbo_only.py`（基础训练）、`train_curved_elbo.py`（后验形状对照）；
`train_mixture_elbo.py` 是未被验证集选择的混合后验候选。阶段模型和报告均已保留，
前两批测试转为开发记录，最终选择在第三批测试生成之前固定。
纯 SIR 数值参考在同一最终数据上的 α 相关性为 0.893、RMSE 为 0.0183、覆盖率为 88.5%；
它不包含神经残差，不能称为完整 Hybrid 的精确后验。

进一步的[冻结模型诊断](results_elbo/diagnosis/REPORT.md)显示，纯 SIR 可弯曲后验的
区间宽度中位数已达到参考的 99.6%，仍有中心误差。因此欠覆盖不能直接等同于
“区间太窄”。第三批数据现用于开发诊断；后续训练更新的评价另用新保留数据。

此前尝试过后验指导训练和逐曲线优化，这些探索保留在本地，不包含在本次上传的
ELBO 方案中；它们的成绩不能用作当前模型的成绩。下面的三参数 Colab 对齐
版本保留作为历史基准，不与最新两参数实验混用。

## 原三参数 Colab 对齐基准

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

## Project background

Earlier work explored parameter inference for a stochastic household SIR model, using a household epidemic simulator, an ABC-MCMC baseline, and a VAE trained on simulated data. The current aligned experiment above uses a homogeneous SIR mechanism.

Tools: Python, PyTorch, NumPy, Pandas.

## 两参数模型的使用与复现

需要 Python、PyTorch、NumPy、Matplotlib。从仓库根目录运行。
输入 `counts.npy` 是一条 `(120,)` 或多条 `(N, 120)` 非负每日病例数。

```bash
# 当前默认模型：一次编码器推断
python infer_elbo_only.py counts.npy

# 显式使用最新实验候选（默认模型尚未替换）
python infer_elbo_only.py counts.npy --checkpoint results_elbo/sampling_ablation/mc32/model.pt

# 验证普通 ELBO、后验变换以及单次编码器推断
python tests/check_elbo_only.py
python tests/check_curved_elbo.py
python tests/check_mixture_elbo.py
```

`experiment_alpha.py` 提供两参数模拟器和纯 SIR 网格诊断辅助函数。
训练路线依次为 `train_elbo_only.py`、`train_curved_elbo.py`、
`train_fresh_elbo.py`、`train_sampling_elbo.py`；具体参数与数据种子见
[实验记录](results_elbo/PROTOCOL.md)。保存的权重、指标、合成测试数据和报告位于
`results_elbo/`。历史测试集一旦用于后续诊断，会标为开发数据；重新运行评价脚本
不等于获得一批未使用过的新测试数据。
