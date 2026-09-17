"""Report final untouched confirmation results, without changing models."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
p=Path('results_elbo/confirmation');r=json.loads((p/'evaluation.json').read_text());selection=r['selection']
labels={'previous_gaussian':'第一阶段高斯 Hybrid','selected_hybrid':'验证集选定的 Hybrid','curved_matched':'可弯曲后验 + 纯 SIR'}
grid=r['constant_sir_grid'][-1]
md='''# 只用 ELBO 的最终确认实验

## 没有改动的训练原则

- 训练只用病例曲线、先验和 ELBO，不输入真实 β/α 或数值参考后验。
- 保留神经网络修正 β(t) 后进入 SIR 的 Hybrid decoder。
- 物理潜空间只包含 β、α；已知 ρ=0.10、k=10。4 维残差潜变量只用于神经修正。
- 测试时只调用一次编码器，不做逐曲线优化、不更新权重。必要的后验抽样只用于汇总分布。

## 排查得到的证据

首先检查了 SIR/负二项/KL 实现及导数，没有发现能解释旧结果的明显公式错误。
第一阶段用更充分的观测数据训练，参数点估计改善，但在种子 9103 的曲线上发现区间仍偏窄。
因此该批数据已转为开发记录，不能继续作为独立最终证据。

第二阶段保持 decoder 和 ELBO 不变，比较：

1. 在原二维高斯后验上增加可逆的弯曲变换，表示 β、α 之间非线性的依赖。
2. 相同继续训练预算，保持高斯后验。
3. 保持高斯，并添加由观测值计算的累计病例特征。

所有训练都不使用参数标签或后验教师。弯曲变换的 Jacobian 行列式为 1，KL 有解析表达式；
已通过逆变换、Jacobian、独立 Monte Carlo 密度计算以及梯度测试。

## 验证集选择

'''
md+=f"选择结果：`{selection['selected']}`。只按验证 ELBO 选择，先保存选择记录，再生成最终测试曲线。\n\n| 候选 | 验证负 ELBO（越小越好） | 继续训练轮次 |\n|---|---:|---:|\n"
for name,a in selection['candidates'].items():md+=f"| {name} | {a['val_loss']:.5f} | {a['epoch']} |\n"
md+='''
## 新保留数据上的 α 结果

训练 8,000 条（7101），验证 500 条（7102）；最终确认 1,000 条（10103），与第一阶段测试种子不同。
初始阶段 160 轮；第二阶段候选各继续 100 轮，按验证损失保留权重。第二阶段每条曲线用 8 个正负成对随机样本估计 ELBO，验证使用 32 样本。

| 方法 | 相关系数 | RMSE | 偏差 | 90% 区间覆盖率 |
|---|---:|---:|---:|---:|
'''
for name in labels:
 a=r['models'][name]['metrics']['alpha'];md+=f"| {labels[name]} | {a['corr']:.3f} | {a['rmse']:.4f} | {a['bias']:+.4f} | {a['cover90']:.1%} |\n"
a=grid['metrics']['alpha'];md+=f"| 纯 SIR 网格参考 | {a['corr']:.3f} | {a['rmse']:.4f} | {a['bias']:+.4f} | {a['cover90']:.1%} |\n"
md+='\n## 选定 Hybrid 的全部指标\n\n| 参数 | 相关系数 | RMSE | 偏差 | 90% 覆盖率 |\n|---|---:|---:|---:|---:|\n'
for name,a in r['models']['selected_hybrid']['metrics'].items():md+=f"| {name} | {a['corr']:.3f} | {a['rmse']:.4f} | {a['bias']:+.4f} | {a['cover90']:.1%} |\n"
md+=f"\nβ/α 联合 90% 区域覆盖率：{r['models']['selected_hybrid']['joint_cover90']:.1%}。\n"
md+='''
## 不能过度解释的地方

- 近似后验不等于精确后验。相关性提高但覆盖率不足时，不能称为彻底解决。
- 本轮固定已知 ρ、k，不能直接推广到原三参数 Colab、未知报告倍数或真实疫情。
- 数值网格没有神经残差，只是纯 SIR 的参考，不是 Hybrid 的完整后验。
- 第一阶段比较同时涉及编码器结构、输入和初始化；不能把全部改善归因于其中一项。
- 第一阶段另一个随机初始化的 Hybrid 已记录在 `results_elbo/final/`；第二阶段最终候选尚未做多训练种子重复。
- 本轮没有干预/模型误设测试，不能宣称神经网络已经能稳健处理真实数据中的偏离。

## 数值参考稳定性

'''
for g in r['constant_sir_grid']:
 md+=f"- {g['size']}×{g['size']}、范围 ±{g['span']}，最大边界质量 {g['max_boundary_mass']:.3g}"
 if 'max_mean_change' in g:md+=f"，相对前一网格最大均值变化 {g['max_mean_change']:.3g}"
 md+='。\n'
md+='''
## 文件与复现

- `train_elbo_only.py`：第一阶段普通 ELBO 训练。
- `train_curved_elbo.py`：第二阶段后验形状/同预算训练对照。
- `infer_elbo_only.py`：统一的一次编码器推断入口。
- `evaluate_elbo_confirm.py`：仅按验证结果选择，再运行新保留集。
- `tests/check_elbo_only.py`、`tests/check_curved_elbo.py`：公式与推断行为测试。
- `results_elbo/confirmation/selection.json`：测试前的选择记录。
- `results_elbo/confirmation/evaluation.json`：完整指标。

```bash
/opt/anaconda3/envs/epidemic/bin/python train_curved_elbo.py
/opt/anaconda3/envs/epidemic/bin/python train_curved_elbo.py --checkpoint results_elbo/matched/model.pt --out results_elbo/curved_matched
/opt/anaconda3/envs/epidemic/bin/python train_curved_elbo.py --freeze_curve --out results_elbo/continued_gaussian
/opt/anaconda3/envs/epidemic/bin/python train_curved_elbo.py --cumulative --freeze_curve --out results_elbo/cumulative_gaussian
/opt/anaconda3/envs/epidemic/bin/python evaluate_elbo_confirm.py
/opt/anaconda3/envs/epidemic/bin/python summarize_elbo_confirm.py
```

先完成第一阶段训练，才能运行上述继续训练命令。修改后若再根据此最终保留集调整方法，需换新测试集。
原 Colab 基准保留；后验教师与逐曲线优化的探索不用于本轮成绩。本轮尚未推送 GitHub。

![Final alpha comparison](alpha_confirmation.png)
'''
(p/'REPORT.md').write_text(md)
fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
a=np.load(p/'previous_gaussian.npz');b=np.load(p/'selected_hybrid.npz');gm=np.load(p/'grid_mean.npy')
truth=a['truth'][:,1];series=[a['mean'][:,1],b['mean'][:,1],gm[:,1]]
titles=['Previous Gaussian Hybrid','Selected Hybrid (ELBO only)','Constant-SIR grid reference']
lo=min(truth.min(),*(x.min() for x in series));hi=max(truth.max(),*(x.max() for x in series))
for ax,x,title in zip(axes,series,titles):
 ax.scatter(truth,x,s=9,alpha=.45,color='#267887',edgecolors='none');ax.plot([lo,hi],[lo,hi],'--',color='#888')
 ax.set(xlim=(lo,hi),ylim=(lo,hi),xlabel='True alpha (per day)',ylabel='Posterior mean alpha (per day)')
 ax.set_aspect('equal');ax.set_title(title+f'\nr = {np.corrcoef(x,truth)[0,1]:.3f}',fontsize=10)
fig.suptitle('New holdout: 1,000 outbreaks; one-pass encoder, no parameter labels in training',fontsize=12)
fig.savefig(p/'alpha_confirmation.png',dpi=150);plt.close(fig)
print(p/'REPORT.md')
