"""Create a self-contained report of the Monte Carlo sample-count experiment."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path('results_elbo/sampling_ablation');out=root/'evaluation'
    d=json.loads((out/'evaluation.json').read_text());noise=json.loads((root/'sampling_noise.json').read_text())
    labels={'default':'当前默认 Hybrid','fresh':'上一轮新数据 Hybrid','mc8':'继续训练，8 次抽样','mc32':'继续训练，32 次抽样'}
    selected=d['selection']['selected']
    old=d['models']['default']['alpha'];new=d['models'][selected]['alpha']
    reduction=100*(1-new['rmse']/old['rmse'])
    text=['# 增加 ELBO 抽样数的对照','',
          f"验证集在生成测试数据前选择了：**{labels[selected]}**。本报告使用种子 13103 的 2,000 条新曲线。",'',
          f"所选版本相对当前默认版本的 α RMSE 变化为降低 {reduction:.2f}%；覆盖率变化为 {(new['cover90']-old['cover90'])*100:+.2f} 个百分点。",'',
          '**解释：相对默认模型有小幅改善，但不能全部归因于增加抽样。** 新版本还包含上一轮新数据训练和本轮继续训练。',
          '32 次与 8 次抽样的直接对照中，MSE 差异的 95% 配对区间包含零；本轮没有明确证明增加抽样能提高 α 点估计精度。',
          '验证集分数差也很小，不能据此声称 32 次抽样稳定优于 8 次。默认权重暂未替换。','',
          '## 为什么尝试这个改动','',
          '冻结上一轮模型，只用原训练集的两个批次，各重复 64 次随机抽样。检查 α 均值输出层梯度的波动，没有用真实参数或数值参考。','',
          '| 批次 | 8 次抽样梯度噪声 RMS | 32 次抽样梯度噪声 RMS | 32 / 8 |','|---|---:|---:|---:|']
    for batch,s in noise.items():
        a=s['8']['alpha_mean_gradient_noise_rms'];b=s['32']['alpha_mean_gradient_noise_rms']
        text.append(f'| {batch} | {a:.3f} | {b:.3f} | {b/a:.3f} |')
    text += ['', '训练信号的随机波动确实降低；这本身不证明泛化精度一定提高。',
             '', '## 对照设置','',
             '- 同一个起始权重、相同模型和训练集标准化。每轮 8,000 条新模拟观测，共 40 轮。',
             '- 新观测种子 30000–30039、批次顺序种子 40000–40039；两个分支的数据和批次一致。',
             '- 唯一改变是普通 ELBO 中的似然抽样数 8 / 32；不是 IWAE，不改变 KL 权重。',
             '- 32 次抽样用更多计算，不能声称它在相同算力预算下更好。',
             '- 验证保存权重用 64 次抽样；最终选择用五组随机流、每组 128 次抽样的观测 ELBO。',
             '- 仍只估计 β、α，ρ/k 已知。训练不给真实参数和参考后验；预测一次编码器前向，无逐曲线优化。',
             '', '## 新测试集 α 结果','',
             '| 版本 | 相关性 | RMSE | 90% 区间实际覆盖率 |','|---|---:|---:|---:|']
    for n in labels:
        s=d['models'][n]['alpha']
        text.append(f"| {labels[n]} | {s['corr']:.4f} | {s['rmse']:.5f} | {s['cover90']:.2%} |")
    ref=d['pure_sir_alpha_reference']
    text.append(f"| 纯 SIR 数值参考 | {ref['corr']:.4f} | {ref['rmse']:.5f} | {ref['cover90']:.2%} |")
    text += ['', '**纯 SIR 参考不包含神经网络修正，不是完整 Hybrid 的精确后验。** 它在同一病例生成模型下仍有非零误差；不能把恢复真值的全部误差都归咎于编码器。',
             '', '## 配对比较','',
             '同一测试样本配对自助抽样 4,000 次；以下区间没有包含重新训练的波动，不参与模型选择。','']
    for name,s in d['paired_comparisons'].items():
        lo,hi=s['mse_bootstrap95'];cl,ch=s['coverage_bootstrap95']
        text.append(f"- {name}：MSE 差异 {s['mse_difference']:.8f}，95% 区间 [{lo:.8f}, {hi:.8f}]；覆盖率差异 {100*s['coverage_difference']:+.2f} 个百分点，95% 区间 [{100*cl:+.2f}, {100*ch:+.2f}]。")
    text += ['', '## 验证集选择','', '| 候选 | 平均负 ELBO（越低越好） |', '|---|---:|']
    for n in labels:text.append(f"| {labels[n]} | {d['selection']['scores'][n]['mean']:.6f} |")
    text += ['', '## 核查与限制','',
             f"- 网格边界最大质量 {ref['max_boundary_mass']:.3g}；前 32 条曲线粗细网格 α 均值最大变化 {ref['first32_max_mean_change_vs_coarse']:.3g}。",
             '- 普通 ELBO 与后验变换的测试通过；公共推断接口只调用一次编码器，权重不变。',
             '- 仅一个训练随机种子、合成数据；没有验证真实病例或模型失配。',
             '- 不与原三参数 Colab 成绩直接比较。没有更改默认推断权重。',
             '', f"验证选中的候选权重：`{d['selection']['checkpoint']}`。"]
    (out/'REPORT.md').write_text('\n'.join(text)+'\n')
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    names=['default','fresh','mc8','mc32'];titles=['Default','Fresh data','Continue: 8 draws','Continue: 32 draws']
    axs[0].bar(titles,[d['models'][n]['alpha']['rmse'] for n in names],color=['#8795a4','#648fa8','#28939c','#166178'])
    axs[0].axhline(ref['rmse'],ls='--',color='black',label='Pure SIR reference');axs[0].set(ylabel='Alpha RMSE',title='Point estimation');axs[0].legend(fontsize=8)
    axs[1].bar(titles,[100*d['models'][n]['alpha']['cover90'] for n in names],color=['#8795a4','#648fa8','#28939c','#166178'])
    axs[1].axhline(90,ls=':',color='black',label='Nominal 90%');axs[1].set(ylim=(0,100),ylabel='Coverage (%)',title='90% posterior intervals');axs[1].legend(fontsize=8)
    for ax in axs:ax.tick_params(axis='x',labelrotation=20)
    fig.tight_layout();fig.savefig(out/'alpha_sampling.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
