"""Report the preregistered fresh-observation ablation without changing selection."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    out=Path('results_elbo/fresh_ablation/evaluation')
    d=json.loads((out/'evaluation.json').read_text())
    labels={'hybrid_original':'Hybrid 原版','hybrid_fixed':'Hybrid 继续训练固定数据',
            'hybrid_fresh':'Hybrid 每轮新数据','matched_original':'纯 SIR 原版',
            'matched_fixed':'纯 SIR 继续训练固定数据','matched_fresh':'纯 SIR 每轮新数据'}
    selected=d['selection']['selected']
    text=['# 每轮更新训练曲线的单因素对照','',
          f"验证集预先选定：**{labels[selected]}**。下表为随后生成的 2,000 条新曲线（种子 12103）。",
          '', '**本轮结论：更新训练曲线略微改善中心估计，但没有改善区间覆盖率，不能称为彻底解决。**',
          '候选权重单独保留，默认推断权重未替换。训练没有使用参数答案或参考后验。',
          '', '## 改动与约束','',
          '- 唯一对照因素：继续使用原 8,000 条曲线，或每轮生成 8,000 条新的曲线。',
          '- 相同起始权重、结构、标准化、优化器计划、60 轮训练、8 样本普通 ELBO。',
          '- 两个物理未知参数仍是 β、α；ρ=0.10、k=10 为已知常数。',
          '- 训练只接收观测曲线。真实参数只用于最终评价；网格参考只用于训练结束后的诊断。',
          '- 测试仍然一次编码器前向传播，不做逐曲线优化，不调整区间宽度。',
          '- Hybrid 保留神经网络 + SIR decoder；纯 SIR 是诊断对照。',
          '', '## 新测试集的 α 结果','',
          '| 模型 | 相关性 | RMSE | 90% 覆盖率 |','|---|---:|---:|---:|']
    for n in labels:
        s=d['models'][n]['alpha']
        text.append(f"| {labels[n]} | {s['corr']:.4f} | {s['rmse']:.5f} | {s['cover90']:.2%} |")
    s=d['pure_sir_alpha_reference']
    text.append(f"| 纯 SIR 数值参考 | {s['corr']:.4f} | {s['rmse']:.5f} | {s['cover90']:.2%} |")
    text += ['', '数值参考只与纯 SIR 是相同模型；不是完整 Hybrid 的精确后验。',
             '', '## 新数据相对固定数据的差异','',
             '同一测试曲线配对自助抽样 4,000 次，95% 区间仅描述测试差异的波动，不参与模型选择。',
             'MSE 差异小于零表示每轮新数据的点估计更好；覆盖率差异以百分点表示。','']
    for kind in ['matched','hybrid']:
        s=d['fresh_minus_fixed'][kind];lo,hi=s['mse_difference_bootstrap95'];cl,ch=s['coverage_difference_bootstrap95']
        text.append(f"- {kind}：MSE 差异 {s['mse_difference']:.7f}，95% 区间 [{lo:.7f}, {hi:.7f}]；覆盖率差异 {s['coverage_difference']*100:+.2f} 个百分点，95% 区间 [{cl*100:+.2f}, {ch*100:+.2f}]。")
    text += ['', '## 后验中心的辅助检查','',
             '以下为编码器 α 后验均值与数值参考均值之差，按参考 α 标准差归一化后的 RMS。越小越接近参考中心。','']
    ref=np.load(out/'alpha_reference.npz')
    center={}
    for name in labels:
        p=np.load(out/(name+'.npz'))
        center[name]=float(np.sqrt(np.mean(((p['mean'][:,1]-ref['mean'])/ref['sd'])**2)))
        text.append(f'- {labels[name]}：{center[name]:.3f}。')
    d['reference_center_error_rms']=center
    (out/'evaluation.json').write_text(json.dumps(d,indent=2))
    text += ['', '## 验证集选择依据','', '| 候选 | 五组随机流的平均负 ELBO（越低越好） |', '|---|---:|']
    for n in ['hybrid_original','hybrid_fixed','hybrid_fresh']:
        text.append(f"| {labels[n]} | {d['selection']['scores'][n]['mean']:.6f} |")
    text += ['', '## 解释边界','',
             '- 这是一次训练种子的对照；配对区间没有包含重新训练带来的波动。',
             '- 换新曲线同时增加了独立训练样本数量，不能声称仅改变采样顺序就有效。',
             '- 全部数据仍为固定参数 SIR 合成病例；没有验证真实数据或干预情形。',
             '- 即使数值参考也不能从每条有噪声曲线精确恢复生成参数；准确后验也有非零估计误差。',
             '- 不把这次两参数结果冒充原三参数 Colab 结果。',
             '', '详细记录：`selection.json`、`evaluation.json`；诊断前提见 `../../diagnosis/REPORT.md`。']
    (out/'REPORT.md').write_text('\n'.join(text)+'\n')
    fig,axs=plt.subplots(1,3,figsize=(12,3.8))
    allvalues=np.concatenate([np.load(out/(name+'.npz'))[field][:,1] for name in ['hybrid_original','hybrid_fixed','hybrid_fresh'] for field in ['truth','mean']])
    limits=(float(allvalues.min())-.01,float(allvalues.max())+.01)
    for ax,name,title in zip(axs,['hybrid_original','hybrid_fixed','hybrid_fresh'],['Original Hybrid','Fixed data, continued','Fresh data each epoch']):
        p=np.load(out/(name+'.npz'));m=d['models'][name]['alpha']
        ax.scatter(p['truth'][:,1],p['mean'][:,1],s=5,alpha=.3,color='#167d9a')
        ax.plot(limits,limits,'k--',lw=1)
        ax.set(xlabel='Generating alpha (evaluation only)',ylabel='Encoder posterior mean',title=f"{title}\nRMSE {m['rmse']:.5f} | coverage {m['cover90']:.2%}",xlim=limits,ylim=limits)
    fig.tight_layout();fig.savefig(out/'alpha_fresh_comparison.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
