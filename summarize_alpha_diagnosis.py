"""Render the frozen-model diagnosis; no model fitting or selection."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    out=Path('results_elbo/diagnosis')
    d=json.loads((out/'diagnosis.json').read_text())
    names={'matched_gaussian':'纯 SIR + 普通高斯','matched_curved':'纯 SIR + 可弯曲后验','hybrid_curved':'Hybrid + 可弯曲后验'}
    text=['# α 后验问题定位','',
          '## 结论','',
          '**当前版本不能再简单归因为“区间太窄”。** 在严格同模型的纯 SIR 对照中，可弯曲后验的区间宽度中位数已达到参考的 99.6%，但中心仍有误差。',
          '这里的“中心”是 log(α) 后验均值；比较参考后验不等于要求每次估计都等于生成真值。',
          '', '## 冻结权重后的比较','',
          '| 模型 | α RMSE | 90% 覆盖率 | 区间宽度/参考（中位数） | 中心误差（参考标准差单位，RMS） |',
          '|---|---:|---:|---:|---:|']
    for n,label in names.items():
        s=d['models'][n]['all']
        text.append(f"| {label} | {s['rmse']:.5f} | {s['coverage']:.1%} | {s['median_width_ratio']:.3f} | {s['center_error_in_reference_sd']:.3f} |")
    text += ['', '纯 SIR 数值参考在这 1,000 条曲线上的覆盖率为 88.5%。它不是完整 Hybrid 的精确后验；Hybrid 行只作辅助比较。',
             '', '## 区分中心和宽度','',
             '以下为纯 SIR 可弯曲后验的诊断性替换，**不进入训练或实际推断**：','',
             '- 原始覆盖率：86.8%。',
             '- 仅把 log(α) 中心换为参考中心，保留编码器标准差：88.1%。',
             '- 保留编码器中心，换用参考区间的宽度与不对称形状：87.1%。',
             '', '这支持优先调查中心误差；不能据此证明某一个训练因素是唯一原因，也不能用参考中心修补测试预测。',
             '', '## 是否只发生在信息少的曲线','',
             '| 观测病例总数分组（各 250 条） | 纯 SIR 可弯曲覆盖率 | 同组参考覆盖率 |',
             '|---|---:|---:|']
    for i in range(4):
        s=d['models']['matched_curved'][f'case_count_quartile_{i+1}']
        text.append(f"| 第 {i+1} 四分位（从少到多） | {s['coverage']:.1%} | {s['reference_coverage']:.1%} |")
    text += ['', '误差并不只集中在病例最少的一组。按最后 30 天是否占总病例 10% 以上分组，两组也都有差距。',
             '这个比例只是观测窗口末期活动的粗略指标，不等于确定疫情是否结束。分组结果是描述性诊断，不宜逐组追求恰好 90%。',
             '', '## 下一项单因素对照','',
             '只改变是否每轮生成新的训练曲线，保留结构、ELBO、训练批数、输入标准化和推断方式。',
             '同时跑继续训练固定数据的对照，防止混淆训练时长与数据更新。没有参数标签或后验教师。',
             '', '## 核查与边界','',
             '- 网格：751×751，标准化潜变量范围 ±6；最大边界概率质量小于 10⁻⁶。',
             '- 新实现的 α 后验均值与此前保存的网格均值最大差异小于 10⁻⁶。',
             '- 权重冻结，预测仍为一次编码器前向传播。诊断没有更改模型。',
             '- 种子 11103 此后属于开发数据；新改动必须使用新的最终测试集。',
             '- 两参数、已知 ρ/k 的诊断不能直接解释原三参数 Colab 的全部误差。']
    (out/'REPORT.md').write_text('\n'.join(text)+'\n')
    ref=np.load(out/'reference.npz')
    fig,axs=plt.subplots(1,3,figsize=(13,3.8))
    for name,label,color in [('matched_gaussian','SIR Gaussian','#8795a4'),('matched_curved','SIR curved','#167d9a')]:
        p=np.load(out/(name+'.npz'))
        axs[0].hist((p['hi']-p['lo'])/(ref['hi']-ref['lo']),bins=np.linspace(.5,1.5,45),histtype='step',lw=2,label=label,color=color)
    axs[0].axvline(1,color='black',ls='--',lw=1);axs[0].set(xlabel='Interval width / reference width',ylabel='Curves',title='Width is now close to reference');axs[0].legend(fontsize=8)
    p=np.load(out/'matched_curved.npz')
    axs[1].scatter(ref['mean'],p['mean'],s=6,alpha=.4,color='#167d9a');axs[1].plot([.1,.4],[.1,.4],'k--',lw=1)
    axs[1].set(xlabel='Reference posterior mean of alpha',ylabel='Encoder posterior mean of alpha',title='Some center error remains')
    groups=[d['models']['matched_curved'][f'case_count_quartile_{i+1}'] for i in range(4)]
    axs[2].plot(range(1,5),[s['coverage']*100 for s in groups],'o-',label='SIR curved',color='#167d9a')
    axs[2].plot(range(1,5),[s['reference_coverage']*100 for s in groups],'o--',label='SIR grid',color='#333333')
    axs[2].axhline(90,ls=':',color='gray');axs[2].set(xlabel='Observed case-count quartile (low to high)',ylabel='90% interval coverage (%)',title='Not confined to few-case curves',xticks=[1,2,3,4]);axs[2].legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'alpha_diagnosis.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
