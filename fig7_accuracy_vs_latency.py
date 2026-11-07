"""fig7_accuracy_vs_latency.py - Figure 7 of the paper.

Top-1 accuracy versus Raspberry Pi 3B+ latency (TFLite FP32 filled, INT8 hollow).
Data: latency from the repeated Pi sessions; FP32 accuracy from training, INT8 accuracy on the x86 host; means of 3 seeds.
The values below are the measured results reported in the paper; edit them if
you re-run the experiments.

Usage:  python fig7_accuracy_vs_latency.py [output_folder]     (default: figures/out)
Writes: fig_pareto.pdf and fig_pareto.png
Needs:  matplotlib, numpy
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, os, sys
from matplotlib.lines import Line2D
OUT=sys.argv[1] if len(sys.argv)>1 else os.path.join(os.path.dirname(os.path.abspath(__file__)),'out')
os.makedirs(OUT,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.titlesize':9,'axes.labelsize':8.5,
  'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,
  'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#c3c2b7','axes.linewidth':0.8,
  'xtick.color':'#52514e','ytick.color':'#52514e','axes.labelcolor':'#0b0b0b','text.color':'#0b0b0b',
  'grid.color':'#e1e0d9','grid.linewidth':0.6,'savefig.bbox':'tight','savefig.pad_inches':0.03})
C={'Gated (ours)':'#eb6834','DepGraph':'#2a78d6','Unpruned':'#1baf7a'}
INK2='#52514e'; REF='#898781'
W=6.9
def save(fig,name):
    fig.savefig(os.path.join(OUT,name+'.pdf')); fig.savefig(os.path.join(OUT,name+'.png'),dpi=200); plt.close(fig)

DS=['CIFAR-10','CIFAR-100','Tiny-ImageNet-200']

P=[('CIFAR-10','Unpruned','FP32',58.7,96.09),('CIFAR-10','Unpruned','INT8',110.4,95.89),('CIFAR-10','Gated (ours)','FP32',33.9,94.74),('CIFAR-10','Gated (ours)','INT8',76.0,94.41),('CIFAR-10','DepGraph','FP32',35.3,93.66),('CIFAR-10','DepGraph','INT8',86.9,93.36),
   ('CIFAR-100','Unpruned','FP32',55.6,82.31),('CIFAR-100','Unpruned','INT8',109.2,80.50),('CIFAR-100','Gated (ours)','FP32',30.8,78.09),('CIFAR-100','Gated (ours)','INT8',75.8,73.90),('CIFAR-100','DepGraph','FP32',34.3,72.98),('CIFAR-100','DepGraph','INT8',86.0,66.05),
   ('Tiny-ImageNet-200','Unpruned','FP32',55.4,72.29),('Tiny-ImageNet-200','Unpruned','INT8',108.7,69.86),('Tiny-ImageNet-200','Gated (ours)','FP32',31.7,66.51),('Tiny-ImageNet-200','Gated (ours)','INT8',75.8,61.20),('Tiny-ImageNet-200','DepGraph','FP32',34.0,63.81),('Tiny-ImageNet-200','DepGraph','INT8',85.6,56.80)]
fig,axs=plt.subplots(1,3,figsize=(W,2.6))
for ax,ds in zip(axs,DS):
    g=[p for p in P if p[0]==ds]
    for _,m,rt,lat,acc in g:
        ax.scatter(lat,acc,s=34,facecolor=C[m] if rt=='FP32' else 'white',edgecolor=C[m],linewidth=1.6,zorder=3)
    a=[p[4] for p in g]; ax.set_ylim(np.floor(min(a))-1,np.ceil(max(a))+1)
    ax.set_xlim(20,120); ax.set_xticks([20,70,120]); ax.set_title(ds,fontweight='bold',loc='left')
    ax.grid(True); ax.set_axisbelow(True)
axs[0].set_ylabel('Top-1 (%)'); axs[1].set_xlabel('Latency on Raspberry Pi 3B+ (ms, batch 1)')
h=[Line2D([],[],marker='o',ls='',color=C[m],label=m) for m in ['Unpruned','Gated (ours)','DepGraph']]+\
  [Line2D([],[],marker='o',ls='',markerfacecolor='#52514e',markeredgecolor='#52514e',label='TFLite FP32'),Line2D([],[],marker='o',ls='',markerfacecolor='white',markeredgecolor='#52514e',label='TFLite INT8')]
fig.legend(handles=h,frameon=False,loc='upper center',ncol=5,bbox_to_anchor=(0.5,1.08))
fig.tight_layout(w_pad=1.2)
save(fig,'fig_pareto')
print('wrote', os.path.join(OUT, 'fig_pareto') + '.pdf / .png')
