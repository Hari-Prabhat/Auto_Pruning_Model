"""fig8_throughput.py - Figure 8 of the paper.

Sustained throughput on the Raspberry Pi 3B+ (images per second, 60 s, batch 1).
Data: raspberry_pi/run_pi_bench.sh, mean over the 8 packages measured with cool-down.
The values below are the measured results reported in the paper; edit them if
you re-run the experiments.

Usage:  python fig8_throughput.py [output_folder]     (default: figures/out)
Writes: fig_throughput.pdf and fig_throughput.png
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

th={'ONNX Runtime':(10.3,16.9,15.6),'TFLite FP32':(16.6,29.1,26.7),'TFLite INT8':(8.7,12.3,10.9)}
fig,ax=plt.subplots(figsize=(W,2.7))
ms=['Unpruned','Gated (ours)','DepGraph']; bw=0.26
for i,(rt,v) in enumerate(th.items()):
    for j,m in enumerate(ms):
        x=i+(j-1)*(bw+0.02)
        ax.bar(x,v[j],width=bw,color=C[m],zorder=3,label=m if i==0 else None)
        ax.text(x,v[j]+0.5,f'{v[j]:.1f}',ha='center',va='bottom',fontsize=7.5)
ax.set_xticks(range(3)); ax.set_xticklabels(list(th.keys())); ax.tick_params(axis='x',length=0)
ax.set_ylabel('Images per second'); ax.set_ylim(0,33); ax.yaxis.grid(True); ax.set_axisbelow(True)
ax.legend(frameon=False,ncol=3,loc='lower left',bbox_to_anchor=(0,1.0))
save(fig,'fig_throughput')
print('wrote', os.path.join(OUT, 'fig_throughput') + '.pdf / .png')
