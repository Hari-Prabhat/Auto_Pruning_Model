"""fig6_pi_speedup.py - Figure 6 of the paper.

Speed-up over the unpruned model on the Raspberry Pi 3B+, per runtime and dataset.
Data: raspberry_pi/run_pi_latency_reps.sh: 270 sessions; paired per-session ratio, median of 5 sessions, mean of 3 seeds.
The values below are the measured results reported in the paper; edit them if
you re-run the experiments.

Usage:  python fig6_pi_speedup.py [output_folder]     (default: figures/out)
Writes: fig_speedup.pdf and fig_speedup.png
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

sp={'ONNX Runtime':{'CIFAR-10':(1.61,1.53),'CIFAR-100':(1.65,1.54),'Tiny-ImageNet-200':(1.59,1.52)},
    'TFLite FP32':{'CIFAR-10':(1.73,1.64),'CIFAR-100':(1.83,1.61),'Tiny-ImageNet-200':(1.74,1.63)},
    'TFLite INT8':{'CIFAR-10':(1.46,1.27),'CIFAR-100':(1.44,1.27),'Tiny-ImageNet-200':(1.44,1.27)}}
fig,ax=plt.subplots(figsize=(W,3.4))
y=0; yt=[]; yl=[]
for rt,dd in sp.items():
    ax.text(1.012,y+0.45,rt,fontweight='bold',ha='left',va='bottom')
    for ds,(g,dg) in dd.items():
        ax.plot([dg,g],[y,y],color='#e1e0d9',lw=3,zorder=2,solid_capstyle='round')
        ax.scatter([dg],[y],s=34,color=C['DepGraph'],zorder=3,edgecolor='white',linewidth=0.6)
        ax.scatter([g],[y],s=34,color=C['Gated (ours)'],zorder=3,edgecolor='white',linewidth=0.6)
        ax.text(dg-0.018,y,f'{dg:.2f}×',ha='right',va='center',fontsize=7.2,color=INK2)
        ax.text(g+0.018,y,f'{g:.2f}×',ha='left',va='center',fontsize=7.2)
        yt.append(y); yl.append(ds); y-=0.8
    y-=1.0
ax.axvline(1.0,color='#c3c2b7',lw=1); ax.axvline(2.0,color=REF,lw=1,ls=(0,(5,4)))
ax.text(1.0,1.25,'1.0× = unpruned',fontsize=7.5,color=INK2,va='bottom',ha='center')
ax.text(2.0,1.25,'2.0× = MAC reduction',fontsize=7.5,color=INK2,ha='center',va='bottom')
ax.set_yticks(yt); ax.set_yticklabels(yl); ax.tick_params(axis='y',length=0); ax.spines['left'].set_visible(False)
ax.set_xlim(0.9,2.1); ax.set_ylim(y+1.2,1.7); ax.xaxis.grid(True); ax.set_axisbelow(True)
ax.set_xlabel('Speed-up over the unpruned model on Raspberry Pi 3B+ (batch 1; median of 5 sessions, mean of 3 seeds)')
h=[Line2D([],[],marker='o',ls='',color=C['Gated (ours)'],label='Gated (ours)'),Line2D([],[],marker='o',ls='',color=C['DepGraph'],label='DepGraph')]
ax.legend(handles=h,frameon=False,loc='upper left',bbox_to_anchor=(0.0,1.12),ncol=2)
save(fig,'fig_speedup')
print('wrote', os.path.join(OUT, 'fig_speedup') + '.pdf / .png')
