"""fig5_gate_trajectory.py - Figure 5 of the paper.

MACs of the network with binarised gates after each phase-A epoch and after export.
Data: hard-gate MACs in the training logs (CIFAR-10 seed 0; CIFAR-100 and Tiny-ImageNet-200 seeds 0-2, mean).
The values below are the measured results reported in the paper; edit them if
you re-run the experiments.

Usage:  python fig5_gate_trajectory.py [output_folder]     (default: figures/out)
Writes: fig_gate_trajectory.pdf and fig_gate_trajectory.png
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

traj={'CIFAR-10 (seed 0)':[100,100,100,100,97.6,90.2,83.0,76.7,71.1,65.5,62.0,59.7,58.3,57.9,57.7,49.98],
 'CIFAR-100 (mean of 3 seeds)':[100,100,100,99.6,96.9,91.4,85.8,81.4,77.5,74.0,70.1,67.0,64.5,62.3,59.7,49.94],
 'Tiny-ImageNet-200 (mean of 3 seeds)':[100,100,100,99.6,96.2,89.9,86.4,83.3,79.2,76.3,73.6,70.2,67.5,64.9,62.7,49.96]}
TC={'CIFAR-10 (seed 0)':'#6250d6','CIFAR-100 (mean of 3 seeds)':'#e87ba4','Tiny-ImageNet-200 (mean of 3 seeds)':'#008300'}
MK={'CIFAR-10 (seed 0)':'o','CIFAR-100 (mean of 3 seeds)':'s','Tiny-ImageNet-200 (mean of 3 seeds)':'^'}
fig,ax=plt.subplots(figsize=(W,3.0))
ep=np.arange(1,17)
for k,v in traj.items():
    ax.plot(ep[:15],v[:15],color=TC[k],lw=1.8,marker=MK[k],ms=3.6,label=k,zorder=3)
    ax.plot(ep[14:],v[14:],color=TC[k],lw=1.4,ls=(0,(3,2)),marker=MK[k],ms=3.6,zorder=3)
ax.axhline(50,color=REF,lw=1.1,ls=(0,(5,4)))
ax.text(1.1,50.8,'budget: 50% of the MACs',fontsize=7.5,color=INK2)
ax.annotate('projection + rounding\nto multiples of 8',xy=(15.5,54.5),xytext=(12.2,51.6),fontsize=7.5,color=INK2,ha='center',
            arrowprops=dict(arrowstyle='-',color=REF,lw=0.8))
ax.set_xticks([1,5,10,15,16]); ax.set_xticklabels(['1','5','10','15','export'])
ax.set_xlim(0.6,16.4); ax.set_ylim(45,102)
ax.set_yticks([50,60,70,80,90,100]); ax.set_yticklabels([f'{t}%' for t in [50,60,70,80,90,100]])
ax.yaxis.grid(True); ax.set_axisbelow(True)
ax.set_xlabel('Phase-A epoch (gate training), then export')
ax.set_ylabel('MACs with binarised gates\n(% of the unpruned model)')
ax.legend(frameon=False,loc='upper right')
save(fig,'fig_gate_trajectory')
print('wrote', os.path.join(OUT, 'fig_gate_trajectory') + '.pdf / .png')
