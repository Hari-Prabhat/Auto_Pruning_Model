"""fig3_accuracy_change.py - Figure 3 of the paper.

Top-1 change versus the unpruned model at 50% MACs (dots: seeds 0-2, bar: mean).
Data: full-test top-1 of every training run (log.json, edgenext_bench.py summary); CIFAR-10 seeds 1-2 re-evaluated from the released checkpoints.
The values below are the measured results reported in the paper; edit them if
you re-run the experiments.

Usage:  python fig3_accuracy_change.py [output_folder]     (default: figures/out)
Writes: fig_acc_change.pdf and fig_acc_change.png
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

d={('CIFAR-10','Gated (ours)'):[-1.39,-1.52,-1.12],('CIFAR-10','DepGraph'):[-2.50,-2.04,-2.75],
   ('CIFAR-100','Gated (ours)'):[-4.14,-4.35,-4.17],('CIFAR-100','DepGraph'):[-10.23,-6.90,-10.84],
   ('Tiny-ImageNet-200','Gated (ours)'):[-5.95,-5.70,-5.71],('Tiny-ImageNet-200','DepGraph'):[-9.06,-7.89,-8.49]}
fig,ax=plt.subplots(figsize=(W,3.3))
y=0; yt=[]; yl=[]; grp=[]
for ds in DS:
    grp.append((ds,y))
    for m in ['Gated (ours)','DepGraph']:
        v=d[(ds,m)]; mu=np.mean(v)
        ax.scatter(v,[y]*3,s=26,color=C[m],alpha=0.8,zorder=3,edgecolor='white',linewidth=0.6)
        ax.plot([mu,mu],[y-0.32,y+0.32],color=C[m],lw=2.6,zorder=4,solid_capstyle='round')
        ax.text(mu,y+0.42,f'{mu:.2f}'.replace('-','−'),ha='center',va='bottom',fontsize=7,color=INK2)
        yt.append(y); yl.append(m); y-=1
    y-=1.0
ax.axvline(0,color=REF,lw=1.2)
ax.text(-0.15,1.45,'unpruned model',ha='right',va='bottom',fontsize=7.5,color=INK2)
ax.set_yticks(yt); ax.set_yticklabels(yl)
for ds,yy in grp: ax.text(-12.6,yy+0.55,ds,fontweight='bold',ha='left',va='bottom',fontsize=8.5)
ax.set_xlim(-12.7,0.4); ax.set_ylim(y+1.3,1.95)
ax.set_xticks(range(-12,1,2)); ax.set_xticklabels([str(t).replace('-','−') for t in range(-12,1,2)])
ax.xaxis.grid(True); ax.set_axisbelow(True)
ax.set_xlabel('Top-1 change vs the unpruned model (percentage points); dots: seeds 0–2, bar: mean')
ax.tick_params(axis='y',length=0); ax.spines['left'].set_visible(False)
save(fig,'fig_acc_change')
print('wrote', os.path.join(OUT, 'fig_acc_change') + '.pdf / .png')
