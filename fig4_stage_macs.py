"""fig4_stage_macs.py - Figure 4 of the paper.

MACs kept per stage by gated pruning and DepGraph (% of the unpruned stage).
Data: per-stage MACs counted on the 18 exported pruned models and the unpruned models (224 x 224 input).
The values below are the measured results reported in the paper; edit them if
you re-run the experiments.

Usage:  python fig4_stage_macs.py [output_folder]     (default: figures/out)
Writes: fig_stage_macs.pdf and fig_stage_macs.png
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

rows=[('Stage 1','56×56',33.87,21.9,19.3,24.0,16.7),('Stage 2','28×28',44.16,30.0,28.5,31.3,56.4),
      ('Stage 3','14×14',87.33,57.0,55.1,59.1,54.5),('Stage 4','7×7',31.97,88.1,82.1,91.8,62.1)]
tot=sum(r[2] for r in rows)
fig,ax=plt.subplots(figsize=(W,3.5))
y=0; yt=[];yl=[]
for st,res,base,g,lo,hi,dg in rows:
    ax.text(-1,y+0.5,f'{st}',fontweight='bold',ha='left',va='bottom')
    ax.text(14,y+0.5,f'{res} feature maps · {base:.1f} M MACs ({100*base/tot:.0f}% of the unpruned model)',ha='left',va='bottom',color=INK2,fontsize=7.5)
    ax.barh(y,g,height=0.62,color=C['Gated (ours)'],zorder=3)
    ax.plot([lo,hi],[y,y],color='#0b0b0b',lw=1.1,zorder=4); 
    ax.text(hi+1.2,y,f'{g:.0f}%',va='center',fontsize=7.5)
    yt.append(y); yl.append('Gated (ours)'); y-=0.8
    ax.barh(y,dg,height=0.62,color=C['DepGraph'],zorder=3)
    ax.text(dg+1.2,y,f'{dg:.0f}%',va='center',fontsize=7.5)
    yt.append(y); yl.append('DepGraph'); y-=1.75
ax.axvline(100,color=REF,lw=1.2,zorder=2); ax.text(100,1.25,'unpruned stage',ha='center',va='bottom',fontsize=7.5,color=INK2)
ax.set_yticks(yt); ax.set_yticklabels(yl); ax.tick_params(axis='y',length=0); ax.spines['left'].set_visible(False)
ax.set_xlim(0,104); ax.set_ylim(y+1.1,1.8); ax.xaxis.grid(True); ax.set_axisbelow(True)
ax.set_xlabel('MACs kept per stage (% of the unpruned stage); gated: mean of 9 models, whisker min–max')
save(fig,'fig_stage_macs')
print('wrote', os.path.join(OUT, 'fig_stage_macs') + '.pdf / .png')
