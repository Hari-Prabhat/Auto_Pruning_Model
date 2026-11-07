"""fig9_int8_agreement.py - Figure 9 of the paper.

Agreement of Raspberry Pi INT8 predictions with the x86 host for the same model file.
Data: raspberry_pi/run_pi_bench.sh on 1,000 test images versus LiteRT 2.3.0 on the host; 27 INT8 files.
The values below are the measured results reported in the paper; edit them if
you re-run the experiments.

Usage:  python fig9_int8_agreement.py [output_folder]     (default: figures/out)
Writes: fig_int8_agreement.pdf and fig_int8_agreement.png
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

ag={'CIFAR-10':{'Unpruned':[99.2,95.0,99.6],'Gated (ours)':[98.9,99.9,99.2],'DepGraph':[99.2,98.9,93.8]},
    'CIFAR-100':{'Unpruned':[77.0,76.7,98.0],'Gated (ours)':[76.1,78.3,96.9],'DepGraph':[69.2,59.3,78.8]},
    'Tiny-ImageNet-200':{'Unpruned':[63.2,95.3,65.2],'Gated (ours)':[65.2,68.1,65.9],'DepGraph':[64.1,34.6,62.7]}}
fig,ax=plt.subplots(figsize=(W,3.3))
y=0; yt=[];yl=[]; grp=[]
for ds,dd in ag.items():
    grp.append((ds,y))
    for m,v in dd.items():
        ax.scatter(v,[y]*3,s=28,color=C[m],alpha=0.85,zorder=3,edgecolor='white',linewidth=0.6)
        yt.append(y); yl.append(m); y-=0.8
    y-=1.0
for ds,yy in grp: ax.text(30.5,yy+0.42,ds,fontweight='bold',va='bottom')
ax.set_yticks(yt); ax.set_yticklabels(yl); ax.tick_params(axis='y',length=0); ax.spines['left'].set_visible(False)
ax.set_xlim(30,101); ax.set_ylim(y+1.3,1.25); ax.xaxis.grid(True); ax.set_axisbelow(True)
ax.set_xticks(range(30,101,10)); ax.set_xticklabels([f'{t}%' for t in range(30,101,10)])
ax.set_xlabel('Test images (of 1,000) on which Pi 3B+ TFLite INT8 and the x86 host predict the same class (same model file)')
save(fig,'fig_int8_agreement')
print('wrote', os.path.join(OUT, 'fig_int8_agreement') + '.pdf / .png')
