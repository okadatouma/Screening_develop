# -*- coding: utf-8 -*-
"""初期化直後の局所エネルギー分布（band_check.py が保存した 48 点）。"""
import os, sys, argparse
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
import style as ST

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",default="out/fig_init.png")
    a=ap.parse_args(); ST.use()
    import matplotlib.pyplot as plt
    z=np.load(os.path.join(HERE,"init_spread.npz"))
    keys=["base","fixv","band8","env8_t1","env8_t05"]
    lab=["単調窓\n＋学習 value","単調窓\n＋固定 value","帯窓 8 本\n（raised cosine）",
         "輪切り 8 本\n$\\tau=1$","輪切り 8 本\n$\\tau=0.5$"]
    col=[ST.SUB,"#4C6FA5","#8a8a8a",ST.GREEN,ST.ENJI]
    fig=plt.figure(figsize=(8.6,3.3))
    fig.text(0.02,0.965,"初期化直後の局所エネルギー $E_L$（学習済み run $r_s=30$ の配置 48 点）",
             fontsize=11.4,weight="bold",color=ST.GREEN,va="top")
    ax=fig.add_axes([0.105,0.235,0.870,0.585])
    rng=np.random.default_rng(0)
    for i,k in enumerate(keys):
        E=z[k]; med=np.median(E); mad=np.median(np.abs(E-med))
        ax.plot(i+0.16*rng.normal(size=E.size),E-med,"o",ms=3.4,color=col[i],alpha=0.65,mew=0)
        ax.plot([i-0.3,i+0.3],[mad,mad],color=col[i],lw=1.6)
        ax.plot([i-0.3,i+0.3],[-mad,-mad],color=col[i],lw=1.6)
        ax.text(i,1.1e5,"MAD %.0f"%mad,ha="center",va="center",fontsize=8.8,color=col[i],weight="bold")
        ax.text(i,3.0e4,"標準偏差 %.0f"%E.std(),ha="center",va="center",fontsize=8.2,color=col[i])
    ax.set_yscale("symlog",linthresh=30)
    ax.set_xticks(range(5)); ax.set_xticklabels(lab,fontsize=8.8)
    ax.set_ylabel("$E_L$ − 中央値")
    ax.axhline(0,color=ST.INK,lw=0.7,alpha=0.5)
    ax.grid(alpha=.2,axis="y")
    ax.set_xlim(-0.55,4.55); ax.set_ylim(-2e4,3e5)
    fig.text(0.02,0.075,"横線 = 中央絶対偏差（MAD）。縦軸は symlog。",fontsize=8.4,color=ST.SUB)
    _m=[np.median(np.abs(z[k]-np.median(z[k]))) for k in keys]
    fig.text(0.02,0.026,"輪切り（$p=3$）は base の %.0f 倍（$\\tau=1$）／ %.0f 倍（$\\tau=0.5$）—— 鋭く切るほど遷移帯の曲率が上がる。"%(_m[3]/_m[0],_m[4]/_m[0]),
             fontsize=8.4,color=ST.SUB)
    fig.savefig(a.out,dpi=200,facecolor="white"); print("wrote",a.out)

if __name__=="__main__": main()
