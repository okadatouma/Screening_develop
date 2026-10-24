# 過去に集計したチャネル寄与の数値を直接記した再描画用スクリプト。checkpointの再解析はしない。
# 元の環境の絶対パスとout/qk_decomp.pngへの保存があるため、実行前に入出力を確認する。
import sys, numpy as np
sys.path.insert(0,'/home/claude/run/screening_analysis')
import style as ST; ST.use()
import matplotlib.pyplot as plt
CAT=["並進不変の対\n$\\cos b_m\\!\\cdot\\!r_{ij}$","重心\n$\\cos b_m\\!\\cdot\\!(r_i{+}r_j)$","交差高調波\n$m\\neq n$","スピン×位置","スピン×スピン"]
D={"初期化（8 種）":[11.6,12.5,23.0,42.6,10.2],"液体 $r_s$=5":[41.6,6.6,20.5,24.4,6.8],
   "境界 $r_s$=30":[43.4,5.9,22.2,21.7,6.9],"結晶 $r_s$=60":[35.2,8.4,26.6,24.4,5.5]}
E=[11.6*0+2.1,3.3,2.5,3.9,3.6]
dim=[16,16,32,32,4]
col=["#9aa0a6",ST.GREEN,"#4C6FA5",ST.ENJI]
fig=plt.figure(figsize=(9.6,4.3))
fig.text(0.02,0.965,"内容チャネル $s_{ij}\\propto f_i A_h f_j^{T}$ の中身（$A_h$ は 5×5）",
         fontsize=13.0,weight="bold",color=ST.GREEN,va="top")
fig.text(0.02,0.905,"分散への寄与。$f=[\\sin b_1\\!\\cdot\\!r,\\ \\sin b_2\\!\\cdot\\!r,\\ \\cos b_1\\!\\cdot\\!r,\\ \\cos b_2\\!\\cdot\\!r,\\ \\sigma]$、16 ヘッドの平均。"
         "灰の帯 = 初期化 8 種の標準偏差、点線 = 成分の数え上げ（無構造なら）",fontsize=9.4,color=ST.SUB,va="top")
ax=fig.add_axes([0.075,0.235,0.905,0.565])
x=np.arange(5); w=0.20
for i,(k,v) in enumerate(D.items()):
    ax.bar(x+(i-1.5)*w,v,w*0.92,label=k,color=col[i],
           yerr=(E if i==0 else None),error_kw=dict(ecolor="#555555",lw=1.0,capsize=2.5))
for j,d0 in enumerate(dim):
    ax.plot([x[j]-2.1*w,x[j]+2.1*w],[d0,d0],ls=":",lw=1.2,color="#333333")
ax.set_xticks(x); ax.set_xticklabels(CAT,fontsize=9.2)
ax.set_ylabel("分散への寄与 [%]"); ax.set_ylim(0,52); ax.grid(alpha=.2,axis="y")
ax.legend(fontsize=9.0,ncol=4,frameon=False,loc="upper center",bbox_to_anchor=(0.5,1.16))
fig.text(0.02,0.075,"学習で並進不変の対が 11.6% → 35–43%（初期化のばらつきの 14 倍）。絶対位置とスピンに依る成分は 2 倍前後に抑えられる。",
         fontsize=8.8,color=ST.SUB)
fig.text(0.02,0.022,"結晶だけ並進不変が下がり交差高調波が上がる —— ただし結晶の run は step 数が 2 倍なので、学習量の効果と分離できていない。",
         fontsize=8.8,color=ST.SUB)
fig.savefig("out/qk_decomp.png",dpi=190,facecolor="white"); print("wrote out/qk_decomp.png")
