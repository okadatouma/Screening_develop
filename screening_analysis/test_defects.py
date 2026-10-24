# 格子サイトの占有を入れ替えた合成配置で、既存の秩序指標の応答を調べる。
# Hamiltonianの厳密解や学習結果の検証ではなく、固定runに対応する解析の感度確認。
import sys; sys.path.insert(0,'screening_analysis')
import numpy as np, moire_analysis as MA
f='results/2deg-Coulomb/CustomScreenformer/el9_9_rs3.4156_tri_l2h16_eps10'
d=MA.load(f); L,MO,ne=d['L'],d['MO'],d['ne']
sites=MA.sites_in_cell(d)
lab=(np.rint(np.linalg.solve(MO,sites.T).T).astype(int)); sub=(lab[:,0]+lab[:,1])%3
rng=np.random.default_rng(1)
print('  欠陥数  psi_K   psi_sub   S(K)/bg')
for ndef in (0,1,2,3,4,6,9):
    B=192; R=np.zeros((B,ne,2))
    for b in range(B):
        occ=list(np.where(sub!=2)[0]); emp=list(np.where(sub==2)[0])
        for _ in range(ndef):
            i=rng.integers(len(occ)); j=rng.integers(len(emp))
            occ[i],emp[j]=emp[j],occ[i]
        R[b]=sites[occ]+0.12*rng.normal(size=(ne,2))
    dd=dict(d); dd['R']=R
    op=MA.order_parameter(dd); oc=MA.occupation_summary(dd)
    print('  %5d  %6.3f  %7.3f  %8.2f'%(ndef,op['psi_K'],oc['psi_sub'],op['K_over_bg']))
