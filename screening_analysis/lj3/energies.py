"""lj3（入力 12 成分・Trim 固定）と lj2 のエネルギーの比較。最後の 10,000 step の平均（誤差は 500 step のブロック）と、同じ step の窓での平均。"""
import numpy as np, pandas as pd
base = 'results/lambda-jellium/CustomScreenformer'
conv = (2 / 15**2) / 36 * 1000          # コード単位（36 電子の合計）-> 1 電子あたり mRy
def blk(a, bs=500):
    m = len(a) // bs
    b = a[:m * bs].reshape(m, bs).mean(1)
    return a.mean(), b.std(ddof=1) / np.sqrt(m)
runs = [('0.33', 'WC', 'cw33', 'hw33'), ('0.33', 'AHC', 'ca33', 'ha33'), ('0.72', 'WC', 'cw72', 'hw72'), ('0.72', 'AHC', 'ca72', 'ha72')]
rows = []
for lam, typ, t2, t3 in runs:
    st = {k: pd.read_csv(f'{base}/N36_rs15_lam{lam}_{t}/train_stats.csv') for k, t in (('lj2', t2), ('lj3', t3))}
    smax = min(st['lj2'].step.max(), st['lj3'].step.max())
    r = dict(lam=lam, type=typ)
    for k, s in st.items():
        last = s[s.step >= s.step.max() - 10000]
        e, de = blk(last.energy.values)
        r[f'E_{k}'] = e; r[f'dE_{k}'] = de; r[f'steps_{k}'] = int(s.step.max())
        r[f'sig_{k}'] = last.locstd.mean(); r[f'nm_{k}'] = last.nminus.mean() / 36
        r[f'E0_{k}'] = s.energy.values[:20].mean()
        # 同じ step の窓（両方に共通する最後の 10,000 step）
        w = s[(s.step > smax - 10000) & (s.step <= smax)]
        r[f'Ew_{k}'], r[f'dEw_{k}'] = blk(w.energy.values)
        for a, b in ((0, 1000), (4000, 6000), (14000, 16000), (29000, 31000)):
            r[f'E{a//1000}k_{k}'] = s[(s.step >= a) & (s.step < b)].energy.mean()
    rows.append(r)
d = pd.DataFrame(rows)
pd.set_option('display.width', 250)
print("最後の 10,000 step（各 run の最後）")
for _, r in d.iterrows():
    print(f"  lam {r.lam} {r.type:3s}  lj2 {r.E_lj2:9.3f}±{r.dE_lj2:.3f} (step {r.steps_lj2})  lj3 {r.E_lj3:9.3f}±{r.dE_lj3:.3f} (step {r.steps_lj3})"
          f"  差 lj3-lj2 = {r.E_lj3 - r.E_lj2:+7.3f} = {(r.E_lj3 - r.E_lj2) * conv:+.3f} mRy/e   sigma {r.sig_lj2:.1f} -> {r.sig_lj3:.1f}   n-/N {r.nm_lj2:.3f} -> {r.nm_lj3:.3f}")
print("同じ step の窓（共通の最後の 10,000 step）")
for _, r in d.iterrows():
    print(f"  lam {r.lam} {r.type:3s}  lj2 {r.Ew_lj2:9.3f}±{r.dEw_lj2:.3f}  lj3 {r.Ew_lj3:9.3f}±{r.dEw_lj3:.3f}  差 {r.Ew_lj3 - r.Ew_lj2:+7.3f}")
print("途中の step（平均の窓 0-1k, 4-6k, 14-16k, 29-31k）")
for _, r in d.iterrows():
    print(f"  lam {r.lam} {r.type:3s}  lj2 " + " ".join(f"{r[f'E{a}k_lj2']:9.3f}" for a in (0, 4, 14, 29))
          + "   lj3 " + " ".join(f"{r[f'E{a}k_lj3']:9.3f}" for a in (0, 4, 14, 29)) + f"   step0 lj2 {r.E0_lj2:.3f} lj3 {r.E0_lj3:.3f}")
print("相の順（E_AHC - E_WC）")
for lam in ('0.33', '0.72'):
    w = d[(d.lam == lam) & (d.type == 'WC')].iloc[0]; a = d[(d.lam == lam) & (d.type == 'AHC')].iloc[0]
    for k in ('lj2', 'lj3'):
        x = a[f'E_{k}'] - w[f'E_{k}']; dx = np.hypot(a[f'dE_{k}'], w[f'dE_{k}'])
        print(f"  lam {lam} {k}: {x:+7.3f} ± {dx:.3f} = {x*conv:+.3f} ± {dx*conv:.3f} mRy/e  低い方 {'WC' if x > 0 else 'AHC'}")
d.to_csv('energies_lj3_vs_lj2.csv', index=False)
