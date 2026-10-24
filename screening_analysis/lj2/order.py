# 固定されたN36の保存標本から密度Fourier成分、構造因子、密度ピーク付近の負擬スピン比を計算する。
# 新しくMCMCを回す処理ではない。標本数1024・特定パスを仮定し、order.jsonへ保存する。
import numpy as np, glob, json
base = 'results/lambda-jellium/CustomScreenformer'
a = np.sqrt(2 * np.pi / np.sqrt(3))
A = np.array([[a, -a / 2], [0, a * np.sqrt(3) / 2]])          # 結晶の格子（列）
B = 2 * np.pi * np.linalg.inv(A).T
G6 = np.array([B[:, 0], B[:, 1], B[:, 0] + B[:, 1]]); G6 = np.concatenate([G6, -G6])
hf = {(r['lam'], r['kind']): r for r in json.load(open('/home/claude/ljslides/hf_summary.json'))['runs']}
out = []
for lam in ['0.33', '0.4', '0.48', '0.56', '0.64', '0.72']:
    for typ in ['cw', 'ca']:
        tag = typ + {'0.33':'33','0.4':'40','0.48':'48','0.56':'56','0.64':'64','0.72':'72'}[lam]
        ck = sorted(glob.glob(f'{base}/N36_rs15_lam{lam}_{tag}/qmcjax_ckpt_*.npz'))[-1]
        z = np.load(ck, allow_pickle=True); d = z['data'].item()
        R = np.asarray(d['positions'])[0].reshape(1024, 36, 2).astype(np.float64)
        S = np.asarray(d['spins'])[0]
        ph = np.exp(-1j * np.einsum('gd,wnd->wgn', G6, R))       # (W, 6, N)
        rho = ph.sum(-1)                                           # (W, 6)
        rho_m = (ph * (S[:, None, :] < 0)).sum(-1)
        mod = np.abs(rho.mean(0)).mean() / 36                      # |<rho_G>|/N
        SG = (np.abs(rho) ** 2).mean() / 36                        # <|rho_G|^2>/N
        # 単位胞での密度と擬スピン -1 の密度（分数座標で折り返し）
        f = np.einsum('dk,wnk->wnd', np.linalg.inv(A), R) % 1.0
        nb = 18
        H, _, _ = np.histogram2d(f[..., 0].ravel(), f[..., 1].ravel(), bins=nb, range=[[0, 1], [0, 1]])
        Hm, _, _ = np.histogram2d(f[..., 0][S < 0], f[..., 1][S < 0], bins=nb, range=[[0, 1], [0, 1]])
        i, j = np.unravel_index(np.argmax(H), H.shape)
        # 山のまわり 3x3 の平均で n_-/n
        sl = [((i + di) % nb, (j + dj) % nb) for di in (-1, 0, 1) for dj in (-1, 0, 1)]
        frac_peak = sum(Hm[p] for p in sl) / sum(H[p] for p in sl)
        h = hf[(float(lam), 'wc' if typ == 'cw' else 'ahc')]
        out.append(dict(lam=float(lam), type='WC' if typ == 'cw' else 'AHC', mod=mod, hf_mod=h['rhoG'], SG=SG,
                        nm=float((S < 0).mean()), peak_nm=frac_peak, peak_over_mean=H.max() / H.mean()))
        print(f"lam {lam:4s} {'WC ' if typ == 'cw' else 'AHC'}: |<rho_G>|/N = {mod:.3f} (HF {h['rhoG']:.3f})   S(G) = {SG:6.2f}   "
              f"n_-/N = {(S < 0).mean():.3f}   山での n_-/n = {frac_peak:.2f}   山/平均 = {H.max() / H.mean():.2f}", flush=True)
json.dump(out, open('order.json', 'w'), indent=1)
