"""モアレ系の秩序を測る — 一体密度、密度‑密度相関、構造因子、サイト占有。

    python screening_analysis/moire_analysis.py <folder> [<folder> ...]

外場があるので並進対称性が破れ、一体密度 n(r) 自体が意味を持つ（クーロンだけの
系では平坦だった）。ν=2/3 の generalized Wigner crystal は 3 サイトのうち 2 つが
埋まる √3×√3 秩序なので、モアレ Brillouin ゾーンの角 K に Bragg ピークが立つ。
"""
import glob, json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)


def load(folder):
    cfg = json.load(open(os.path.join(folder, "config.json")))
    m = re.search(r"unserializable object: \[\[(.*?)\]\]", json.dumps(cfg))
    L = np.array([float(x) for x in re.findall(
        r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", m.group(1))][:4]).reshape(2, 2)
    ne = sum(cfg["system"]["electrons"])
    pk = cfg["system"]["make_local_energy_kwargs"]["potential_kwargs"]
    rs = float(pk["interaction_energy_scale"])
    U = float(pk.get("moire_potential_strength", 0.0))
    phi = float(pk.get("moire_potential_phi", 0.0))
    mm = re.findall(r"unserializable object: \[\[(.*?)\]\]", json.dumps(cfg))
    MO = (np.array([float(x) for x in re.findall(
        r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", mm[1])][:4]).reshape(2, 2) if len(mm) > 1 else None)
    ck = sorted(glob.glob(os.path.join(folder, "qmcjax_ckpt_*.npz")))[-1]
    pos = np.asarray(np.load(ck, allow_pickle=True)["data"].item()["positions"][0])
    return dict(L=L, MO=MO, ne=ne, rs=rs, U=U, phi=phi, folder=folder,
                R=pos.reshape(-1, ne, 2), cfg=cfg)


def wrap(R, L):
    """すべての電子を基本セル内に折り返す。"""
    f = np.linalg.solve(L, R.reshape(-1, 2).T).T
    return ((f - np.floor(f)) @ L.T).reshape(R.shape)


def density(d, nb=90):
    """一体密度 n(r)。セル内に折り返してから 2 次元ヒストグラム。"""
    L, R = d["L"], wrap(d["R"], d["L"])
    f = np.linalg.solve(L, R.reshape(-1, 2).T).T
    H, xe, ye = np.histogram2d(f[:, 0], f[:, 1], bins=nb, range=[[0, 1], [0, 1]])
    cell = abs(np.linalg.det(L)) / nb ** 2
    return H / (len(d["R"]) * cell), xe, ye          # 分数座標のグリッド上の密度


def pair2d(d, nb=81):
    """基準電子から見た周囲の密度 g(x,y)（最小像）。"""
    R, L, ne = d["R"], d["L"], d["ne"]
    dr = R[:, :, None, :] - R[:, None, :, :]
    f = dr @ np.linalg.inv(L).T
    dv = (f - np.round(f)) @ L.T
    off = ~np.eye(ne, dtype=bool)
    xy = dv[:, off].reshape(-1, 2)
    lim = 0.5 * abs(np.linalg.det(L)) / max(np.linalg.norm(L[:, 0]), np.linalg.norm(L[:, 1]))
    H, xe, ye = np.histogram2d(xy[:, 0], xy[:, 1], bins=nb, range=[[-lim, lim], [-lim, lim]])
    cell = (xe[1] - xe[0]) * (ye[1] - ye[0])
    return H / (len(R) * ne * (ne / abs(np.linalg.det(L))) * cell), xe, ye, lim


def sofk(d, nmax=9):
    """許される k 上の S(k)。k = m b1 + n b2（b はセルの逆格子）。"""
    L, R, ne = d["L"], d["R"], d["ne"]
    B = 2 * np.pi * np.linalg.inv(L).T
    ks, S = [], []
    for m in range(-nmax, nmax + 1):
        for n in range(-nmax, nmax + 1):
            if m == n == 0: continue
            k = m * B[:, 0] + n * B[:, 1]
            S.append(float((np.abs(np.exp(1j * (R @ k)).sum(1)) ** 2).mean() / ne))
            ks.append(k)
    return np.array(ks), np.array(S)


def order_parameter(d):
    """√3×√3 秩序の指標: モアレ BZ の角 K での S(k) を背景で割る。"""
    MO = d["MO"]
    bm = 2 * np.pi * np.linalg.inv(MO).T                  # モアレ格子の逆格子
    Ks = [(bm[:, 0] + bm[:, 1]) / 3, (2 * bm[:, 0] - bm[:, 1]) / 3,
          (-bm[:, 0] + 2 * bm[:, 1]) / 3]
    Gm = [bm[:, 0], bm[:, 1], bm[:, 0] + bm[:, 1]]
    ks, S = sofk(d)
    kk = np.linalg.norm(ks, axis=1)

    def at(vs, tol=1e-3):
        out = []
        for v in vs:
            i = np.argmin(np.linalg.norm(ks - v, axis=1))
            if np.linalg.norm(ks[i] - v) < tol * max(1, np.linalg.norm(v)) + 1e-6:
                out.append(S[i])
        return out
    sK, sG = at(Ks), at(Gm)
    bg = S[(kk > 0.5 * np.linalg.norm(Ks[0])) & (kk < 3 * np.linalg.norm(Gm[0]))].mean()
    ne = d["R"].shape[1]
    # 完全な nu=2/3 秩序では |rho_K| = N/2 なので S(K) = N/4 が上限
    return dict(S_K=np.mean(sK) if sK else np.nan, nK=len(sK),
                S_G=np.mean(sG) if sG else np.nan, nG=len(sG), bg=bg,
                S_K_max=ne / 4.0,
                K_over_bg=(np.mean(sK) / bg) if sK else np.nan,
                G_over_bg=(np.mean(sG) / bg) if sG else np.nan,
                psi_K=((np.mean(sK) / bg - 1) / (ne / 4.0 - 1)) if sK else np.nan)


def sites_in_cell(d):
    """スーパーセル内のモアレ谷の位置（重複なし）。"""
    L, MO = d["L"], d["MO"]
    out, seen = [], set()
    for m in range(-40, 41):
        for n in range(-40, 41):
            r = m * MO[:, 0] + n * MO[:, 1]
            f = np.mod(np.linalg.solve(L, r), 1.0)
            f = np.where((f < 1e-7) | (f > 1 - 1e-7), 0.0, f)
            key = (round(float(f[0]), 5), round(float(f[1]), 5))
            if key in seen: continue
            seen.add(key); out.append(f @ L.T)
    return np.array(out)


def site_occupation(d, R=None):
    """各モアレ谷の占有数。電子を最も近い谷に割り当てる（最小像）ので
    合計は必ず N になり、しきい値パラメータが要らない。"""
    L = d["L"]
    R = wrap(d["R"] if R is None else R, L)
    sites = sites_in_cell(d)
    dr = R[:, :, None, :] - sites[None, None, :, :]
    f = dr @ np.linalg.inv(L).T
    dd = np.linalg.norm((f - np.round(f)) @ L.T, axis=-1)      # (B, ne, nsite)
    who = np.argmin(dd, axis=-1)                               # (B, ne)
    occ = np.zeros((R.shape[0], len(sites)))
    for b in range(R.shape[0]):
        np.add.at(occ[b], who[b], 1.0)
    return sites, occ


def occupation_summary(d, R=None):
    """占有数の分布。√3x√3 (nu=2/3) なら {0,1} の二山、液体なら Poisson 的。"""
    sites, occ = site_occupation(d, R)
    mu = occ.mean(0)
    nel = occ.sum(1).mean()
    nmax = int(occ.max())
    hist = np.array([(occ == k).mean() for k in range(nmax + 1)])
    # 完全な nu=2/3 秩序: 1/3 が 0、2/3 が 1 -> 谷ごと平均の分散が最大
    nu = nel / len(sites)
    p = nu - np.floor(nu)
    ideal = float(np.sqrt(p * (1 - p))) if 0 < p < 1 else np.nan
    # √3×√3 の 3 つの部分格子（(m+n) mod 3）ごとの平均占有。
    # サイト 1 つ 1 つのばらつきは有限系・変分状態の癖でも出るので秩序変数にならない。
    # 部分格子で均すと、ハニカム秩序（3 つのうち 1 つが空く）だけが残る。
    lab = np.rint(np.linalg.solve(d["MO"], sites.T).T).astype(int)
    sub = (lab[:, 0] + lab[:, 1]) % 3
    subm = np.array([mu[sub == k].mean() for k in range(3)])
    return dict(sites=sites, occ=occ, mu=mu, contrast=float(mu.std()),
                hist=hist, nsites=len(sites), nu=nu, ideal_contrast=ideal,
                sublattice=subm, sub_contrast=float(subm.std()),
                psi_sub=float(subm.std() / ideal) if ideal == ideal else np.nan,
                psi=float(mu.std() / ideal) if ideal == ideal else np.nan)


# --------------------------------------------------------------------------
# 距離まわりの共通ルーチン
#   周期ノルム d_p はネットワークが実際に使う距離で、遠方で d_ref に飽和する。
#   物理量（a_M など）と比べるには Euclid の最小像距離に直す必要がある。
# --------------------------------------------------------------------------
_IMG = np.array([[m, n] for m in (-1, 0, 1) for n in (-1, 0, 1)], dtype=float)


def min_image(dr, L):
    """Euclid の最小像ベクトル。斜交セルなので round だけでは足りず、
    隣の 8 個の像も見る。"""
    f = dr @ np.linalg.inv(L).T
    v = (f - np.round(f)) @ L.T                       # (..., 2)
    cand = v[..., None, :] + _IMG @ L.T                # (..., 9, 2)
    i = np.argmin(np.linalg.norm(cand, axis=-1), axis=-1)
    return np.take_along_axis(cand, i[..., None, None], axis=-2)[..., 0, :]


def min_image_dist(dr, L):
    return np.linalg.norm(min_image(dr, L), axis=-1)


def periodic_norm(dr, L):
    """ネットワークの feature layer と同じ周期ノルム。"""
    s = dr @ np.linalg.inv(L).T
    M = L.T @ L
    a = 1 - np.cos(2 * np.pi * s); b = np.sin(2 * np.pi * s)
    return (1 / (2 * np.pi)) * np.sqrt(
        np.einsum("...m,mn,...n->...", a, M, a) +
        np.einsum("...m,mn,...n->...", b, M, b))


def d_ref(L):
    return float(periodic_norm(_scan(L), L).max())


def _scan(L, n=400):
    g = np.linspace(0, 1, n, endpoint=False)
    X, Y = np.meshgrid(g, g, indexing="ij")
    return np.stack([X, Y], -1).reshape(-1, 2) @ L.T


def pnorm_to_euclid(L, n=400):
    """周期ノルム -> Euclid 最小像距離 の対応表 (dp, de) を作る。
    向きにほとんど依らないので 1 対 1 に扱える。"""
    r = _scan(L, n)
    dp = periodic_norm(r, L); de = min_image_dist(r, L)
    o = np.argsort(dp)
    dp, de = dp[o], de[o]
    k = max(1, len(dp) // 600)
    dp = dp[:len(dp) // k * k].reshape(-1, k).mean(1)
    de = de[:len(de) // k * k].reshape(-1, k).mean(1)
    return dp, de


def euclid_to_w(de, L):
    """Euclid の半径を周期ノルムの窓に直す（w_to_euclid の逆）。"""
    dp, dee = pnorm_to_euclid(L)
    return np.interp(np.asarray(de, dtype=float), dee, dp,
                     left=float(dp.min()), right=float(dp.max()))


def w_to_euclid(w, L):
    """遮蔽窓 w（周期ノルム）を Euclid の半径に直す。d_ref を超える窓は inf。"""
    dp, de = pnorm_to_euclid(L)
    w = np.asarray(w, dtype=float)
    out = np.interp(w, dp, de, left=0.0, right=np.inf)
    return np.where(w >= dp.max(), np.inf, out)


if __name__ == "__main__":
    print("\n  %-30s %6s %6s %8s %8s %8s %8s" %
          ("run", "r_s", "U", "S(K)/bg", "S(G)/bg", "占有平均", "占有コントラスト"))
    print("  " + "-" * 84)
    for f in sys.argv[1:]:
        d = load(f)
        op = order_parameter(d)
        os_ = occupation_summary(d); mu = os_["mu"]
        print("  %-30s %6.3g %6.3g %8.2f %8.2f %8.3f %8.3f"
              % (os.path.basename(f), d["rs"], d["U"], op["K_over_bg"], op["G_over_bg"],
                 mu.mean(), os_["contrast"]))
