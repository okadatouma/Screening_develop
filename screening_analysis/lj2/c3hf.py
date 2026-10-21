"""HF の Slater 行列式の C3 固有値（厳密）：<Phi|U|Phi> = det M、M_jk = <phi_j|U|phi_k>。
U：位置を中心 c のまわりに 120° 回し、擬スピンに D = diag(1, e^{i theta}) を掛ける（h(Rq) = D h(q) D^dag なので H を保つ）。
平面波スピノル |q> = e^{iq.r} u_q に対し U|q> = e^{i(q - Rq).c} |Rq>（D u_q = u_{Rq} がちょうど成り立つ）。"""
import numpy as np

a = np.sqrt(2 * np.pi / np.sqrt(3))
A = np.array([[a, -a / 2], [0, a * np.sqrt(3) / 2]])          # 結晶の格子（列）
B = 2 * np.pi * np.linalg.inv(A).T                              # 結晶の逆格子（列）
Bs = B / 6.0                                                    # スーパーセルの逆格子（列）


def load_table(path):
    t = np.loadtxt(path)
    ks = []
    for row in t[:, :2]:
        if not ks or np.max(np.abs(ks[-1] - row)) > 1e-9:
            ks.append(row.copy())
    nk = len(ks); ng = len(t) // nk
    return np.array(ks), t[:ng, 2:4], (t[:, 4] + 1j * t[:, 5]).reshape(nk, ng)


def rot(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def c3_matrix(k, G, c, center, theta=2 * np.pi / 3):
    R = rot(theta)
    q = k[:, None, :] + G[None, :, :]                           # (Nk, NG, 2)
    Rq = q @ R.T
    inv = np.linalg.inv(Bs)
    mq = np.rint(q.reshape(-1, 2) @ inv.T).astype(int)          # スーパーセルの逆格子の整数座標
    mRq = np.rint(Rq.reshape(-1, 2) @ inv.T).astype(int)
    lut = {tuple(m): i for i, m in enumerate(mq)}
    nk, ng = c.shape
    M = np.zeros((nk, nk), complex)
    phase = np.exp(1j * ((q - Rq) @ np.asarray(center)))       # (Nk, NG)
    for idx, m in enumerate(mRq):
        jj = lut.get(tuple(m), -1)
        if jj < 0:
            continue
        kk, g = divmod(idx, ng)                                 # 元の軌道 kk、平面波 g
        j, gp = divmod(jj, ng)                                  # 回した先の軌道 j、平面波 gp
        M[j, kk] += np.conj(c[j, gp]) * c[kk, g] * phase[kk, g]
    return M


def c3_value(k, G, c, center):
    return np.linalg.det(c3_matrix(k, G, c, center))


def find_centers(k, G, c, n=24):
    """結晶の単位胞を n x n で走査して |det M| の大きい点を探す（分数座標と値）。"""
    out = []
    for u in range(n):
        for v in range(n):
            f = np.array([u / n, v / n]); r = A @ f
            out.append((abs(c3_value(k, G, c, r)), f, r))
    out.sort(key=lambda t: -t[0])
    return out


if __name__ == "__main__":
    import sys
    for path, lam in [(p, float(p.split('lam')[1].split('_')[0])) for p in sys.argv[1:]]:
        k, G, c = load_table(path)
        top = find_centers(k, G, c)
        print(path, " 上位：", [(round(t[0], 3), tuple(np.round(t[1], 3))) for t in top[:6]])
