"""解析的な二成分波動関数の小実装（NumPy、2D、学習/energy計算なし）。

既存lj_orbitals.pyは ψ=e^J Σ_d det[φ_HF + ε W_s h(R,S)] を作る。
HF係数は固定、軌道射影とTransformerおよびJastrow長さが学習可能。
本ファイルはその大きなネットワークをコピーせず、φ→det→J/BF→Pfを順に書く。

位置は既存模型のコード長さ、波数はその逆数。格子ベクトルはlatticeの列。
厳密な下側バンド軌道はc_G u_(k+G)に制限される。任意の二成分出力はこの制約を持たない。
一般のJastrow/Backflow/Geminalを加えた状態は、有限Δの二バンド試行状態として扱う。
例外：J/BFなしで、δFも下側バンド軌道の外積だけから作るPfaffianは厳密投影を保つ。
AHC/WCの識別・基底状態最適化・Chern数の認定は、振幅の評価だけではできない。
大N/節近傍のlog振幅安定化や、JAXでの学習用微分APIは今回実装しない。
"""
from functools import partial
import numpy as np

from .spinor import lower_spinor, select_component
from .pfaffian import pfaffian, slater_geminal


def load_hf_table(path):
    """lj_hf.LJHF.orbitals_textの6列をNumPyだけで読む。返値k[N,2],G[NG,2],c[N,NG]。

    既存loaderはJAX等をimportするため、この小デモでは直接読み、単一行も2Dとして扱う。
    連続したkのブロックと、各ブロックの同じGの並びを検査する。表そのものは変更しない。
    """
    table = np.loadtxt(path, ndmin=2)
    if table.shape[1] != 6 or not np.isfinite(table).all():
        raise ValueError('HF表は有限な6列 kx ky Gx Gy Re(c) Im(c) が必要です')
    starts = [0]
    for row in range(1, len(table)):
        if not np.allclose(table[row, :2], table[row-1, :2], atol=1e-9, rtol=0):
            starts.append(row)
    blocks = np.split(table, starts[1:])
    G = blocks[0][:, 2:4]
    for block in blocks:
        if block.shape != blocks[0].shape or not np.allclose(block[:, 2:4], G, atol=1e-9, rtol=0):
            raise ValueError('各kのブロックは同じGを同じ順序で含む必要があります')
    k = np.array([block[0, :2] for block in blocks])
    coeff = np.array([block[:, 4] + 1j*block[:, 5] for block in blocks])
    return k, G, coeff


def bloch_spinor(r, k, G, coeff, lam):
    """φ_a(r)の両成分：Σ_G c_aG u_(k+G) exp(i(k+G)·r) → [2]。"""
    q = np.asarray(k) + np.asarray(G)
    u = lower_spinor(q, lam)
    terms = np.asarray(coeff) * np.exp(1j * (q @ np.asarray(r)))
    return np.sum(terms[:, None] * u, axis=0)


def bloch_orbital(r, s, k, G, coeff, lam):
    """離散ラベル方式：先にu(s)を選んで足す。+1は上成分、−1は下成分。"""
    q = np.asarray(k) + np.asarray(G)
    component = select_component(lower_spinor(q, lam), s)
    terms = np.asarray(coeff) * np.exp(1j * (q @ np.asarray(r)))
    return np.sum(terms * component)


def unprojected_bloch_spinor(r, k, G, coeff):
    """任意の二成分軌道：Σ_G C_(G,σ) exp(i(k+G)·r)、Cは[NG,2]の複素数。

    各Gに独立な4実数を持ち、C_Gσ=c_G u_(k+G)(σ)という下側バンド制約を課さない。
    このr→[2]関数もorbital_matrix(...,spinor_output=True)へ渡せる。
    """
    coeff = np.asarray(coeff, dtype=complex)
    if coeff.shape != (len(G), 2):
        raise ValueError('二成分係数のshapeは[NG,2]です')
    phase = np.exp(1j * ((np.asarray(k) + np.asarray(G)) @ np.asarray(r)))
    return phase @ coeff


def make_bloch_orbitals(k, G, coeff, lam, *, spinor_output=False):
    """表から軌道関数のlistを作る。spinor_output=Trueでは各関数がr→[2]を返す。"""
    if np.shape(coeff) != (len(k), len(G)):
        raise ValueError('coeffのshapeは[軌道数, Gの数]です')
    function = bloch_spinor if spinor_output else bloch_orbital
    return [partial(function, k=ka, G=G, coeff=ca, lam=lam) for ka, ca in zip(k, coeff)]


def orbital_matrix(R, S, orbitals, *, spinor_output=False):
    """Φ_ia=φ_a(r_i,s_i)、shape[N,軌道数]。行が電子、列が軌道。"""
    R, S = np.asarray(R, dtype=float), np.asarray(S)
    if R.ndim != 2 or R.shape[1] != 2 or S.shape != (len(R),) or not np.isin(S, (-1, 1)).all():
        raise ValueError('Rは[N,2]、Sは[N]の±1配列です')
    Phi = np.empty((len(R), len(orbitals)), dtype=complex)
    for i, (r, s) in enumerate(zip(R, S)):
        for a, orbital in enumerate(orbitals):
            # B方式：軌道への入力は位置だけ。ただしHilbert空間の擬スピンは残る。
            Phi[i, a] = select_component(orbital(r), s) if spinor_output else orbital(r, s)
    return Phi


def slater_wavefunction(R, S, orbitals, *, spinor_output=False):
    """ψ_S=det Φ。スピン別に分割せず、全電子のN×N行列式を取る。"""
    Phi = orbital_matrix(R, S, orbitals, spinor_output=spinor_output)
    if Phi.shape[0] != Phi.shape[1]:
        raise ValueError('Slater行列式には電子数と同じ数の軌道が必要です')
    return np.linalg.det(Phi)


def periodic_distance(dr, lattice=None):
    """周期ノルム。dr[2]→長さ。PBCなしではEuclid距離。

    A=lattice, f=A^-1 dr, M=A.T Aに対し d²=(a.T M a+b.T M b)/(2π)²、
    a=1−cos(2πf), b=sin(2πf)。生のCoulomb距離や最小像距離ではない。
    """
    dr = np.asarray(dr, dtype=float)
    if lattice is None:
        return np.linalg.norm(dr)
    A = np.asarray(lattice, dtype=float)
    f = np.linalg.solve(A, dr)
    a, b = 1-np.cos(2*np.pi*f), np.sin(2*np.pi*f)
    metric = A.T @ A
    return np.sqrt(a @ metric @ a + b @ metric @ b) / (2*np.pi)


def jastrow(R, S, parameters=None, lattice=None):
    """J=Σ_i<j −a_ss' exp(−d_ij/b_ss')。aは実数/複素、bは正の実数。

    parameters={'a': [a_pp,a_mm,a_pm], 'b': [b_pp,b_mm,b_pm]}。
    a=0でJ=0。小距離の傾きa/bは任意で、カスプを自動保証しない。
    既存λ版のγに合わせるならa=γb（既存はb=softplus(theta)）。
    """
    if parameters is None:
        return 0j
    if np.shape(S) != (len(R),) or not np.isin(S, (-1, 1)).all():
        raise ValueError('Sは電子数に一致した±1配列です')
    if np.iscomplexobj(parameters['b']):
        raise ValueError('Jastrowの距離幅bは実数です')
    a = np.broadcast_to(np.asarray(parameters['a'], dtype=complex), (3,))
    b = np.broadcast_to(np.asarray(parameters['b'], dtype=float), (3,))
    if not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(b <= 0):
        raise ValueError('Jastrowのaは有限、bは有限で正です')
    J = 0j
    for i in range(len(R)):
        for j in range(i):
            channel = 0 if S[i] == S[j] == 1 else (1 if S[i] == S[j] == -1 else 2)
            d = periodic_distance(np.asarray(R[i])-R[j], lattice)
            J -= a[channel] * np.exp(-d / b[channel])
    return J


def relative_displacement(dr, lattice=None, mode='smooth'):
    """Backflow用の奇な相対ベクトル。smoothはA sin(2πA^-1 dr)/(2π)。"""
    dr = np.asarray(dr, dtype=float)
    if lattice is None:
        return dr
    A = np.asarray(lattice, dtype=float)
    f = np.linalg.solve(A, dr)
    if mode == 'smooth':
        # 原点でdrに一致する傾き。周期的・滑らかだが最小像ベクトルとは異なる。
        return A @ np.sin(2*np.pi*f) / (2*np.pi)
    if mode != 'minimum_image':
        raise ValueError('modeはsmoothまたはminimum_imageです')
    f = f - np.rint(f)
    best = A @ f
    # 斜交セルでも最短像を探す。σ_min|f−n|<=|A(f−n)|を使い探索範囲を囲む。
    sigma_min = np.linalg.svd(A, compute_uv=False)[-1]
    radius = int(np.ceil(np.linalg.norm(best)/sigma_min)) + 1
    for n1 in range(-radius, radius+1):
        for n2 in range(-radius, radius+1):
            candidate = A @ (f - [n1, n2])
            if np.linalg.norm(candidate) < np.linalg.norm(best):
                best = candidate
    return best                           # 像の切替面では微分不連続。


def backflow(R, parameters=None, lattice=None, *, spins=None):
    """r̃_i=r_i+Σ_j≠i η_ss' exp(−d²/b²) v_periodic(r_i−r_j)。

    {'eta': 実数または[++,--,+-], 'b': 正の実数, 'mode': 'smooth'}。
    spins=Noneなら擬スピン非依存（etaはスカラー）。座標は実数のまま。
    周期境界でr̃_iはr_iと同じ格子ベクトルだけ動き、Bloch位相を保つ。
    smoothではdを周期ノルム、minimum_imageでは最短像のEuclid距離とする。
    """
    R = np.asarray(R, dtype=float)
    if parameters is None:
        return R.copy()
    if spins is not None and (np.shape(spins) != (len(R),) or not np.isin(spins, (-1, 1)).all()):
        raise ValueError('spinsは電子数に一致した±1配列です')
    if np.iscomplexobj(parameters['b']):
        raise ValueError('Backflowの距離幅bは実数です')
    eta = np.asarray(parameters['eta'])
    b = np.broadcast_to(np.asarray(parameters['b'], dtype=float), (3,))
    if np.iscomplexobj(eta) or not np.isfinite(eta).all() or not np.isfinite(b).all() or np.any(b <= 0):
        raise ValueError('Backflowのetaは有限の実数、bは有限で正です')
    if spins is None and eta.ndim != 0:
        raise ValueError('擬スピン依存etaにはspinsを指定してください')
    eta = np.broadcast_to(eta, (3,))
    shifted = R.copy()
    for i in range(len(R)):
        for j in range(len(R)):
            if i == j:
                continue
            channel = 0
            if spins is not None:
                channel = 0 if spins[i] == spins[j] == 1 else (1 if spins[i] == spins[j] == -1 else 2)
            dr = R[i] - R[j]
            mode = parameters.get('mode', 'smooth')
            vector = relative_displacement(dr, lattice, mode)
            d = periodic_distance(dr, lattice) if mode == 'smooth' else np.linalg.norm(vector)
            shifted[i] += eta[channel] * np.exp(-d*d/b[channel]**2) * vector
    return shifted


def geminal_matrix(R, S, orbital_a, orbital_b, *, spinor_output=False):
    """δF_ij=a_i b_j−b_i a_j。二電子の交換で反転する複素Geminal。"""
    values = orbital_matrix(R, S, [orbital_a, orbital_b], spinor_output=spinor_output)
    a, b = values[:, 0], values[:, 1]
    return np.outer(a, b) - np.outer(b, a)


def wavefunction(R, S, orbitals, *, jastrow_parameters=None, backflow_parameters=None,
                 lattice=None, use_pfaffian=False, eta=0.0, geminal_orbitals=None,
                 spinor_output=False):
    """ψ=e^J Pf[Φ(r̃,S) J0 Φ(r̃,S).T + η δF(R,S)]、またはe^J det Φ。

    NoneでJ/BFをオフ、eta=0でGeminal補正をオフ。Pfaffian分岐は偶数N専用。
    δFは式どおり元のRで評価する。新しい自由度には占有空間外の追加軌道を使う。
    占有N軌道内だけのF=Φ B Φ.TではPf=det Φ Pf(B)なので、形は定数倍しか変わらない。
    """
    J = jastrow(R, S, jastrow_parameters, lattice)
    shifted = backflow(R, backflow_parameters, lattice, spins=S)
    Phi = orbital_matrix(shifted, S, orbitals, spinor_output=spinor_output)
    if Phi.shape[0] != Phi.shape[1]:
        raise ValueError('軌道数は電子数と一致させてください')
    if use_pfaffian:
        F = slater_geminal(Phi)
        if eta != 0:
            if geminal_orbitals is None or len(geminal_orbitals) != 2:
                raise ValueError('eta≠0には追加のGeminal軌道2本が必要です')
            F += eta * geminal_matrix(R, S, *geminal_orbitals, spinor_output=spinor_output)
        amplitude = pfaffian(F)
    else:
        if eta != 0:
            raise ValueError('Geminal補正を使うときはuse_pfaffian=Trueです')
        amplitude = np.linalg.det(Phi)
    return np.exp(J) * amplitude
