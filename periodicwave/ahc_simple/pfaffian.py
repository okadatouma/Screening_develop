"""複素反対称行列のPfaffian。転置Tを使い、共役転置は使わない。

小規模NumPyデモ：符号付き消去で計算し、sqrt(det F)は使わない。
大行列のlog振幅安定化やJAX用の微分規則は今回の範囲外。
消去法の参考：M. Wimmer, arXiv:1102.3440（2012）。
"""
import numpy as np


def _skew_matrix(F):
    F = np.asarray(F)
    if F.ndim != 2 or F.shape[0] != F.shape[1] or F.shape[0] % 2:
        raise ValueError('Pfaffianには偶数次の正方行列が必要です')
    F = np.array(F, dtype=np.result_type(F.dtype, np.complex64), copy=True)
    scale = max(1.0, float(np.max(np.abs(F))) if F.size else 1.0)
    tolerance = 64 * np.finfo(F.real.dtype).eps * scale
    if not np.isfinite(F).all() or np.max(np.abs(F + F.T), initial=0.) > tolerance:
        raise ValueError('Fは有限な反対称行列 F=-F.T にしてください')
    return F


def pfaffian4(F):
    """Pf(F)=F12 F34 − F13 F24 + F14 F23。添字を0から書く。"""
    F = _skew_matrix(F)
    if F.shape != (4, 4):
        raise ValueError('pfaffian4は4×4専用です')
    return F[0, 1]*F[2, 3] - F[0, 2]*F[1, 3] + F[0, 3]*F[1, 2]


def pfaffian(F):
    """一般の偶数N。各2行を消去するO(N³)のpivot付き計算。"""
    A = _skew_matrix(F)
    n = A.shape[0]
    if n == 4:
        return pfaffian4(A)
    value = 1.0 + 0j                       # 空行列のPfaffianは1。
    for k in range(0, n-1, 2):
        pivot_index = k + 1 + int(np.argmax(np.abs(A[k, k+1:])))
        if A[k, pivot_index] == 0:
            return 0j                     # 節をepsilonで持ち上げない。
        if pivot_index != k+1:
            # 電子ラベルの1交換はPfの符号を反転する。
            A[[k+1, pivot_index], :] = A[[pivot_index, k+1], :]
            A[:, [k+1, pivot_index]] = A[:, [pivot_index, k+1]]
            value = -value
        pivot = A[k, k+1]
        value *= pivot
        a = A[k, k+2:].copy()
        b = A[k+1, k+2:].copy()
        # 残った反対称ブロック：F'ij = Fij − (aki bkj − bki akj)/pivot。
        A[k+2:, k+2:] += (np.outer(b, a) - np.outer(a, b)) / pivot
    return value


def canonical_pairs(n):
    """J0=diag([[0,1],[-1,0]], ...) はPf(J0)=1となる規約。"""
    if n % 2:
        raise ValueError('この最小実装は偶数電子数専用です')
    J0 = np.zeros((n, n))
    for a in range(0, n, 2):
        J0[a, a+1] = 1.0
        J0[a+1, a] = -1.0
    return J0


def slater_geminal(Phi):
    """F_HF=Φ J0 Φ.T。Pf(F_HF)=det Φ（ΦはN×N、N偶数）。"""
    Phi = np.asarray(Phi)
    if Phi.ndim != 2 or Phi.shape[0] != Phi.shape[1]:
        raise ValueError('Slaterへの還元にはN×N軌道行列が必要です')
    return Phi @ canonical_pairs(len(Phi)) @ Phi.T
