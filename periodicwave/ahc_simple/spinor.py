"""λ-jelliumの擬スピン。成分の順序は (+1, -1) で、実スピンではない。"""
import numpy as np


def one_particle_hamiltonian(k, lam, delta=1.0):
    """h0(k)=Δ[[λ²k², -λk_-],[-λk_+,1]]。kは長さ2、自由運動k²/2は別。"""
    kx, ky = np.asarray(k, dtype=float)
    return delta * np.array([[lam**2 * (kx*kx + ky*ky), -lam*(kx-1j*ky)],
                             [-lam*(kx+1j*ky), 1.0]], dtype=complex)


def lower_spinor(k, lam):
    """下側固有ベクトルu_k。入力 [...,2] → 出力 [...,2]。"""
    k = np.asarray(k, dtype=float)
    if (k.ndim == 0 or k.shape[-1] != 2 or not np.isfinite(k).all()
            or not np.isscalar(lam) or not np.isreal(lam) or not np.isfinite(lam)):
        raise ValueError('kは有限な[...,2]、lamは有限の実数を指定してください')
    # 1. h0 u=0を満たす固有ベクトル。位相は上成分が実数正となる規約。
    vector = np.stack((np.ones(k.shape[:-1], dtype=complex),
                       lam * (k[..., 0] + 1j*k[..., 1])), axis=-1)
    # 2. 各運動量で規格化する。既存pbc.lj_hf.spinorと同じ規約。
    norm = np.sqrt(np.sum(np.abs(vector)**2, axis=-1, keepdims=True))
    return vector / norm


def pseudospin(u):
    """3. 規格化した状態の<u|σ_(x,y,z)|u>。入力 [...,2] → [...,3]。"""
    u = np.asarray(u, dtype=complex)
    if u.ndim == 0 or u.shape[-1] != 2:
        raise ValueError('スピノルの最後の軸は2成分です')
    up, down = u[..., 0], u[..., 1]
    norm2 = np.abs(up)**2 + np.abs(down)**2
    if not np.isfinite(norm2).all() or np.any(norm2 == 0):
        raise ValueError('ゼロ/非有限スピノルには期待値を定義できません')
    overlap = np.conj(up) * down
    return np.stack((2*overlap.real, 2*overlap.imag,
                     np.abs(up)**2 - np.abs(down)**2), axis=-1) / norm2[..., None]


def select_component(u, s):
    """二成分を計算した後で、評価する離散配置s=±1の成分を選ぶ。"""
    if s not in (-1, 1):
        raise ValueError('擬スピンは+1または-1です')
    return np.asarray(u)[..., 0 if s == 1 else 1]
