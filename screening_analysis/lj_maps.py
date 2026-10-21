# -*- coding: utf-8 -*-
"""λ-jellium の走りを 2 つの図で見る（2026-10-09）。

(1) 電荷の並び：基準の電子から見た他の電子の密度 g(x, y)（対密度、電子と配置で平均）。
    有限の周期セルでは一体密度 ρ(r) は並進で平均されて平らになるので、結晶かどうかは
    対密度と構造因子 S(G) で見る。S(G) は三角格子の逆格子の最短ベクトル 6 本での平均。
(2) q·k 行列：ヘッド h の内容チャネル s^h_ij = q̄_i·k̄_j（窓を掛ける前のコサイン類似度）。
    1 層目なので q, k は電子 i 自身の入力 f_i = (sin 2πs_1, sin 2πs_2, cos 2πs_1, cos 2πs_2, σ_i)
    だけで決まる（s：セルの分数座標、σ：擬スピン ±1）。

    python lj_maps.py <pulled_lj の CustomScreenformer フォルダ> <出力フォルダ> [2DEG 結晶 2DEG 液体]
"""
import os, sys, glob, json, re
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)

LAMS = [0.33, 0.4, 0.48, 0.56, 0.64, 0.72]

def lattice_of(cfg):
    m = re.search(r"unserializable object: \[\[(.*?)\]\]", json.dumps(cfg["system"]["make_local_energy_kwargs"]["lattice"]))
    v = [float(x) for x in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", m.group(1))][:4]
    return np.array(v).reshape(2, 2)
def load(folder):
    cfg = json.load(open(os.path.join(folder, "config.json")))
    ck = sorted(glob.glob(os.path.join(folder, "qmcjax_ckpt_*.npz")))[-1]
    z = np.load(ck, allow_pickle=True)
    d = z["data"].item()
    ne = sum(cfg["system"]["electrons"])
    pos = np.asarray(d["positions"])[0].reshape(-1, ne, 2)
    sp = np.asarray(d["spins"])[0] if "spins" in d else None
    return cfg, z, pos, sp, lattice_of(cfg), ne, ck
def sk(pos, L, nmax=12):
    """per-walker S(k) on the supercell reciprocal grid: S(k)=<|rho_k|^2>/N."""
    s = pos @ np.linalg.inv(L).T            # fractional
    ne = pos.shape[1]
    out = {}
    for m1 in range(-nmax, nmax + 1):
        for m2 in range(-nmax, nmax + 1):
            if (m1, m2) == (0, 0): continue
            ph = np.exp(2j * np.pi * (s @ np.array([m1, m2])))
            out[(m1, m2)] = (np.abs(ph.sum(1)) ** 2).mean() / ne
    return out


def dref_of(L, npts=201):
    lat = np.asarray(L, float); metric = lat.T @ lat
    ax = np.linspace(0.0, 1.0, npts, endpoint=False)
    g = np.stack(np.meshgrid(ax, ax, indexing="ij"), -1)
    a = 1 - np.cos(2 * np.pi * g); b = np.sin(2 * np.pi * g)
    return float((1 / (2 * np.pi)) * np.sqrt(np.einsum("...m,mn,...n->...", a, metric, a)
                                             + np.einsum("...m,mn,...n->...", b, metric, b)).max())


def first_shell(L, ne):
    """セルの分数座標で表した、三角格子の最短の逆格子ベクトル 6 本（整数ベクトル）。"""
    nc = int(round(np.sqrt(ne)))
    B = 2 * np.pi * np.linalg.inv(L).T          # セルの逆格子（列）
    cand = [(m1, m2) for m1 in range(-2 * nc, 2 * nc + 1) for m2 in range(-2 * nc, 2 * nc + 1)
            if (m1, m2) != (0, 0) and m1 % nc == 0 and m2 % nc == 0]
    k = np.array([np.linalg.norm(B @ np.array(c)) for c in cand])
    kmin = k.min()
    return [c for c, kk in zip(cand, k) if kk < kmin * 1.01]


def structure_factor(pos, L, shell):
    s = pos @ np.linalg.inv(L).T
    return float(np.mean([(np.abs(np.exp(2j * np.pi * (s @ np.array(c))).sum(1)) ** 2).mean()
                          / pos.shape[1] for c in shell]))


def pair_density(pos, L, R=4.2, nb=105, sp=None, which=None):
    """基準の電子 i から見た他の電子 j の密度 / 平均密度。which=(si, sj) で擬スピンを選ぶ。"""
    nw, ne, _ = pos.shape
    Li = np.linalg.inv(L)
    edges = np.linspace(-R, R, nb + 1)
    H = np.zeros((nb, nb)); nref = 0
    for b in range(0, nw, 64):
        P = pos[b:b + 64]
        dr = P[:, None, :, :] - P[:, :, None, :]          # [w, i, j, 2] = r_j - r_i
        f = dr @ Li.T; f -= np.round(f); dr = f @ L.T
        mask = ~np.eye(ne, dtype=bool)[None].repeat(P.shape[0], 0)
        if which is not None:
            S = sp[b:b + 64]
            mask &= (S[:, :, None] == which[0]) & (S[:, None, :] == which[1])
            nref += (S == which[0]).sum()
        else:
            nref += P.shape[0] * ne
        h, _, _ = np.histogram2d(dr[..., 0][mask], dr[..., 1][mask], bins=[edges, edges])
        H += h
    area = abs(np.linalg.det(L)); da = (edges[1] - edges[0]) ** 2
    if which is None:
        rho0 = (ne - 1) / area
    else:
        rho0 = (sp == which[1]).sum(1).mean() / area      # 相手の擬スピンの平均密度
    return H.T / (max(nref, 1) * da * rho0), edges


def qk(params, pos, sp, L):
    """s^h_ij：[配置, ヘッド, i, j]。"""
    p = params["layers"]
    E = np.asarray(p["embed"])[0]
    at = p["attention"][0]
    Wq, Wk = np.asarray(at["q_w"])[0], np.asarray(at["k_w"])[0]
    H = Wq.shape[1] // 16
    s = pos @ np.linalg.inv(L).T
    f = np.concatenate([np.sin(2 * np.pi * s), np.cos(2 * np.pi * s), sp[..., None]], -1)
    x = f @ E
    q = (x @ Wq).reshape(*x.shape[:2], H, 16); k = (x @ Wk).reshape(*x.shape[:2], H, 16)
    q = q / np.sqrt((q * q).sum(-1, keepdims=True) + 1e-12)
    k = k / np.sqrt((k * k).sum(-1, keepdims=True) + 1e-12)
    return np.einsum("wihd,wjhd->whij", q, k)


def head_lengths(params, cfg, L):
    """ヘッドを並べる長さ：入れ子窓なら窓の幅 w_h、輪切りなら輪の中心。"""
    at = params["layers"]["attention"][0]
    dref = dref_of(L)
    if "mask_rings" in at:
        b = dref * np.asarray(at["mask_rings"])[0]
        return 0.5 * (b[:-1] + b[1:]), "輪の中心"
    om = np.asarray(at["mask_omega"])[0]
    return dref * np.log1p(np.exp(om)), "窓の幅"


def radial_g(pos, L, sp, same=True, rmax=4.0, nb=40):
    """擬スピン（スピン）で分けた動径分布 g(r)：same=True で同じ向きの対、False で逆向きの対。"""
    nw, ne, _ = pos.shape
    Li = np.linalg.inv(L)
    edges = np.linspace(0, rmax, nb + 1)
    H = np.zeros(nb); npair_ref = 0.0
    area = abs(np.linalg.det(L))
    for b in range(0, nw, 64):
        P = pos[b:b + 64]; S_ = sp[b:b + 64]
        dr = P[:, None, :, :] - P[:, :, None, :]
        f = dr @ Li.T; f -= np.round(f); d = np.linalg.norm(f @ L.T, axis=-1)
        eq = S_[:, :, None] == S_[:, None, :]
        m = (eq if same else ~eq) & ~np.eye(ne, dtype=bool)[None]
        H += np.histogram(d[m], bins=edges)[0]
        npair_ref += m.sum()                     # 一様なら面積比で分布する対の数
    shell = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    return 0.5 * (edges[1:] + edges[:-1]), H / (npair_ref * shell / area)
