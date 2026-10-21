# -*- coding: utf-8 -*-
"""
screening_electrons.py
======================
screening_electrons.ipynb と中身は同じですが、Jupyter を使わずに
ターミナルからそのまま実行できます。

    python screening_electrons.py

図はウィンドウを開かず、同じフォルダに
    screening_fig1.png ... screening_fig4.png
として保存されます。所要時間は CPU で3〜5分です。

必要なもの: torch, matplotlib, numpy （periodicwave は不要）
"""
import matplotlib
matplotlib.use("Agg")          # ウィンドウを開かない
import matplotlib.pyplot as _plt
_fig_no = [0]
def _save_instead_of_show(*a, **k):
    _fig_no[0] += 1
    name = f"screening_fig{_fig_no[0]}.png"
    _plt.savefig(name, dpi=140, bbox_inches="tight")
    print(f"[figure saved: {name}]")
    _plt.close("all")
_plt.show = _save_instead_of_show


# # 遮蔽注意（Screening Attention）をゼロから実装する — 電子系ハンズオン版
# 
# `transformer_shiawase.ipynb` の電子系バージョンです。あちらは「文字を並べて次の文字を当てる」
# 言語モデルでしたが、こちらは **2次元の調和トラップに閉じ込めた6個の電子の基底状態**を
# 変分モンテカルロ（VMC）で求めます。注意機構は softmax ではなく、
# 教科書 Part 3 の**遮蔽注意**（Trim・Softmask・分母なし集約・TanhNorm・ゲート）を使います。
# 
# **この notebook でやること**
# 
# 1. 「トークン＝電子」になると何が変わるかを確認する
# 2. 厳密に解ける自由フェルミオン状態を土台に置き、局所エネルギーの機構を検証する（E = 14 ちょうど）
# 3. softmax 注意と**遮蔽注意**を両方実装して、重み行列を見比べる
# 4. VMC で両方を学習させ、エネルギーと**学習された遮蔽長**を比べる
# 
# `periodicwave/CustomScreenformer.py` の遮蔽注意の中身はそのまま移植してあります。
# 簡素化したのは周りの物理（周期境界条件・Ewald 和・KFAC を使わない）だけです。
# 
# **実行時間の目安**: 全セルで3〜5分（CPU）。上から順に Shift+Enter。

# --- Step 0: セットアップ ---
# torch が無ければ下の行の # を外して一度実行
# %pip install torch matplotlib

import math, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)   # 局所エネルギーの厳密性チェックのため倍精度
print("PyTorch", torch.__version__)

# ## Step 1: 「トークン」は電子である
# 
# 言語モデルではトークンは文字で、**並び順**を持っていました。ここではトークンは電子そのものです。
# 電子には並び順がありません。この一点から3つの帰結が出ます。
# 
# | 言語モデル | 電子系 |
# |---|---|
# | 位置埋め込みを足す | **足さない**（順序が無いので、足すと嘘になる） |
# | 因果マスク（未来を見ない） | **無い**（時間方向が無い。全電子が全電子を見る） |
# | 出力は次トークンの確率分布 | 出力は**反対称**な波動関数（交換で符号が反転） |
# 
# さらに、Softmask の「距離」も置き換わります。言語では距離はトークン位置の差 |i − j| でしたが、
# 電子系では**物理的な電子間距離 |r_i − r_j|** です。
# つまり遮蔽窓 w は「何ボーア半径先まで見るか」という**物理的な遮蔽長**になります。
# 
# 系の設定（以下すべて ℏ = m = ω = 1 の自然単位）:
# 
# - 2次元の調和トラップ  V_trap = ½ Σ_i |r_i|²
# - スピン偏極した電子 N = 6 個（全部アップスピン → 行列式は1個で済む）
# - 電子間相互作用  V_ee = λ Σ_{i<j} 1/√(|r_i − r_j|² + a²)
# 
# 最後の a はソフトコア長です。素の 1/r はカスプ（尖り）を持ち、それを正しく表現するには
# Jastrow 因子のカスプ条件が要るのですが、この notebook の主題は遮蔽なので、
# カスプを丸めて話を単純にしています。

# --- 系のパラメータ ---
N    = 6      # 電子数（スピン偏極）
D    = 2      # 空間次元
LAM  = 1.0    # 相互作用の強さ λ
SOFT = 0.3    # ソフトコア長 a
D_REF = 3.0   # 遮蔽窓 w の基準長（このトラップでの「大きめの電子間距離」）

def pair_dist(r):
    '''全ペアの距離 (B,N,N)。eps は対角（自分自身）で sqrt が微分不能になるのを防ぐ'''
    diff = r.unsqueeze(2) - r.unsqueeze(1)          # (B,N,N,D)
    return torch.sqrt((diff*diff).sum(-1) + 1e-12)

# 電子ごとの入力特徴量: (x, y, r²)。位置埋め込みは足さない
def features(r):
    return torch.cat([r, (r*r).sum(-1, keepdim=True)], dim=-1)   # (B,N,3)

r_demo = torch.randn(1, N, D)
print("電子配置 r        :", r_demo.shape, "= (バッチ, 電子数, 次元)")
print("特徴量            :", features(r_demo).shape)
print("ペア距離行列      :", pair_dist(r_demo).shape)
print("\nペア距離行列（対称・対角0）:")
print(pair_dist(r_demo)[0].round(decimals=2))

# ## Step 2: 土台 — 厳密に解ける自由フェルミオン状態
# 
# 相互作用を切った（λ = 0）ときの基底状態は厳密に分かります。
# 2次元調和振動子の準位は E = n_x + n_y + 1 で、下から6個を詰めると
# 
# (0,0), (1,0), (0,1), (2,0), (1,1), (0,2) → E = 1 + 2 + 2 + 3 + 3 + 3 = **14**
# 
# この状態は Slater 行列式 det[φ_a(r_i)] で厳密に書けます。
# 
# **この notebook の方針**: ニューラルネットに波動関数を丸ごと作らせるのではなく、
# この厳密な行列式を土台に固定し、**ネットには相関因子（Jastrow 因子）だけを学ばせます**。
# 
# $$\psi(R) = \underbrace{\det[\phi_a(r_i)]}_{\text{厳密な自由状態}} \times \exp\Big(\sum_i f_i(R)\Big),
# \qquad f_i = \text{遮蔽 Transformer の出力}$$
# 
# この形の利点:
# 
# - 反対称性は行列式が担保する（exp(...) は正なので符号を変えない）
# - ネットの出力を 0 に初期化すれば、**出発点が厳密な自由状態**になる
# - したがって学習の進み具合が「厳密解からどれだけ下げられたか」として読める
# 
# 言語モデルの notebook で「学習前 loss = ln 436 になれば実装は健全」という
# チェックがあったのと同じ役割を、この **E = 14 ちょうど**が果たします。

# --- 厳密な自由フェルミオン状態（Slater 行列式） ---
ORB = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2)][:N]
E_FREE = sum(nx + ny + 1 for nx, ny in ORB)

def hermite(n, x):
    return [torch.ones_like(x), 2*x, 4*x**2 - 2][n]

def log_slater(r):
    '''log|det[φ_a(r_i)]| （ガウス因子込み）と符号を返す'''
    x, y = r[..., 0], r[..., 1]
    M = torch.stack([hermite(nx, x)*hermite(ny, y) for nx, ny in ORB], dim=-1)  # (B,N,N)
    sign, logdet = torch.linalg.slogdet(M)
    return sign, logdet - 0.5*(r*r).sum((-1, -2))

print("占有する軌道 (n_x, n_y) :", ORB)
print("自由フェルミオンの厳密なエネルギー E =", E_FREE)

# 反対称性の確認: 電子0と電子1を入れ替えると符号が反転する
r = torch.randn(1, N, D)
r_swap = r.clone(); r_swap[:, [0,1]] = r_swap[:, [1,0]]
s1, l1 = log_slater(r); s2, l2 = log_slater(r_swap)
print(f"\n交換前: sign={s1.item():+.0f}, log|ψ|={l1.item():.6f}")
print(f"交換後: sign={s2.item():+.0f}, log|ψ|={l2.item():.6f}")
print(f"→ |ψ| は不変 (差 {abs(l1-l2).item():.2e})、符号は反転 → 反対称 ✓")

# ## Step 3: 局所エネルギー — ここが物理の心臓部
# 
# 変分モンテカルロでは、波動関数 ψ に対して**局所エネルギー**
# 
# $$E_L(R) = \frac{H\psi(R)}{\psi(R)}
# = -\tfrac12 \sum_i \Big( \nabla_i^2 \ln|\psi| + |\nabla_i \ln|\psi||^2 \Big)
# + V_{\rm trap}(R) + V_{ee}(R)$$
# 
# を配置 R について評価し、|ψ|² に従うサンプルで平均を取ります。すると
# 
# $$\langle E_L \rangle = \frac{\langle\psi|H|\psi\rangle}{\langle\psi|\psi\rangle} \ \ge\ E_0$$
# 
# となり、これを下げるのが学習です。**損失関数がエネルギーそのもの**で、教師データはありません。
# 
# `log|ψ|` の**2階微分**が必要なので、ネットワークは座標について2回微分可能でなければなりません。
# 遮蔽注意が Trim に ReLU の**二乗**を、Softmask に raised cosine を使うのは、まさにこの要請のためです。
# 
# $E_L$ のばらつき（locstd）も重要です。ψ が H の厳密な固有状態なら $E_L(R)$ は R によらず定数なので、
# **locstd → 0 が波動関数の質そのものの指標**になります。

# --- 局所エネルギー ---
def local_energy(logpsi_fn, r, lam=LAM):
    B = r.shape[0]
    rr = r.detach().clone().requires_grad_(True)
    lp = logpsi_fn(rr)                                        # (B,)
    # 1階微分 ∇ log|ψ|（2階微分のために計算グラフを残す）
    g = torch.autograd.grad(lp.sum(), rr, create_graph=True)[0].reshape(B, -1)
    # ラプラシアン = ヘッセ行列の対角の和。座標ごとに1回ずつ微分する
    lap = torch.zeros(B, dtype=r.dtype)
    for c in range(N*D):
        gc = torch.autograd.grad(g[:, c].sum(), rr, retain_graph=(c < N*D-1))[0]
        lap = lap + gc.reshape(B, -1)[:, c]
    kinetic = -0.5*(lap + (g*g).sum(-1))
    trap    = 0.5*(r*r).sum((-1, -2))
    d2      = ((r.unsqueeze(2) - r.unsqueeze(1))**2).sum(-1)
    vee     = lam*(1.0/torch.sqrt(d2 + SOFT**2)).triu(1).sum((-1, -2))
    return (kinetic + trap + vee).detach()

# ★健全性チェック★ 厳密な自由状態を入れて λ=0 にすれば、E_L はどの配置でも 14 のはず
r = torch.randn(256, N, D)
EL = local_energy(lambda rr: log_slater(rr)[1], r, lam=0.0)
print(f"厳密な自由状態, λ=0:  E_L = {EL.mean():.8f}  ± {EL.std():.2e}   (厳密値 {E_FREE})")
print("→ 平均が 14、ばらつきが 0 ならば、運動エネルギー・ラプラシアン・ポテンシャルの実装は健全")

EL1 = local_energy(lambda rr: log_slater(rr)[1], r, lam=LAM)
print(f"\n相互作用を入れると λ={LAM}: E_L = {EL1.mean():.4f} ± {EL1.std():.4f}")
print("→ もはや固有状態ではないのでばらつきが出る。これを学習で下げていく")

# ## Step 4: 比較対象 — softmax 注意
# 
# まず従来型を実装します。言語モデル版との違いは**因果マスクが無い**ことだけです。
# 
# $$s_{ij} = \frac{q_i \cdot k_j}{\sqrt{d}}, \qquad
# w_{ij} = \mathrm{softmax}_j(s_{ij}), \qquad \mathrm{out}_i = \sum_j w_{ij} v_j$$
# 
# softmax は分母 $\sum_j e^{s_{ij}}$ で割るので、**重みの和は必ず 1** になります。
# この「自動的な正規化」が後で効いてきます。

class SoftmaxAttention(nn.Module):
    def __init__(self, C, H, dh):
        super().__init__()
        self.H, self.dh = H, dh
        self.q = nn.Linear(C, H*dh, bias=False)
        self.k = nn.Linear(C, H*dh, bias=False)
        self.v = nn.Linear(C, H*dh, bias=False)
        self.o = nn.Linear(H*dh, C, bias=False)

    def weights(self, x, d=None):
        B, n, _ = x.shape
        q = self.q(x).view(B, n, self.H, self.dh)
        k = self.k(x).view(B, n, self.H, self.dh)
        s = torch.einsum('bihd,bjhd->bhij', q, k) * self.dh**-0.5
        return F.softmax(s, dim=-1)                 # 行ごとに和が1（因果マスクは無い）

    def forward(self, x, d):
        B, n, _ = x.shape
        v = self.v(x).view(B, n, self.H, self.dh)
        out = torch.einsum('bhij,bjhd->bihd', self.weights(x), v).reshape(B, n, -1)
        return self.o(out)

_a = SoftmaxAttention(C=32, H=4, dh=8)
_x = torch.randn(1, N, 32)
_w = _a.weights(_x)
print("注意行列の形 (B, ヘッド, i, j):", tuple(_w.shape))
print("ヘッド0の重み:")
print(_w[0,0].round(decimals=3))
print("各行の和 =", _w[0,0].sum(-1).round(decimals=6), " ← softmax なので必ず 1")

# ## Step 5: 遮蔽注意 — 7段階
# 
# ここが本題です。`CustomScreenformer.py` の `make_screening_attention` の中身をそのまま移植します。
# コメントの番号は実物のコードの `# [SCREEN] 1.` 〜 `7.` と一致しています。
# 
# | 段階 | 式 | 狙い |
# |---|---|---|
# | 1. 単位長正規化 | $\bar q = q/|q|$, $s_{ij} = \bar q_i\cdot\bar k_j$ | cos 類似度なので必ず $-1\le s\le 1$。$1/\sqrt{d}$ が不要になる |
# | 2. Trim | $\alpha_{ij} = \big(\mathrm{ReLU}(s_{ij}-(1-r))/r\big)^2$ | 似ていない対を**厳密に 0** にする。二乗なので境界で滑らか |
# | 3. Softmask | $m_{ij} = \tfrac12\big(1+\cos(\pi\, d_{ij}/w)\big)$ | 距離 $d\ge w$ を**厳密に 0** に。$d$ は物理的な電子間距離 |
# | 4. 分母なし集約 | $h_i = \sum_j \alpha_{ij} m_{ij} \bar v_j$ | softmax の分母を使わない |
# | 5. TanhNorm | $o = \tanh(|h|)\, h/|h|$ | 分母が無い代わりに、長さをここで抑える |
# | 6. ゲート | $o \leftarrow o \odot \tanh(\mathrm{SiLU}(W_G x))$ | 重い FFN の代わり。**電子ごとに独立**なので置換同変性を壊さない |
# | 7. 出力倍率 | $\Delta x = e^{s_O} W_O\, o$ | 分母を捨てた代償を埋める（Step 8 で詳述） |
# 
# 学習される「つまみ」は**ヘッドごと**に持ちます。制約付きの量を勾配法で扱うため、
# 制約のない実数を持って写像で押し込みます:
# 
# $$r_h = 2\,\sigma(\rho_h) \in (0,2), \qquad w_h = \mathrm{softplus}(\omega_h) > 0$$
# 
# **ヘッドごとに異なる $w_h$ を持つこと**が multiscreen（多重遮蔽）の実体です。

class ScreeningAttention(nn.Module):
    def __init__(self, C, H, dh, w_init, r_init=1.9):
        super().__init__()
        self.H, self.dh = H, dh
        self.q = nn.Linear(C, H*dh, bias=False)
        self.k = nn.Linear(C, H*dh, bias=False)
        self.v = nn.Linear(C, H*dh, bias=False)
        self.g = nn.Linear(C, H*dh, bias=False)      # ゲート用 W_G
        self.o = nn.Linear(H*dh, C, bias=False)      # 出力射影 W_O
        # 学習するつまみ（制約のない実数として持つ）
        self.rho   = nn.Parameter(torch.full((H,), math.log(r_init/(2-r_init))))
        self.omega = nn.Parameter(torch.tensor([math.log(math.expm1(w)) for w in w_init]))
        self.sO    = nn.Parameter(torch.zeros(()))   # 出力倍率（Step 8 で較正）

    @staticmethod
    def unit(x, eps=1e-6):
        '''単位長正規化。eps は原点で微分不能になるのを防ぐため（2階微分が要るので必須）'''
        return x / torch.sqrt((x*x).sum(-1, keepdim=True) + eps*eps)

    def screening_weights(self, x, d):
        B, n, _ = x.shape
        # [SCREEN] 1. 単位長正規化 → cos 類似度
        qb = self.unit(self.q(x).view(B, n, self.H, self.dh))
        kb = self.unit(self.k(x).view(B, n, self.H, self.dh))
        s  = torch.einsum('bihd,bjhd->bhij', qb, kb)          # 必ず -1 ≤ s ≤ 1
        # [SCREEN] 2. Trim（受容幅 r、ヘッドごとに学習）
        r  = (2*torch.sigmoid(self.rho)).view(1, self.H, 1, 1)
        alpha = (F.relu(s - (1 - r))/r)**2                    # s ≤ 1-r は厳密に 0
        # [SCREEN] 3. Softmask（遮蔽窓 w、ヘッドごとに学習。距離は物理的な電子間距離）
        w  = F.softplus(self.omega).view(1, self.H, 1, 1)
        t  = torch.clamp(d.unsqueeze(1)/w, max=1.0)
        m  = 0.5*(1 + torch.cos(math.pi*t))                   # d ≥ w は厳密に 0
        return alpha*m

    def forward(self, x, d):
        B, n, _ = x.shape
        wgt = self.screening_weights(x, d)
        vb  = self.unit(self.v(x).view(B, n, self.H, self.dh))
        # [SCREEN] 4. 分母なし集約（softmax の分母を使わない）
        h = torch.einsum('bhij,bjhd->bihd', wgt, vb)
        # [SCREEN] 5. TanhNorm（方向はそのまま、長さだけ抑える）
        nrm = torch.sqrt((h*h).sum(-1, keepdim=True) + 1e-12)
        h = h*torch.tanh(nrm)/nrm
        h = h.reshape(B, n, -1)
        # [SCREEN] 6. ゲート（電子ごとに独立 → 置換同変性を壊さない）
        h = h*torch.tanh(F.silu(self.g(x)))
        # [SCREEN] 7. 出力射影 と 学習可能な倍率
        return torch.exp(self.sO)*self.o(h)

# ヘッドごとに異なる初期遮蔽長（multiscreen の初期化）
H = 4
W_INIT = [D_REF*0.8*2.5**((2*i/(H-1)) - 1) for i in range(H)]
print("ヘッドごとの初期遮蔽長 w :", [round(w, 3) for w in W_INIT])
print("（同じ値で始めるとヘッドが対称のままで、遮蔽長が動く動機が生まれない）")

# ### Trim と Softmask を絵で見る
# 
# 2つの窓関数がどんな形をしているかを描きます。ポイントは**厳密に 0 になる領域があること**と、
# **その境界が滑らかであること**（値も1階微分も 0 で繋がる）です。
# 後者は局所エネルギーにラプラシアンが要るための必須条件です。

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))

s = torch.linspace(-1, 1, 400)
for r_ in [0.5, 1.0, 1.5, 1.9]:
    ax[0].plot(s, (F.relu(s - (1 - r_))/r_)**2, label=f"r = {r_}")
ax[0].set_xlabel("cosine similarity  s"); ax[0].set_ylabel(r"Trim  $\alpha$")
ax[0].set_title("Trim: dissimilar pairs are exactly zero"); ax[0].legend(); ax[0].grid(alpha=.3)

dd = torch.linspace(0, 8, 400)
for w_ in W_INIT:
    m = 0.5*(1 + torch.cos(math.pi*torch.clamp(dd/w_, max=1.0)))
    ax[1].plot(dd, m, label=f"w = {w_:.2f}")
ax[1].set_xlabel("electron-electron distance  d"); ax[1].set_ylabel("Softmask  m")
ax[1].set_title("Softmask: beyond w, exactly zero (multiscreen)"); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.show()

# ## Step 6: 重み行列を見比べる
# 
# 同じ電子配置に対して、softmax の重みと遮蔽の重みを並べて描きます。
# **遮蔽の方には真っ黒（＝厳密に 0）のマスが現れる**のが要点です。
# softmax はどんなに小さくても必ず正の重みを与えるので、真っ黒にはなりません。

torch.manual_seed(3)
r1 = torch.randn(1, N, D)*1.3
d1 = pair_dist(r1)
x1 = nn.Linear(3, 32, bias=False)(features(r1))

sm = SoftmaxAttention(32, H, 8)
sc = ScreeningAttention(32, H, 8, w_init=W_INIT)
w_sm = sm.weights(x1)[0].detach()
w_sc = sc.screening_weights(x1, d1)[0].detach()

fig = plt.figure(figsize=(13, 5.6))
gs = fig.add_gridspec(2, 5, width_ratios=[1.25, 1, 1, 1, 1])

axp = fig.add_subplot(gs[:, 0])
axp.scatter(r1[0,:,0], r1[0,:,1], s=260, c="#2a78d6", zorder=3)
for i in range(N):
    axp.annotate(str(i), (r1[0,i,0], r1[0,i,1]), color="w", ha="center", va="center", zorder=4)
cir = plt.Circle((float(r1[0,0,0]), float(r1[0,0,1])), W_INIT[0], fill=False,
                 ls="--", color="#2e8b57", lw=1.6)
axp.add_patch(cir)
axp.set_aspect("equal"); axp.grid(alpha=.3)
axp.set_title(f"positions\n(dashed: head 0 window\n w={W_INIT[0]:.2f} around e0)", fontsize=9)

for h in range(H):
    a1 = fig.add_subplot(gs[0, h+1]); a1.imshow(w_sm[h], cmap="viridis", vmin=0)
    a1.set_title(f"softmax  head {h}", fontsize=9); a1.set_xticks(range(N)); a1.set_yticks(range(N))
    a2 = fig.add_subplot(gs[1, h+1]); a2.imshow(w_sc[h], cmap="viridis", vmin=0)
    a2.set_title(f"screening  head {h}", fontsize=9); a2.set_xticks(range(N)); a2.set_yticks(range(N))
plt.tight_layout(); plt.show()

print(f"softmax   の重みのうち厳密に 0 の割合 : {float((w_sm == 0).double().mean()):.1%}")
print(f"screening の重みのうち厳密に 0 の割合 : {float((w_sc == 0).double().mean()):.1%}")
print("\n距離行列（どの対が窓の外にいるか）:")
print(d1[0].round(decimals=2))

# ## Step 7: 波動関数を組み立てる
# 
# $$\log|\psi(R)| = \underbrace{\log|\det[\phi_a(r_i)]| - \tfrac12\sum_i |r_i|^2}_{\text{固定：厳密な自由状態}}
# \;+\; \underbrace{\sum_i f_i(R)}_{\text{Transformer が学ぶ相関因子}}$$
# 
# Transformer 部分は言語モデルと同じ骨格です:
# 
# ```
# 特徴量 → 埋め込み → [注意 → 残差 → LayerNorm → MLP → 残差 → LayerNorm] × L → 出力
# ```
# 
# 違うのは**位置埋め込みが無い**ことと、注意に**電子間距離 d を渡す**こと。
# 最終層の重みを 0 に初期化してあるので、初期状態では $f = 0$、つまり厳密な自由状態から出発します。

class Wave(nn.Module):
    def __init__(self, kind="screen", C=32, H=4, dh=8, L=2):
        super().__init__()
        self.embed = nn.Linear(3, C, bias=False)       # 位置埋め込みは足さない
        mk = (lambda: SoftmaxAttention(C, H, dh)) if kind == "softmax" else \
             (lambda: ScreeningAttention(C, H, dh, w_init=W_INIT))
        self.att = nn.ModuleList(mk() for _ in range(L))
        self.mlp = nn.ModuleList(nn.Sequential(nn.Linear(C, C), nn.Tanh()) for _ in range(L))
        self.ln1 = nn.ModuleList(nn.LayerNorm(C) for _ in range(L))
        self.ln2 = nn.ModuleList(nn.LayerNorm(C) for _ in range(L))
        self.out = nn.Linear(C, 1, bias=False)
        nn.init.zeros_(self.out.weight)                # f = 0 → 厳密な自由状態から出発

    def jastrow(self, r):
        d = pair_dist(r)
        x = self.embed(features(r))
        for a, m, l1, l2 in zip(self.att, self.mlp, self.ln1, self.ln2):
            x = l1(x + a(x, d))       # 注意 + 残差 + LayerNorm
            x = l2(x + m(x))          # MLP  + 残差 + LayerNorm
        return self.out(x).squeeze(-1).sum(-1)         # Σ_i f_i（電子について和 → 対称）

    def forward(self, r):
        return log_slater(r)[1] + self.jastrow(r)

model = Wave("screen")
print("パラメータ数 :", sum(p.numel() for p in model.parameters()))

# 反対称性の確認（学習後も保たれる：交換で行列式の符号だけが変わり、Jastrow は不変）
r = torch.randn(1, N, D)
r_sw = r.clone(); r_sw[:, [0,1]] = r_sw[:, [1,0]]
print(f"交換で log|ψ| が不変か : 差 = {abs(model(r) - model(r_sw)).item():.2e}")
print(f"Jastrow が対称か       : 差 = {abs(model.jastrow(r) - model.jastrow(r_sw)).item():.2e}")

# 初期状態のエネルギー（λ=0 なら厳密に 14 のはず）
r = torch.randn(256, N, D)
EL0 = local_energy(model, r, lam=0.0)
print(f"\n初期状態 (f=0), λ=0 : E_L = {EL0.mean():.8f} ± {EL0.std():.2e}  (厳密値 {E_FREE})")

# ## Step 8: 出力倍率の較正 — 分母を捨てた代償
# 
# ここが遮蔽注意でいちばん間違えやすい所です。
# 
# softmax は分母 $\sum_j e^{s_{ij}}$ で割るので、**重みの和が必ず 1** になり、出力の大きさが自動的に揃います。
# 遮蔽注意はその分母を捨てるので、**出力の大きさが誰にも保証されません**。
# 
# 初期状態では $q$ と $k$ はランダムな方向なので、cos 類似度は $1/\sqrt{d_{\rm head}}$ 程度しかありません。
# Trim はそれを二乗するので、重みは非常に小さくなります。結果として遮蔽ブランチの出力は
# 残差ストリームに比べて桁違いに小さくなり、**注意が実質的に効かないまま学習が進みます**
# （層が「MLP だけ」に退化する）。
# 
# 対策は、教科書のマップにある学習可能スカラー $e^{s_O}$ の初期値を**実測して合わせる**ことです。
# 下のセルは、初期状態で両者の大きさの比を測り、softmax と同じ水準から出発するように $s_O$ を決めます。

@torch.no_grad()
def branch_ratio(model, r):
    '''各層で |注意の出力| / |残差ストリーム| を測る'''
    d = pair_dist(r); x = model.embed(features(r)); out_ratios = []
    for a, m, l1, l2 in zip(model.att, model.mlp, model.ln1, model.ln2):
        out = a(x, d)
        out_ratios.append(float(out.norm(dim=-1).mean()/x.norm(dim=-1).mean()))
        x = l1(x + out); x = l2(x + m(x))
    return out_ratios

@torch.no_grad()
def calibrate(model, r, target=0.35):
    '''遮蔽ブランチが残差ストリームの target 倍の大きさから始まるように s_O を決める'''
    d = pair_dist(r); x = model.embed(features(r))
    for a, m, l1, l2 in zip(model.att, model.mlp, model.ln1, model.ln2):
        out = a(x, d)
        if hasattr(a, 'sO'):
            ratio = max(float(out.norm(dim=-1).mean()/x.norm(dim=-1).mean()), 1e-12)
            a.sO += math.log(target/ratio)
            out = a(x, d)
        x = l1(x + out); x = l2(x + m(x))
    return model

r_cal = torch.randn(512, N, D)
torch.manual_seed(0); m_sm = Wave("softmax")
torch.manual_seed(0); m_sc = Wave("screen")

print("較正前:")
print("  softmax   の |注意| / |残差| =", [round(v, 3) for v in branch_ratio(m_sm, r_cal)])
print("  screening の |注意| / |残差| =", [round(v, 3) for v in branch_ratio(m_sc, r_cal)],
      "← 桁違いに小さい")
calibrate(m_sc, r_cal, target=0.35)
print("\n較正後:")
print("  screening の |注意| / |残差| =", [round(v, 3) for v in branch_ratio(m_sc, r_cal)])
print("  設定された倍率 e^{s_O}       =", [round(float(torch.exp(a.sO.detach())), 2) for a in m_sc.att])

# ## Step 9: VMC で学習する
# 
# 1ステップの中身:
# 
# 1. **サンプリング** — メトロポリス法で $|\psi|^2$ に従う電子配置を更新
# 2. **局所エネルギー** — 各配置で $E_L(R)$ を計算（ラプラシアンが要る）
# 3. **勾配** — エネルギーの微分は
#    $\partial_\theta E = 2\,\big\langle (E_L - \langle E_L\rangle)\, \partial_\theta \log|\psi| \big\rangle$
#    という形になるので、$E_L$ 自体を微分する必要はありません（3階微分を避けられる）
# 4. **更新** — Adam
# 
# なお本物のコードは Adam ではなく **KFAC**（自然勾配法）を使い、ウォーカーも 1024 個です。
# この notebook は 128 個 + Adam なので、$E_L$ の裾のばらつきは大きめに出ます。

@torch.no_grad()
def metropolis(model, r, step, nsweep=8):
    lp = 2*model(r); acc = 0.0
    for _ in range(nsweep):
        prop = r + step*torch.randn_like(r)
        lpn  = 2*model(prop)
        take = torch.rand(r.shape[0]) < torch.exp(lpn - lp)
        r  = torch.where(take[:, None, None], prop, r)
        lp = torch.where(take, lpn, lp)
        acc += take.double().mean().item()/nsweep
    return r, acc

def train(kind, steps=150, B=96, seed=0, lr=3e-3):
    torch.manual_seed(seed)
    model = Wave(kind)
    calibrate(model, torch.randn(512, N, D), target=0.35)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    r, sw, hist, t0 = torch.randn(B, N, D), 0.5, [], time.time()
    for it in range(steps):
        r, acc = metropolis(model, r, sw)
        sw *= 1.1 if acc > 0.6 else (0.9 if acc < 0.4 else 1.0)   # 採択率を 0.4-0.6 に保つ
        EL = local_energy(model, r)
        # 外れ値は勾配にだけクリップをかける（報告する平均はクリップしない）
        med = EL.median(); mad = (EL - med).abs().mean()
        ELc = EL.clamp(med - 5*mad, med + 5*mad)
        loss = 2*((ELc - ELc.mean()).detach()*model(r)).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        hist.append((float(EL.mean()), float(EL.std()), acc))
        if it % 30 == 0 or it == steps - 1:
            print(f"  {kind:9s} step {it:3d}   E = {EL.mean():8.4f}   locstd = {EL.std():7.4f}"
                  f"   acc = {acc:.2f}   ({time.time()-t0:.0f}s)")
    return model, hist

print(f"自由フェルミオン（相互作用なし）の厳密値 E = {E_FREE}\n")
models, hists = {}, {}
for kind in ("softmax", "screen"):
    models[kind], hists[kind] = train(kind)

import numpy as np
print(f"{'':10s}  {'E (最後30ステップ)':>20s}  {'locstd (中央値)':>16s}")
for kind in ("softmax", "screen"):
    h = np.array(hists[kind][-30:])
    print(f"{kind:10s}  {h[:,0].mean():20.4f}  {np.median(h[:,1]):16.4f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for kind, c in (("softmax", "tab:blue"), ("screen", "tab:orange")):
    h = np.array(hists[kind])
    ax[0].plot(h[:,0], color=c, alpha=.45, lw=.9)
    ax[0].plot(np.convolve(h[:,0], np.ones(15)/15, mode="valid"), color=c, lw=2, label=kind)
    ax[1].plot(h[:,1], color=c, lw=1, label=kind)
ax[0].axhline(E_FREE, ls=":", c="k"); ax[0].text(2, E_FREE+.2, "free fermions (no interaction)", fontsize=8)
ax[0].set_xlabel("step"); ax[0].set_ylabel("energy"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[0].set_title("VMC energy")
ax[1].set_yscale("log"); ax[1].set_xlabel("step"); ax[1].set_ylabel("locstd")
ax[1].set_title("spread of the local energy\n(0 = exact eigenstate)"); ax[1].legend(); ax[1].grid(alpha=.3, which="both")
plt.tight_layout(); plt.show()

# ## Step 10: 学習された遮蔽長を見る
# 
# 遮蔽注意の面白いところは、**学習結果が物理的に読めること**です。
# ヘッドごとの遮蔽窓 $w_h$ は「そのヘッドが何単位長先まで電子を見ているか」を意味します。
# 
# 実際の電子間距離の分布と重ねて描くと、どのヘッドが近傍だけを、どのヘッドが全体を見ているかが分かります。

m = models["screen"]
print("学習された遮蔽長 w（ヘッドごと）:")
for i, a in enumerate(m.att):
    w  = F.softplus(a.omega.detach()).numpy()
    r_ = (2*torch.sigmoid(a.rho.detach())).numpy()
    print(f"  層 {i}:  w = {np.round(w,3)}   (初期値 {np.round(W_INIT,3)})")
    print(f"          r = {np.round(r_,3)}   e^(s_O) = {float(torch.exp(a.sO.detach())):.2f}")

# 学習後の配置から電子間距離の分布を作り、窓と重ねる
r = torch.randn(512, N, D); sw = 0.5
for _ in range(60):
    r, acc = metropolis(m, r, sw); sw *= 1.1 if acc > 0.6 else (0.9 if acc < 0.4 else 1.0)
d = pair_dist(r); dv = d[d > 1e-3].detach().numpy()

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].hist(dv, bins=60, density=True, color="#c9d6e5", edgecolor="#7b8ea3")
cols = plt.cm.viridis(np.linspace(0, .85, H))
for h_ in range(H):
    w_ = float(F.softplus(m.att[0].omega.detach())[h_])
    ax[0].axvline(w_, color=cols[h_], ls="--", lw=1.8, label=f"head {h_}:  w = {w_:.2f}")
ax[0].set_xlabel("electron-electron distance"); ax[0].set_ylabel("density")
ax[0].set_title("learned screening windows vs the actual pair distances")
ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

xg = np.linspace(0, dv.max()*1.05, 300)
for h_ in range(H):
    w_ = float(F.softplus(m.att[0].omega.detach())[h_])
    ax[1].plot(xg, 0.5*(1 + np.cos(np.pi*np.clip(xg/w_, 0, 1))), color=cols[h_], lw=1.8,
               label=f"head {h_}")
ax[1].set_xlabel("electron-electron distance"); ax[1].set_ylabel("Softmask m")
ax[1].set_title("multiscreen: each head owns a length scale")
ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.show()

frac = [(dv < float(F.softplus(m.att[0].omega.detach())[h_])).mean() for h_ in range(H)]
for h_ in range(H):
    print(f"head {h_}: 窓の中に入るペアの割合 = {frac[h_]:.1%}")

# ## まとめ — 実物のコードとの対応
# 
# この notebook で書いた `ScreeningAttention` は、`periodicwave/CustomScreenformer.py` の
# `make_screening_attention` とほぼ1対1に対応しています。
# 
# | この notebook | 実物 `CustomScreenformer.py` |
# |---|---|
# | `unit()` | `_unit_length` |
# | `screening_weights` の `alpha` | `# [SCREEN] 2.` の `alpha` |
# | `screening_weights` の `m` | `# [SCREEN] 3.` の Softmask |
# | `torch.einsum('bhij,bjhd->bihd', wgt, vb)` | `# [SCREEN] 4.` の分母なし集約 |
# | `h*torch.tanh(nrm)/nrm` | `# [SCREEN] 5.` の TanhNorm |
# | `torch.tanh(F.silu(self.g(x)))` | `# [SCREEN] 6.` のゲート |
# | `torch.exp(self.sO)*self.o(h)` | `# [SCREEN] 7.` の出力倍率 |
# | `W_INIT` の幾何級数 | `np.geomspace(...)` による multiscreen 初期化 |
# | `calibrate()` | 自動較正（`_alpha_mean` を数値積分して求める版） |
# | `pair_dist(r)` | `d_ee = r_ee[..., 0]`（周期ノルム） |
# 
# **簡素化した点**（実物との違い）:
# 
# | | この notebook | 実物 |
# |---|---|---|
# | 系 | 調和トラップ、開いた境界 | 周期境界の2次元電子ガス |
# | 相互作用 | ソフトコア Coulomb | 素の Coulomb + Ewald 和 |
# | 距離 | 普通のユークリッド距離 | 周期ノルム |
# | 波動関数 | 固定 Slater 行列式 × ニューラル Jastrow | ネットが軌道ごと作る |
# | 最適化 | Adam | KFAC（自然勾配） |
# | ウォーカー数 | 96 | 1024 |
# 
# 遮蔽注意そのもの（7段階と学習されるつまみ）は**実物と同じ**です。
# 
# **次に試すと面白いこと**
# 
# - `W_INIT` を全ヘッド同じ値にして、遮蔽長が動かなくなることを確かめる
# - Step 8 の較正を飛ばして（`target` を極端に小さくして）、注意が効かなくなる様子を見る
# - `ScreeningAttention.forward` から Softmask を外し、距離の情報が消えると何が起きるか見る
# - `LAM` を大きくして相関を強め、遮蔽長がどう変わるか調べる
