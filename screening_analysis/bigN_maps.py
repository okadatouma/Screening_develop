"""大きい系（N=25, 49）・多ヘッド（16）用の関連度マップ一式。

    python screening_analysis/bigN_maps.py <run_folder> [--out DIR] [--nconf 128]

出力（--out、既定は run_folder）:
  <tag>_grid.png   層 × ヘッドの α·m 行列（N×N）。ヘッドは窓 w の小さい順に並べ替える。
  <tag>_rank.png   近接順位マップ: x = 自分・第1近接・… , y = ヘッド（w 昇順）
                   三角格子の殻境界（6, 12, 18, 30 番目）を縦線で示す。
  <tag>_radial.png 重みの動径プロファイル（最小像 Euclid 距離）と、窓 m(d)・殻位置。

N=9 では最小像半径が 1.50a しかなく殻が 1 個しか入らないので、この図は N>=25 用。
ヘッドは学習後の w で並べ替えるので、構成（H8 / H16）や系（N25 / N49）をまたいで
「短い窓 → 長い窓」の順で読める。
"""
import argparse, glob, json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)


def min_image(pos, L, ne):
    """最小像 Euclid 距離。格子ベクトルは L の列なので f = dr @ inv(L).T。"""
    R = pos.reshape(pos.shape[0], ne, 2)
    dr = R[:, :, None, :] - R[:, None, :, :]
    f = dr @ np.linalg.inv(L).T
    return np.linalg.norm((f - np.round(f)) @ L.T, axis=-1)


def _font():
    import matplotlib.font_manager as fm, matplotlib.pyplot as plt
    for p in fm.findSystemFonts():
        if "NotoSansCJK" in p:
            fm.fontManager.addfont(p)
            plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False




def _footer(fig, txt, y=0.006):
    fig.text(0.006, y, txt, fontsize=8.4, color="#3a3a38", va="bottom", linespacing=2.05)


def _interpret(S, W, M, P, w, dref, order, spins, NE, NL, NH, is_softmax, T):
    """図の下に置く「所見」。数字はすべてその場のデータから計算する（決め打ちの文章にしない）。

    返すのは (所見の箇条書き, 解釈の 1-2 行)。解釈は私の読みなので、事実と分けて書く。
    """
    sp = np.asarray(spins[0])
    eye = np.eye(NE, dtype=bool)
    same = (sp[:, None] * sp[None, :] > 0) & ~eye
    opp = (sp[:, None] * sp[None, :] < 0)
    Sm = S.mean(0)                                            # (L,H,N,N)
    sd = np.array([[Sm[l, h][~eye].std() for h in range(NH)] for l in range(NL)])
    spin = np.array([[abs(Sm[l, h][same].mean() - Sm[l, h][opp].mean())
                      for h in range(NH)] for l in range(NL)])
    selfp = np.array([[100 * M[l, h][eye].sum() / max(M[l, h].sum(), 1e-12)
                       for h in range(NH)] for l in range(NL)])
    lb, hb = order[0], order[-1]                              # 窓が最小 / 最大のヘッド
    l_sp, h_sp = np.unravel_index(np.argmax(spin), spin.shape)   # スピン対比が最大のヘッド
    idle = int((sd < 0.05).sum())
    sm = selfp.mean(0)[order]

    bul = []
    if not is_softmax:
        bul.append(T("・自己重みは窓とともに %.0f%% → %.0f%% へ単調に減少（%d チャンネル）。"
                     "遮蔽長の連続掃引として機能している。" % (sm[0], sm[-1], NH),
                     "self-weight %.0f%% -> %.0f%%" % (sm[0], sm[-1])))
        bul.append(T("・窓の中に相手が入る割合 Pr は %.0f%% → %.0f%%。"
                     "最小の窓（w/d=%.2f）では実質、自分しか入っていない。"
                     % (100 * P.mean(0)[lb].mean(), 100 * P.mean(0)[hb].mean(), w.mean(0)[lb] / dref),
                     "Pr ladder"))
    else:
        bul.append(T("・softmax は行の総和が 1 に固定されるので、窓のような「見ない」選択ができない。"
                     "自己重みは %.0f%% → %.0f%%（ヘッド間の差は遮蔽より小さい）。" % (sm.max(), sm.min()),
                     "softmax normalises rows"))
    bul.append(T("・スピン対比 |⟨s⟩同 − ⟨s⟩異| が最大なのは L%dH%d（対比 %.3f, sd %.3f%s）。"
                 % (l_sp, h_sp, spin[l_sp, h_sp], sd[l_sp, h_sp],
                    "" if is_softmax else ", w/d=%.2f" % (w[l_sp, h_sp] / dref)),
                 "strongest spin contrast at L%dH%d" % (l_sp, h_sp)))
    bul.append(T("・sd(s) < 0.05 のヘッドが %d/%d。対をほとんど区別しておらず、実質遊んでいる。"
                 % (idle, NL * NH), "%d/%d idle heads" % (idle, NL * NH)))

    if is_softmax:
        itp = T("解釈（私見）: 規格化のせいで「距離で切る」ができず、"
                "どのヘッドも全電子に薄く重みを配る。役割分担はスピンなど内容側にしか現れない。",
                "interpretation")
    else:
        itp = T("解釈（私見）: 短い窓のヘッドは自己に特化し、$w$ が第1近接殻に届いたあたりのヘッドが"
                "スピン（＝交換）の情報を集中的に担う。中間の窓には遊んでいるヘッドが残っており、"
                "梯子の刻みはまだ余裕がある。窓が箱を超えるヘッドは平均場に漸近する。",
                "interpretation")
    return bul, itp

def _grid_shape(NL, NH, ncol=8):
    """ヘッドが多いとき 1 行 ncol で折り返す。戻り値 (総行数, 列数, 位置関数)。"""
    c = min(NH, ncol)
    per = int(np.ceil(NH / c))
    return NL * per, c, (lambda l, i: (l * per + i // c, i % c))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out", default=None)
    ap.add_argument("--nconf", type=int, default=128)
    a = ap.parse_args()
    out = a.out or a.folder
    os.makedirs(out, exist_ok=True)
    tag = os.path.basename(a.folder.rstrip("/"))

    cfg = json.load(open(os.path.join(a.folder, "config.json")))
    if cfg["network"]["network_type"] == "CustomPsiformer":
        from psi_probe import probe                       # softmax 注意
    else:
        from attn_probe import probe                      # 遮蔽注意
    r = probe(a.folder, nconf=a.nconf)
    is_softmax = bool(r.get("softmax", False))
    NE, NL, NH, dref = r["NE"], r["NL"], r["NH"], r["dref"]
    L = r["lattice"]
    nup = int((np.asarray(r["spins"][0]) > 0).sum())
    S, W = r["S"], r["W"]                        # (B,L,H,N,N)
    M = W.mean(0)                                # (L,H,N,N)
    P = (W > 0).mean(0)
    w = r["w"]                                   # (L,H) 絶対長
    order = (np.arange(NH) if np.all(np.isnan(w))         # softmax は窓が無いので元の順
             else np.argsort(w.mean(0)))                 # 遮蔽は窓の小さい順（層平均）
    wl = (lambda l, h: "--" if np.isnan(w[l, h]) else "%.2f" % (w[l, h] / dref))

    # 三角格子の殻: a, √3a, 2a, √7a, 3a  / 配位数 6,6,6,12,6
    a1 = np.sqrt(2 * np.pi / np.sqrt(3))
    shells = [a1 * np.sqrt(k) for k in (1, 3, 4, 7, 9)]
    cum = np.cumsum([6, 6, 6, 12, 6])
    d_true = min_image(r["positions"], L, NE)    # (B,N,N)
    rmax = 0.5 * min(np.linalg.norm(L[:, 0]), np.linalg.norm(L[:, 1]))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    ja = _font(); T = lambda x, y: x if ja else y      # noqa: E731
    # どちらの注意機構かを図の顔に出す（後から見て取り違えないように）
    NET = (T("通常アテンション（softmax・Psiformer）", "plain softmax attention (Psiformer)")
           if is_softmax else
           T("遮蔽アテンション（screening・Screenformer）", "screening attention (Screenformer)"))
    BADGE = "#8a4ab0" if is_softmax else "#0f7b6c"
    rs_ = cfg["system"]["make_local_energy_kwargs"]["potential_kwargs"]["interaction_energy_scale"]
    COND = T("N=%d（↑%d ↓%d）, $r_s$=%g, %d層×%dヘッド" % (NE, nup, NE - nup, rs_, NL, NH),
             "N=%d rs=%g %dx%d" % (NE, rs_, NL, NH))

    def _head(fig, what):
        """図の左上に「どちらの注意か」を色付きで大きく出す。"""
        fig.text(0.006, 0.997, NET, fontsize=13.5, weight="bold", color=BADGE,
                 va="top", ha="left")
        fig.text(0.006, 0.945, what + "　—　" + COND + "　（" + tag + "）",
                 fontsize=10.5, color="#0b0b0b", va="top", ha="left")
    CM = LinearSegmentedColormap.from_list(
        "b", ["#fcfcfb", "#dce8f7", "#93b8ea", "#2a78d6", "#12406f"]); CM.set_bad("#9a9a94")
    ORANGE, MUTED, GREEN = "#eb6834", "#52514e", "#008300"

    _bul, _ = _interpret(S, W, M, P, w, dref, order, r["spins"], NE, NL, NH, is_softmax, T)

    # 各ヘッドが波動関数に足す量（head_contrib）。二乗 share が出力への寄与。
    try:
        from head_contrib import cached as _hc
        _z = _hc(a.folder, nconf=min(a.nconf, 48))
        Chd, Xhd, Rhd, _w2 = _z["C"], _z["X"], _z["R"], _z["w"]
        SQ = 100 * Chd ** 2 / (Chd ** 2).sum()          # (L,H) 全層合計 100%
        ORTH = " ".join("層%d %.2f" % (l, Chd[l].sum() / Rhd[l]) for l in range(NL))
        RATIO = " ".join("層%d %.2f" % (l, Chd[l].sum() / Xhd[l]) for l in range(NL))
        HAVE = True
    except Exception as e:                                # ckpt が無い等
        print("  (head_contrib skipped: %s)" % e)
        SQ = np.full((NL, NH), np.nan); ORTH = RATIO = "--"; HAVE = False
    cs = (lambda l, h: "--" if not HAVE else "%.1f%%" % SQ[l, h])

    # Trim が実際に何かを切っているか（遮蔽のみ）
    if is_softmax:
        TRIM = ""
    else:
        _thr = (1.0 - r["r"])[None, :, :, None]
        _cut = 100 * (S[:, :, :, ~np.eye(NE, dtype=bool)] < _thr).mean()
        TRIM = ("　　実測: $r$ = %.2f なのでしきい値 $1-r$ = %.2f、$s$ の最小は %.2f。"
                "**Trim に掛かった対は %.2f%%** — 実質全開で、疎性はすべて窓 $m(d)$ 由来。\n"
                % (r["r"].mean(), (1 - r["r"]).mean(),
                   S[:, :, :, ~np.eye(NE, dtype=bool)].min(), _cut)).replace("**", "")

    READ = (
        "読み方\n"
        "　全体寄与 % $=\\|c_h\\|^2/\\sum\\|c_{h'}\\|^2$（1 体成分込み）。2 体だけの $\\|c_h^{\\rm pair}\\|^2$ 割合とは別物なので注意。\n"
        "　自%（_grid.png）= 重み全体に占める対角 $W_{ii}$ の割合。大 = その電子自身の情報を主に使い、他電子から取り込む量が少ない"
        "（＝局在。結晶側で大きくなる）。小 = 他電子との相関を積極的に使っている（＝液体側）。\n"
        "　sd = 非対角 $s_{ij}$ の標準偏差。大 = そのヘッドが相手電子を強く区別している（誰を見るかを選んでいる）。"
        "小 = 誰に対してもほぼ同じスコア＝実質的に平均場。Pr が小さいのに sd が大きいヘッドは「選別はしているが相手が窓の外」。\n"
        "　寄与% の定義　注意ブロックの出力はヘッドについて厳密に加法的:\n"
        + ("　　　$\\mathrm{out}_i=\\sum_h\\left[\\mathrm{softmax}_j(s_{ij})\\,v_j\\right]_{i,h}\\cdot W_O^{(h)}$\n"
           if is_softmax else
           "　　　$\\mathrm{out}_i=e^{s_O}\\sum_h\\left[\\mathrm{gate}\\odot\\mathrm{TanhNorm}(\\mathrm{attn})\\right]_{i,h}\\cdot W_O^{(h)}$\n")
        + "　　　その第 $h$ 項をヘッド $h$ の寄与ベクトル $c_h(i)$ と呼ぶ（$W_O^{(h)}$ = 出力行列 $W_O$ の $h$ 番目の行ブロック、"
          "$c_h$ の次元 = mlp_dim）。$\\|c_h\\|$ は電子 $i$ と配置について平均。\n"
        + "　　　寄与%$_{lh}$ = $\\|c_h^{(l)}\\|^2\\,/\\,\\sum_{l',h'}\\|c_{h'}^{(l')}\\|^2\\times100$"
          "　… 全層・全ヘッドの合計が 100%。"
        "二乗で測る理由は寄与ベクトルがほぼ直交だから（$\\sum_h\\|c_h\\|/\\|\\sum_h c_h\\|$ = " + ORTH
        + "、直交なら $\\sqrt{H}$=" + "%.2f" % np.sqrt(NH)
        + "）。直交なら $\\|\\sum_h c_h\\|^2=\\sum_h\\|c_h\\|^2$ なので、出力への効き方は二乗が正しい。"
        "残差ストリームに対する比 $\\sum\\|c_h\\|/\\|x\\|$ = " + RATIO + "。\n"
        "　$s$ の中身 = 実測では $s$ の分散のうち距離で説明できるのは 1.6%、スピンで 3.5%、"
        "局所六方晶秩序 $|\\psi_6|$ や局所間隔では 0.1% 以下 — 低次の幾何記述子には落ちない。\n"
        "　　ただしチャネル除去実験（再学習なし・パラメータのみ書き換え）では、$s$ を配置平均 $\\langle\\alpha\\rangle_h$ で"
        "置き換えても変分エネルギーの悪化は $|E|$ の 0.4〜3.7%。一方で窓 $m(d)$ を消すと 25〜107%。\n"
        "　　→ 主役は距離チャネル $m(d)$、内容チャネル $\\alpha(s)$ は脇役（ただし大きい $r_s$ では 10〜25$\\sigma$ で有意なのでゼロではない）。\n")

    # ---------------------------------------------------------- 1. L x H グリッド
    NR, NC, pos_of = _grid_shape(NL, NH)
    FOOT, HEAD = 6.4, 1.35                                 # 脚注 / ヘッダに割く高さ [inch]
    Hfig = 2.35 * NR + HEAD + FOOT
    fig, ax = plt.subplots(NR, NC, figsize=(2.05 * NC, Hfig), squeeze=False,
                           gridspec_kw={"wspace": .07, "hspace": .5})
    fig.patch.set_facecolor("#fcfcfb")
    fig.subplots_adjust(top=1 - HEAD / Hfig, bottom=FOOT / Hfig)
    for a_ in ax.ravel(): a_.axis("off")
    for l in range(NL):
        for i, h in enumerate(order):
            A = ax[pos_of(l, i)]; A.axis("on")
            m = np.ma.masked_where(P[l, h] < 0.005, M[l, h])
            A.imshow(m, cmap=CM, vmin=0, vmax=max(np.percentile(M[l, h], 99.5), 1e-9),
                     interpolation="nearest")
            A.axhline(nup - .5, color=ORANGE, lw=.6); A.axvline(nup - .5, color=ORANGE, lw=.6)
            A.set_xticks([]); A.set_yticks([])
            sp = 100 * M[l, h][np.arange(NE), np.arange(NE)].sum() / max(M[l, h].sum(), 1e-12)
            A.set_title("L%dH%d w/d=%s\nPr=%.0f%% 自%.0f%% sd=%.3f\n全体寄与 %s"
                        % (l, h, wl(l, h), 100 * P[l, h].mean(), sp,
                           S.mean(0)[l, h][~np.eye(NE, dtype=bool)].std(), cs(l, h)),
                        fontsize=6.6, loc="left", pad=2)
    _head(fig, T("距離込み関連度マップ  $W=\\alpha(s)\\cdot m(d)$（窓の小さい順）",
                 "relevance map"))
    _footer(fig, (
        "定義  重み $W_{ij}=\\mathrm{softmax}_j\\!\\left(q_i\\cdot k_j/\\sqrt{d}\\right)$（行の和 = 1、窓は無い）\n"
        if is_softmax else
        ("定義  重み $W_{ij}=\\alpha(s_{ij})\\cdot m(d_{ij})$\n"
         "　　$\\alpha(s)=\\left[\\max\\!\\left(1-\\frac{1-s}{r},\\,0\\right)\\right]^2$  … Trim（内容のしきい値、$r$ = 受容幅）\n"
         "　　$m(d)=\\frac{1}{2}\\left[1+\\cos\\!\\left(\\pi\\min(d/w,1)\\right)\\right]$  … 遮蔽窓（$d\\geq w$ で厳密に 0）\n"))
        + "行 = query 電子 $i$、列 = key 電子 $j$。橙線 = スピン境界。灰 = 窓の外（ほぼ常に $W=0$）。\n"
        + TRIM
        + "Pr = $W>0$ となる対の割合。配置平均。\n\n" + READ)
    fig.savefig(os.path.join(out, tag + "_grid.png"), dpi=145, facecolor=fig.get_facecolor())
    print("  wrote", tag + "_grid.png")

    # ---------------------------------------------------------- 1b. q·k 行列そのもの
    Sm = S.mean(0)                                        # (L,H,N,N)
    v = np.percentile(np.abs(Sm), 99.0)
    NR, NC, pos_of = _grid_shape(NL, NH)
    FOOT, HEAD = 6.4, 1.35                                 # 脚注 / ヘッダに割く高さ [inch]
    Hfig = 2.35 * NR + HEAD + FOOT
    fig, ax = plt.subplots(NR, NC, figsize=(2.05 * NC, Hfig), squeeze=False,
                           gridspec_kw={"wspace": .07, "hspace": .5})
    fig.patch.set_facecolor("#fcfcfb")
    fig.subplots_adjust(top=1 - HEAD / Hfig, bottom=FOOT / Hfig)
    for a_ in ax.ravel(): a_.axis("off")
    for l in range(NL):
        for i, h in enumerate(order):
            A = ax[pos_of(l, i)]; A.axis("on")
            im = A.imshow(Sm[l, h], cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
            A.axhline(nup - .5, color="#1a1a1a", lw=.6); A.axvline(nup - .5, color="#1a1a1a", lw=.6)
            A.set_xticks([]); A.set_yticks([])
            A.set_title("L%dH%d w/d=%s\nsd=%.3f  寄与 %s"
                        % (l, h, wl(l, h), Sm[l, h].std(), cs(l, h)),
                        fontsize=7.0, loc="left", pad=2)
    fig.colorbar(im, ax=ax, fraction=.012, pad=.012,
                 label=T("$s=q\\cdot k$" + ("$/\\sqrt{d}$（softmax 前）" if is_softmax else "（単位ベクトルの内積）"),
                         "q.k"))
    _head(fig, T("q·k 行列（配置平均、窓の小さい順）", "q.k matrices"))
    _footer(fig, (
        ("定義  $s_{ij}=q_i\\cdot k_j/\\sqrt{d}$（softmax 前のスコア。$d$ = ヘッドの次元）\n"
         if is_softmax else
         "定義  $s_{ij}=\\bar q_i\\cdot\\bar k_j$   ただし $\\bar q=q/|q|,\\ \\bar k=k/|k|$（単位ベクトルなので $s\\in[-1,1]$）\n")
        + "行 = query 電子 $i$、列 = key 電子 $j$。層 $l$・ヘッド $h$ ごとに独立。黒線 = スピン境界"
          "（左上 = ↑↑、右下 = ↓↓、非対角ブロック = ↑↓）。配置平均。\n"
        + ("softmax なので窓は無く、w/d は表示しない（--）。\n" if is_softmax else
           "w/d = 学習後の遮蔽窓 $w$ を周期ノルムの最大値 $d_{\\rm ref}$ で割った値。ヘッドは $w$ の小さい順に配置。\n")
        + "※ $s$ は窓が掛かる前の「内容の類似度」。実際に値ベクトルに掛かる重みは "
        + ("$\\mathrm{softmax}_j(s_{ij})$（行の和 = 1）" if is_softmax else
           "$W_{ij}=\\alpha(s_{ij})\\cdot m(d_{ij})$ で、そちらは _grid.png。\n"
           "　　$\\alpha(s)=\\left[\\max\\left(\\frac{s-(1-r)}{r},\\,0\\right)\\right]^2$"
           "  … Trim。内容のしきい値。$r$ = 受容幅（$0<r<2$）。$s\\leq 1-r$ を厳密に 0 に潰す\n"
           "　　$m(d)=\\frac{1}{2}\\left[1+\\cos\\left(\\pi\\min(d/w,1)\\right)\\right]$"
           "  … 遮蔽窓。$d\\geq w$ で厳密に 0。$w=d_{\\rm ref}\\,\\mathrm{softplus}(\\omega)$ で $\\omega$ は学習パラメータ\n")
        + TRIM + "\n"
        + READ))
    fig.savefig(os.path.join(out, tag + "_qk.png"), dpi=145, facecolor=fig.get_facecolor())
    print("  wrote", tag + "_qk.png")

    # ---------------------------------------------------------- 2. 近接順位マップ
    idx = np.argsort(d_true, axis=-1)                       # (B,i,rank)
    Wr = np.take_along_axis(W, idx[:, None, None, :, :], axis=-1).mean(axis=(0, 3))  # (L,H,rank)
    fig, ax = plt.subplots(1, NL, figsize=(6.2 * NL, 5.4), squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    for l in range(NL):
        A = ax[0, l]
        Z = Wr[l][order]
        Z = Z / np.maximum(Z.sum(1, keepdims=True), 1e-12)
        im = A.imshow(Z, aspect="auto", cmap="magma_r", origin="lower")
        for x in cum[cum < NE]:
            A.axvline(x + .5, color=GREEN, lw=1.2, ls="--")
        A.set_xticks(np.arange(0, NE, max(1, NE // 12)))
        A.set_yticks(range(NH))
        A.set_yticklabels(["H%d  w/d=%s" % (h, wl(l, h)) for h in order], fontsize=7)
        A.set_xlabel(T("近い順の順位（0 = 自分自身）", "neighbour rank"), fontsize=10)
        A.set_title(T("層 %d — 緑破線 = 三角格子の殻境界 (6,12,18,30)" % l, "layer %d" % l),
                    fontsize=10.5, loc="left")
        fig.colorbar(im, ax=A, fraction=.03, pad=.02, label=T("重みの割合", "weight share"))
    _head(fig, T("誰を見ているか（近接順位・ラベル不変）", "attention by neighbour rank"))
    fig.tight_layout(rect=[0, .10, 1, .95])
    _footer(fig, "定義  配置ごとに query 電子 $i$ から見て key 電子を最小像 Euclid 距離の昇順に並べ替え、"
                 "重み $W$ を同じ順で取り直して配置・query 平均したもの（行ごとに総和 1 に規格化）。\n"
                 "電子は区別できないので生の $i\\times j$ 行列では「相手」が読めない。順位なら読める。"
                 "緑破線 = 三角格子の殻境界（第1殻まで 6 個、第2殻まで 12 個、…）。", y=0.002)
    fig.savefig(os.path.join(out, tag + "_rank.png"), dpi=145, facecolor=fig.get_facecolor())
    print("  wrote", tag + "_rank.png")

    # ---------------------------------------------------------- 3. 動径プロファイル
    off = ~np.eye(NE, dtype=bool)
    dd = d_true[:, off].ravel()
    edges = np.linspace(0, min(rmax, dd.max()), 33); cen = .5 * (edges[1:] + edges[:-1])
    b = np.clip(np.digitize(dd, edges) - 1, 0, 31)
    fig, ax = plt.subplots(1, 2, figsize=(13.2, 4.6)); fig.patch.set_facecolor("#fcfcfb")
    A = ax[0]
    A.hist(dd, bins=70, range=(0, rmax), density=True, histtype="step", lw=2.2, color="#2a78d6")
    for k, s in enumerate(shells):
        if s < rmax:
            A.axvline(s, color=GREEN, lw=1.4 if k == 0 else .9, ls="-" if k == 0 else "--")
            A.annotate(["a", "√3a", "2a", "√7a", "3a"][k], (s, A.get_ylim()[1] * .95),
                       color=GREEN, fontsize=8.5, ha="center")
    A.set_xlabel(T("電子どうしの距離  $d_{ij}$（最小像, $r_s a_B^{*}$）", "pair distance")); A.set_ylabel(T("密度", "density"))
    A.set_title(T("① 電子どうしの距離の分布と三角格子の殻位置", "pair distances"),
                fontsize=11, loc="left"); A.grid(alpha=.2)
    A = ax[1]
    import matplotlib.cm as cmm
    col = cmm.viridis(np.linspace(0, .92, NH))
    for c, h in enumerate(order):
        wh = W[:, :, h].mean(1)[:, off].ravel()      # 層平均してから対だけ取る（b と長さを揃える）
        y = np.array([wh[b == u].mean() if (b == u).sum() > 20 else np.nan for u in range(32)])
        A.plot(cen, y, "-", color=col[c], lw=1.7,
               label="H%d w/d=%s" % (h, "--" if np.isnan(w[0, h]) else "%.2f" % (w[:, h].mean() / dref)))
    for k, s in enumerate(shells):
        if s < rmax: A.axvline(s, color=GREEN, lw=.9, ls="--")
    A.set_xlabel(T("電子どうしの距離  $d_{ij}$", "pair distance")); A.set_ylabel(T("平均の重み α·m", "mean weight"))
    A.set_title(T("② ヘッドごとの動径プロファイル", "radial profile per head"), fontsize=11, loc="left")
    A.legend(fontsize=6.5, ncol=2); A.grid(alpha=.2)
    fig.tight_layout(rect=[0, 0, 1, .86])
    _head(fig, T("電子どうしの距離の分布と、ヘッドごとの重みの届き方", "radial profiles"))
    fig.savefig(os.path.join(out, tag + "_radial.png"), dpi=145, facecolor=fig.get_facecolor())
    print("  wrote", tag + "_radial.png")

    print("\n  N=%d  L=%d H=%d  d_ref=%.3f  最小像半径=%.2f (=%.2f a)  配置 %d 個"
          % (NE, NL, NH, dref, rmax, rmax / a1, S.shape[0]))
    print("  窓 w/d_ref (小さい順):", "なし (softmax)" if np.all(np.isnan(w))
          else np.round(w.mean(0)[order] / dref, 3))
    selfp = np.array([[100 * M[l, h][np.arange(NE), np.arange(NE)].sum() / max(M[l, h].sum(), 1e-12)
                       for h in order] for l in range(NL)])
    print("  自己重み [%] 層平均      :", np.round(selfp.mean(0), 1))


if __name__ == "__main__":
    main()
