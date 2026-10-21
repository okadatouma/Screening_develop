# `CustomScreenformer.py` コード解説

`periodicwave/CustomScreenformer.py` を上から順に読み解きます。
教科書『Transformerを理解する』Part 3 の内容が、実際のコードのどこに
対応するかを対応付けながら説明します。

---

## 0. 全体の地図

このファイルには5つの「作る関数」と2つの補助関数があります。
下から上へ呼び出されます。

```
make_fermi_net                （最上位：波動関数まるごと作る）
  └─ make_screenformer_layers （特徴量 → 埋め込み → 注意ブロック）
       └─ make_self_attention_block  （残差接続・LayerNorm・MLPで包む）
            └─ make_screening_attention  ★遮蔽注意の本体★
                 ├─ _unit_length          （単位長正規化）
                 └─ _max_periodic_distance（セル内の最大距離 d_ref）
  └─ make_mlp        （MLP）
  └─ make_layer_norm （Layer Normalization）
```

`CustomPsiformer.py` と違うのは実質 `make_screening_attention` だけです。
残り（MLP・LayerNorm・残差接続・行列式）は完全に同一なので、
**性能差はすべて注意機構の違いに帰着します**。

### init / apply というJAXの流儀

このコードには `class` がほとんど出てきません。代わりに、どの「作る関数」も

```python
def make_なにか(ハイパーパラメータ...):
    def init(key):   # パラメータを作って返す
        ...
    def apply(params, 入力):  # パラメータを受け取って計算する
        ...
    return init, apply
```

という形をしています。これはJAXの関数型スタイルで、

- **パラメータは関数の中に隠れず、外から明示的に渡される**（`params` 引数）
- そのおかげで `jax.grad` で微分でき、`jax.jit` でコンパイルでき、
  `jax.vmap` でバッチ化できる

という利点があります。層数やヘッド数といった**学習しない値**は
`make_...` の引数として渡され、内側の `init`/`apply` から
クロージャとして見えます。学習する値だけが `params` に入ります。

この区別が重要です。例えば遮蔽窓 w は学習するので `params` に入りますが、
その基準長 `d_ref` は学習しないのでクロージャ側です。

---

## 1. 補助関数

### 1.1 `_unit_length` — 単位長正規化

```python
def _unit_length(x, eps=1e-6):
  norm = jnp.sqrt(jnp.sum(x * x, axis=-1, keepdims=True) + eps * eps)
  return x / norm
```

最後の軸（特徴量の軸）についてベクトルを長さ1に揃えます。

`eps` を足しているのは**原点での微分可能性のため**です。素朴に
`x / |x|` と書くと、`|x| = 0` で微分が定義されません。VMCでは
log|ψ| の2階微分（ラプラシアン）まで必要なので、この配慮は必須です。
`sqrt(|x|² + eps²)` にしておけば、`x = 0` でも滑らかです。

`keepdims=True` は、割り算のブロードキャストのために形を
`(..., 1)` に保つためです。

### 1.2 `_max_periodic_distance` — 基準長 d_ref

```python
def _max_periodic_distance(lattice) -> float:
  metric = lat.T @ lat
  ...
  a = 1.0 - np.cos(2.0 * np.pi * grid)
  b = np.sin(2.0 * np.pi * grid)
  ...
  return float((1/(2π)) * np.sqrt(cos_term + sin_term).max())
```

**遮蔽窓 w の単位を決めるための基準長**を計算します。

このコードは周期境界条件を使うので、距離には普通のユークリッド距離ではなく
**周期ノルム**を使います（`networks.py` の `periodic_norm` と同じ式）。
分数座標 s に対して

d(s) = (1/2π) √( aᵀM a + bᵀM b ),  a = 1 − cos(2πs), b = sin(2πs)

これは短距離では実距離に一致し、セル境界付近で滑らかに折り返します。
この関数は、**セル内で実現しうる周期ノルムの最大値**を格子走査で求めます。

なぜ必要かというと、遮蔽窓 w を「セルの何倍か」という無次元量で
扱いたいからです。`w/d_ref < 1` なら「窓がセルより小さい＝実際に
遠方を遮蔽している」、`w/d_ref ≥ 1` なら「窓がセル全域を覆う＝遮蔽なし」
と一目で判断できます。

`numpy` で書かれていて `jax.numpy` ではないのは、**初期化時に一度だけ
計算される定数**だからです。勾配も不要ですし、コンパイル対象でもありません。

---

## 2. `ScreenformerOptions`

```python
@attr.s(auto_attribs=True, kw_only=True)
class ScreenformerOptions(networks.BaseNetworkOptions):
  num_layers: int = 2
  num_heads: int = 4
  ...
  use_softmask: bool = True
  init_trim_r: float = 1.9
  init_softmask_w_factor: float = 0.8
  init_softmask_w_spread: float = 2.5
  use_output_scale: bool = True
  init_output_scale: float = 0.0
```

設定をまとめて持ち運ぶための入れ物です。`BaseNetworkOptions`（`networks.py`）
を継承しているので、次元数・行列式の数・envelope・Jastrow・格子といった
共通設定も一緒に入っています。

`attr.s` は「フィールドを書くだけでコンストラクタを自動生成する」ためのもので、
`kw_only=True` はキーワード引数でしか渡せなくする指定です（引数の順序ミス防止）。

前半（`num_layers` 〜 `mlp_activation_fct`）は `PsiformerOptions` と同一、
後半の `use_softmask` 以降が遮蔽版で追加した部分です。

---

## 3. `make_screening_attention` ★本体★

ここがこのファイルの心臓部です。長いので init と apply に分けます。

### 3.1 学習パラメータの持ち方

先に結論を書くと、学習される量は次の6つです。

| パラメータ | 形 | 意味 |
|---|---|---|
| `q_w`, `k_w` | (embed_dim, num_heads × attn_dim) | クエリ・キーの射影 |
| `v_w` | (embed_dim, num_heads × value_dim) | 値の射影 |
| `g_w` | (embed_dim, num_heads × value_dim) | ゲートの射影 W_G |
| `attn_output` | (num_heads × value_dim, embed_dim) | 出力射影 W_O |
| `trim_rho` | (num_heads,) | 受容幅 r のもと |
| `mask_omega` | (num_heads,) | 遮蔽窓 w のもと |
| `out_scale` | スカラー | 出力倍率 s_O |

後ろ3つが遮蔽版で追加されたものです。

**なぜ r や w を直接持たないのか。** r は 0 < r < 2、w は w > 0 という
制約があります。制約付きの値を勾配法で更新すると、範囲外に飛び出して
壊れます。そこで**制約のない実数を持ち、写像で制約内に押し込みます**。

```python
r = 2.0 * jax.nn.sigmoid(params['trim_rho'])   # (0, 2) に必ず入る
w = d_ref * jax.nn.softplus(params['mask_omega'])  # 必ず正
out = jnp.exp(params['out_scale']) * out           # 必ず正
```

`sigmoid` は実数全体を (0,1) へ、`softplus(x) = log(1+eˣ)` は実数全体を
(0,∞) へ滑らかに写します。この「制約を写像で表現する」やり方は
最適化では定石です。逆に初期値を指定するときは逆写像が必要で、
それが `rho_init = log(r/(2−r))`（ロジット）や
`omega_init = log(expm1(w))`（softplusの逆）です。

### 3.2 多重遮蔽（multiscreen）の初期化

```python
centre = max(float(init_softmask_w_factor), 1e-3)
spread = max(float(init_softmask_w_spread), 1.0)
if num_heads == 1 or spread == 1.0:
    w_factors = np.full((num_heads,), centre)
else:
    w_factors = np.geomspace(centre / spread, centre * spread, num_heads)
omega_init = np.log(np.expm1(np.clip(w_factors, 1e-3, None)))
```

ヘッドごとの初期遮蔽長を**幾何級数的にばらけさせます**。
既定値 centre = 0.8, spread = 2.5, ヘッド4個なら

w/d_ref = 0.32, 0.56, 1.14, 2.00

となり、最初から「近傍しか見ないヘッド」と「セル全域を見るヘッド」が
共存します。これが multiscreen の実体です。

**なぜばらけさせるのか。** 全ヘッドを同じ w で始めると、ヘッド同士が
完全に対称になります。対称性を破るのは重み行列のランダム初期化だけで、
遮蔽長そのものは似た値に留まりがちです。実際、当初の実装（全ヘッド同一・
w = 3·d_ref）では600ステップ後も w がほぼ動きませんでした。

もう一つの理由は**勾配の大きさ**です。raised cosine は d ≪ w のとき
m ≈ 1 − (π²/4)(d/w)² とほぼ平坦で、w に対する勾配がほぼゼロです。
窓が広すぎると「動く理由が生まれない」ので、centre は
セル内距離と同程度（0.8）に置いてあります。

### 3.3 出力倍率の自動較正

```python
_sig = 1.0 / np.sqrt(float(attn_dim))
_s_grid = np.linspace(-5.0 * _sig, 5.0 * _sig, 4001)
_pdf = np.exp(-0.5 * (_s_grid / _sig) ** 2); _pdf /= _pdf.sum()
_alpha_mean = float((np.square(np.maximum(_s_grid - (1.0 - init_trim_r), 0.0)
                               / init_trim_r) * _pdf).sum())
_branch_gain = 0.85 if use_gate else 2.4
_branch_target_fraction = 0.3
_scale0 = _branch_target_fraction * np.sqrt(float(embed_dim)) / (
    _alpha_mean * np.sqrt(float(num_electrons)) * _branch_gain)
out_scale_init = float(np.log(_scale0))
```

ここは実装上いちばん重要な部分なので丁寧に説明します。

**問題。** softmaxは重みを分母で割って和を1に正規化します。遮蔽注意は
その分母を捨てるので、**重みの絶対的な大きさが誰にも保証されません**。

初期状態では q と k はランダムな方向なので、その cos類似度
s = q̄·k̄ は 0 のまわりに幅 1/√(attn_dim) で散らばります
（attn_dim = 16 なら幅 0.25）。Trim はこれを二乗するので、
重み α の平均は極めて小さくなります。実測すると
**遮蔽注意の出力はPsiformerの1/150**、残差ストリームの1/170でした。
これでは注意ブランチが存在しないのと同じで、実質「MLPだけ」の
ネットワークとして学習してしまいます。

**解決。** 教科書のマップにある学習可能スカラー e^{s_O} の初期値を
解析的に決めます。上のコードは

1. s がガウス分布 N(0, 1/attn_dim) に従うと仮定し、
   数値積分（`_pdf` による重み付き和）で **Trim重みの期待値 α_mean** を計算
2. 残る段（TanhNorm・ゲート・W_O）のゲイン `_branch_gain` を実測値で補正
3. 注意ブランチが残差ストリームの約30%の大きさから始まるように倍率を決定

としています。`_branch_target_fraction = 0.3` は実測で選んだ値です。
1.0（残差と同じ大きさ）にすると、Trim の折れ点や窓の縁といった
**鋭い構造が増幅されて log|ψ| の曲率が上がり**、局所エネルギーの
ばらつき（locstd）が悪化しました。逆に小さすぎると注意が効きません。

この段落が「なぜ改良で性能が逆転したか」の答えです。

### 3.4 `init` — パラメータの生成

```python
def init(key):
    key, *subkeys = jax.random.split(key, num=6)
    params = {}
    params['q_w'] = network_blocks.init_linear_layer(
        subkeys[0], in_dim=embed_dim, out_dim=num_heads * attn_dim,
        include_bias=False)['w']
    ...
```

`jax.random.split` は乱数キーを分割します。JAXの乱数は**明示的に
キーを渡す**方式で、同じキーからは必ず同じ値が出ます。射影ごとに
別々のサブキーを使うのは、異なる重み行列に同じ値が入らないためです。

`include_bias=False` はバイアス項なしを意味します。q, k は正規化されるので
バイアスを足しても方向はほぼ変わらず、意味が薄いためです。

`num_heads * attn_dim` を一度に出しておいて、後で `reshape` で
ヘッドに分ける、という書き方は Transformer実装の定番です。
ヘッドごとに別の行列を持つのと数学的に等価ですが、行列積が1回で済みます。

### 3.5 `apply` — 遮蔽注意の8段階

入力は `single_electron_stream`（形 `[電子数, embed_dim]`）と
`d_ee`（形 `[電子数, 電子数]`、電子間の周期距離）です。

#### 段階0：射影

```python
q = jnp.dot(stream, params['q_w']).reshape(num_electrons, num_heads, attn_dim)
k = jnp.dot(stream, params['k_w']).reshape(num_electrons, num_heads, attn_dim)
v = jnp.dot(stream, params['v_w']).reshape(num_electrons, num_heads, value_dim)
```

各電子の特徴ベクトルから q, k, v を作り、ヘッドごとに切り分けます。
形は `[i, h, d]`（電子・ヘッド・特徴）。

#### 段階1：単位長正規化 → cos類似度

```python
qbar = _unit_length(q); kbar = _unit_length(k); vbar = _unit_length(v)
s = jnp.einsum('ihd,jhd->hij', qbar, kbar)
```

`einsum` は添字記法そのままの計算です。`'ihd,jhd->hij'` は

s[h,i,j] = Σ_d qbar[i,h,d] · kbar[j,h,d]

を意味します。「入力2つの添字を書き、`->` の後に出力の添字を書く。
出力に現れない添字（ここでは d）は和を取る」というルールです。

qbar, kbar は長さ1なので、この内積は**そのまま cos θ** で、
必ず −1 ≤ s ≤ 1 に収まります。softmaxの `1/√d_k` スケーリングが
不要になるのはこのためです。値が有界なので指数関数の暴走も起きません。

#### 段階2：Trim

```python
r = 2.0 * jax.nn.sigmoid(params['trim_rho'])   # [num_heads]
r = r[:, None, None]                            # [num_heads,1,1] にして放送
alpha = jnp.square(jax.nn.relu(s - (1.0 - r)) / r)
```

α_ij = ( ReLU(s_ij − (1 − r)) / r )²

- s ≤ 1 − r の対は**厳密に 0**（softmaxのように「小さいが非ゼロ」ではない）
- s = 1（完全一致）で α = 1
- 二乗しているので**境界で滑らか**（C¹級、2階微分も有界）

`r[:, None, None]` はヘッド軸だけを持つ配列を `[h,1,1]` に整形し、
`s` の `[h,i,j]` に対してブロードキャストさせるためです。

**ReLUの二乗である理由**：素の ReLU だと境界 s = 1−r で1階微分が
不連続になり、ラプラシアンにデルタ関数的な寄与が出てしまいます。
二乗すれば境界で値も1階微分も0になり、局所エネルギーが壊れません。

#### 段階3：Softmask（遮蔽窓）

```python
if use_softmask:
    w = d_ref * jax.nn.softplus(params['mask_omega'])   # [num_heads]
    t = jnp.clip(d_ee[None, :, :] / w[:, None, None], 0.0, 1.0)
    m = 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    weights = alpha * m
```

m_ij = ( 1 + cos(π · min(d_ij/w, 1)) ) / 2

raised cosine（レイズドコサイン窓）です。

- d = 0 で m = 1、d ≥ w で m = 0（`clip` で t を1に止めているため）
- 境界 d = w で値も1階微分も0 → **C¹級で滑らか**
- d → 0 で dm/dd → 0 → **電子が接近しても新たなカスプを作らない**
  （カスプ条件は従来どおり特徴量とJastrowの担当）

**ここが言語モデルとの最大の相違点です。** 言語では距離はトークン位置の差
|i − j| でした。しかしこのコードでは**トークンは電子そのもの**で、
電子に並び順はありません（波動関数は置換について反対称）。
そこで距離には**物理的な電子間距離 d_ee（周期ノルム）** を使います。

結果として遮蔽窓 w_h は「ヘッド h が何ボーア半径先まで見るか」という
**物理的な遮蔽長**として読めます。これは言語モデルの softmask には無い、
この問題設定特有の意味づけです。

`d_ee[None, :, :]` はヘッド軸を先頭に足して `[1,i,j]` にし、
`w[:, None, None]`（`[h,1,1]`）と割り算できるようにしています。

#### 段階4：分母なし集約

```python
attn = jnp.einsum('hij,jhd->ihd', weights, vbar)
```

h_i = Σ_j α_ij m_ij v̄_j

softmaxの分母 Σ_j exp(...) がありません。これが「Screening Is Enough」の
中心的な主張で、正規化を捨てても Trim と Softmask が重みを抑えるので
発散しない、という設計です。

ただし前述の通り、分母を捨てた代償として**絶対的な大きさの保証が失われる**
ので、段階7の e^{s_O} が必要になります。

#### 段階5：TanhNorm

```python
if use_tanh_norm:
    n = jnp.sqrt(jnp.sum(attn * attn, axis=-1, keepdims=True) + eps * eps)
    attn = attn * jnp.tanh(n) / n
```

o = tanh(|h|) · h/|h|

方向 h/|h| はそのままに、長さだけを tanh で潰します。
|h| が小さいときは tanh(x) ≈ x なのでほぼ恒等写像、
大きいときは長さが1に飽和します。分母を持たない代わりに、
**出力が発散しないことをここで保証**しています。

`eps` は `_unit_length` と同じ理由（原点での微分可能性）です。

#### 段階6：ゲート

```python
attn = attn.reshape(*stream.shape[:-1], -1)     # ヘッドを連結
if use_gate:
    g = jnp.dot(single_electron_stream, params['g_w'])
    attn = attn * jnp.tanh(jax.nn.silu(g))
```

ĝ_i = tanh(SiLU(W_G x_i)) を要素ごとに掛けます。
これは重い FFN の代わりに置かれた軽い非線形処理です。

重要なのは、この演算が**各電子ごとに独立**（要素積）である点です。
電子をまたぐ混合をしないので、**置換同変性が保たれます**。
波動関数の反対称性はこの同変性の上に成り立っているので、
ここを崩す演算を入れてはいけません。

#### 段階7：出力射影と倍率

```python
out = network_blocks.linear_layer(attn, params['attn_output'])
if use_output_scale:
    out = jnp.exp(params['out_scale']) * out
return out
```

Δx_i = e^{s_O} W_O o_i

この Δx が残差ストリームに足されます。3.3節で説明した通り、
`e^{s_O}` は初期値が自動較正され、その後は学習で調整されます。

---

## 4. `make_self_attention_block`

```python
def apply(params, single_electron_stream, d_ee):
    x = single_electron_stream
    for layer in range(num_layers):
        attn_output = attention_apply(params['attention'][layer], x, d_ee)
        x = x + attn_output                       # 残差接続
        if use_layer_norm:
            x = layer_norm_apply(params['ln'][layer][0], x)
        mlp_output = mlp_apply(params['mlp'][layer], x)
        x = x + mlp_output                        # 残差接続
        if use_layer_norm:
            x = layer_norm_apply(params['ln'][layer][1], x)
    return x
```

`CustomPsiformer.py` と**同一**です。違いは `d_ee` を各層の注意に
渡している1点だけ。距離は層によらず同じなので、そのまま使い回します。

Part 2で勉強した「注意 → 残差 → 正規化 → MLP → 残差 → 正規化」を
`num_layers` 回繰り返す構造そのものです。

---

## 5. `make_screenformer_layers`

### 5.1 `d_ref` の確定

```python
d_ref = _max_periodic_distance(options.pbc_lattice)
```

格子はここで初めて分かるので、基準長の計算はこの階層で行い、
下の `make_self_attention_block` → `make_screening_attention` へ
引数として流します。

### 5.2 `apply` — 特徴量から注意へ

```python
ae_features, _ = options.feature_layer.apply(ae=ae, r_ae=r_ae, ee=ee, r_ee=r_ee,
                                             **params['input'])
ae_features = jnp.concatenate((ae_features, spins[..., None]), axis=-1)
x = jnp.dot(ae_features, params['embed'])
d_ee = r_ee[..., 0]
return self_attn_apply(params, x, d_ee)
```

1. **特徴量を作る** — 周期特徴量（分数座標の sin/cos）。位置情報はここから入ります
2. **スピンを連結** — 置換同変性を正しく保つために必要
3. **埋め込み** — 特徴量を `mlp_dim` 次元の残差ストリームへ線形写像
4. **距離を取り出す** — `r_ee` は形 `[i,j,1]` なので、最後の軸を落として
   `[i,j]` にし、Softmask に渡す

**MiPE（位置符号化の回転）を実装していないのはここに理由があります。**
電子には並び順が無いので、順序に基づく位置符号化は置換同変性を壊します。
位置の情報は上の1.で入る周期特徴量が担当します。

---

## 6. `make_fermi_net`

最上位。やっていることは組み立てだけです。

```python
options = ScreenformerOptions(...)                  # 設定を1つにまとめる
screenformer_layers = make_screenformer_layers(...)  # 等変層を作る
orbitals_init, orbitals_apply = networks.make_orbitals(
    nspins=nspins, charges=charges, options=options,
    equivariant_layers=screenformer_layers)          # 軌道まで作る（共通コード）

def network_apply(params, pos, spins, atoms, charges):
    orbitals = orbitals_apply(params, pos, spins, atoms, charges)
    return network_blocks.logdet_matmul(orbitals)    # 行列式で反対称化
```

注目すべきは、**反対称性はここで初めて登場する**ことです。
注意機構は「置換同変」（電子を入れ替えると出力も同じように入れ替わる）
なだけで、反対称ではありません。最後に `logdet_matmul` が
行列式を取ることで、初めて反対称な波動関数になります。

```
置換同変な特徴量  →  軌道行列  →  行列式  →  反対称な ψ
```

この構造は FermiNet 以来の標準的な設計で、遮蔽版でも一切変えていません。
だからこそ `test_screenformer.py` の反対称性テストが通ります。

戻り値は `(位相, log|ψ|)` の組です。複素波動関数なので第1要素は
位相角 θ（`logdet_matmul` の実装を参照）であり、符号 ±1 ではありません。

---

## 7. 一周の流れ

電子座標が与えられてからエネルギーが出るまで：

```
電子座標 pos
  ↓ construct_input_features（周期ノルムで ae, ee, r_ae, r_ee）
  ↓ feature_layer（sin/cos の周期特徴量）
  ↓ + スピン
  ↓ embed（線形）           x⁽⁰⁾  [電子数, 64]
  ↓ ┌─ 遮蔽注意 → 残差 → LayerNorm
  ↓ └─ MLP     → 残差 → LayerNorm      × 4層
  ↓ 軌道への線形射影 → 複素数に詰め直す
  ↓ 行列式（logdet_matmul）
  ψ の位相と log|ψ|
  ↓ folx で座標について2階微分（ラプラシアン）
  局所エネルギー E_L → 平均が変分エネルギー
```

---

## 8. 教科書との対応表

| 教科書・スライドの概念 | コード上の場所 |
|---|---|
| 単位長正規化 | `apply` 段階1 `_unit_length` |
| cos類似度 s_ij | `jnp.einsum('ihd,jhd->hij', qbar, kbar)` |
| Trim（受容幅 r） | `apply` 段階2 `alpha = square(relu(s-(1-r))/r)` |
| Softmask（遮蔽窓 w） | `apply` 段階3 `m = 0.5*(1+cos(pi*t))` |
| 分母なし集約 | `apply` 段階4 `einsum('hij,jhd->ihd', weights, vbar)` |
| TanhNorm | `apply` 段階5 |
| ゲート ĝ = tanh(SiLU(g)) | `apply` 段階6 |
| 出力倍率 e^{s_O} W_O | `apply` 段階7 |
| Multiscreen（複数の遮蔽スケール） | `w_factors = np.geomspace(...)` と学習される `mask_omega` |
| MiPE | **未実装**（置換同変性のため。5節参照） |
| 残差ストリーム | `make_self_attention_block` の `x = x + ...` |

---

## 9. VMC特有の注意点（なぜ滑らかさに拘るか）

局所エネルギー E_L(R) = Hψ(R)/ψ(R) の計算には log|ψ| の
**2階微分**（ラプラシアン）が必要です。したがってネットワークは
電子座標について2回微分可能でなければなりません。

このファイルで滑らかさに配慮した箇所：

| 箇所 | 配慮 |
|---|---|
| `_unit_length` の `eps` | 原点で `x/|x|` が微分不能になるのを回避 |
| Trim の**二乗** ReLU | 境界で1階微分が連続、2階微分が有界 |
| Softmask の raised cosine | 境界 d = w で C¹級。d→0 で dm/dd→0 なので接近時に新たなカスプを作らない |
| TanhNorm の `eps` | 同上 |
| 出力倍率を大きくしすぎない | 鋭い構造を増幅すると曲率が上がり locstd が悪化する（実測） |

最後の行は理論ではなく実測から得た知見です。倍率を残差と同程度（1.0倍）に
すると locstd が悪化し、0.3倍程度が最良でした。
**「滑らかさ」は数式上の性質であると同時に、実際に計算される量の質を
左右する実用的な制約でもある**、という例になっています。
