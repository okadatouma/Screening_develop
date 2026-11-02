# -*- coding: utf-8 -*-
"""λ-jellium（lj3 の設定）の計算を、処理の段ごとに 1 ファイルずつ短く書き直した参照用コード。

本体（periodicwave/）は変えていない。ここは読むためのコードで、学習には使わない。
同じパラメータとウォーカーから本体と同じ値が出ることを check.py で確かめてある（README.md）。

  model.py        ③    run のフォルダ（config.json、チェックポイント）と ① の HF の表を読む
  inputs.py       ⑤-1  1 電子の入力 12 成分と、対の距離 d_ij
  attention.py    ⑤-2  埋め込み → 遮蔽アテンション → MLP（各電子の特徴 h_i）
  orbitals.py     ⑤-3  軌道（HF ＋ ネットワークの補正）と Jastrow
  wavefunction.py ⑤-4  行列式 4 枚の和 → (位相, ln|ψ|)
  sampling.py     ④-1  |ψ|² に従うサンプル（Metropolis、位置と擬スピン）
  hamiltonian.py  ④-2  局所エネルギー E_L = Hψ/ψ（運動、λ の項、擬スピン反転、Δ n_-、Ewald のクーロン）
  gradient.py     ④-3  エネルギーの勾配（E_L の切り詰めを含む）と ④-4 の更新の仕組み
  check.py             本体との突き合わせ

番号は doc/code_map/code_map.pdf の段（③ 設定、④ 学習のループ、⑤ 1 回の ψ の計算）。
"""
