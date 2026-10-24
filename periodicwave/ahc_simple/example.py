"""N=4の複素振幅を比較。例：python -m periodicwave.ahc_simple.example

--plotは任意（Matplotlibがある環境のみ）。VMC学習・energy計算は行わない。
HFを読む場合：--hf-table FILE --lam 既知値 --lattice '[[...],[...]]'
HF表はλやスーパーセルを保存していないので、この2項目を推測しない。
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from periodicwave.ahc_simple.spinor import lower_spinor, pseudospin
from periodicwave.ahc_simple.wavefunctions import (
    load_hf_table, make_bloch_orbitals, slater_wavefunction, wavefunction,
)


def demo_data(n=4, lam=.7, lattice=None):
    """少数平面波のテスト状態。AHC/WCの最適化済み状態ではない。"""
    lattice = np.eye(2)*4 if lattice is None else np.asarray(lattice, dtype=float)
    fractional = np.array([[.08,.12],[.31,.22],[.19,.57],[.64,.41],[.83,.73],[.47,.89]])
    if n not in (4, 6):
        raise ValueError('平面波デモはN=4または6です')
    R = fractional[:n] @ lattice.T
    S = np.array([1,-1]*(n//2))
    integers = np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[1,-1]])
    momenta = integers @ (2*np.pi*np.linalg.inv(lattice).T).T
    orbitals = make_bloch_orbitals(momenta[:n], np.zeros((1,2)), np.ones((n,1)), lam)
    extra = make_bloch_orbitals(momenta[n:n+2], np.zeros((1,2)), np.ones((2,1)), lam)
    return R, S, orbitals, extra, momenta[:n]


def plot_sections(R, S, orbitals, variants, lattice, output):
    """他の電子を固定し、電子0の|ψ|と位相だけを表示する。既存図は上書きしない。"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError('--plotにはMatplotlibが必要です。通常の振幅デモには不要です') from error
    grid = np.linspace(0, 1, 32, endpoint=False)
    fig, axes = plt.subplots(len(variants), 2, figsize=(8, 2.5*len(variants)), squeeze=False)
    for row, (name, options) in enumerate(variants):
        values = np.empty((len(grid), len(grid)), complex)
        for iy, y in enumerate(grid):
            for ix, x in enumerate(grid):
                moved = R.copy(); moved[0] = lattice @ [x,y]
                values[iy,ix] = wavefunction(moved, S, orbitals, lattice=lattice, **options)
        phase = np.ma.masked_where(np.abs(values) == 0, np.angle(values))
        for col, data in enumerate((np.abs(values), phase)):
            image = axes[row,col].imshow(data, origin='lower', extent=(0,1,0,1),
                                         cmap='viridis' if col==0 else 'twilight')
            axes[row,col].set_title(name + (' |psi|' if col==0 else ' phase'))
            axes[row,col].set_xlabel('fractional x');axes[row,col].set_ylabel('fractional y')
            fig.colorbar(image, ax=axes[row,col])
    fig.tight_layout()
    output = Path(output);output.parent.mkdir(parents=True, exist_ok=True)
    target = output;version = 1
    while target.exists():
        target = output.with_name(f'{output.stem}_{version}{output.suffix}');version += 1
    with target.open('xb') as handle:
        fig.savefig(handle, format='png', dpi=140)
    plt.close(fig)
    print('saved:', target)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n',type=int,choices=(4,6),default=4)
    parser.add_argument('--lam',type=float)
    parser.add_argument('--hf-table',type=Path)
    parser.add_argument('--lattice',type=json.loads)
    parser.add_argument('--plot',action='store_true')
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'outputs'/'sections.png')
    args = parser.parse_args(argv)
    if args.hf_table and (args.lam is None or args.lattice is None):
        parser.error('HF表には--lamと元の--latticeを明示してください')
    lam = .7 if args.lam is None else args.lam
    lattice = np.eye(2)*4 if args.lattice is None else np.asarray(args.lattice,dtype=float)
    if lattice.shape != (2,2) or not np.isfinite(lattice).all() or abs(np.linalg.det(lattice)) < 1e-12:
        parser.error('格子は有限な可逆2×2行列（列が格子ベクトル）です')
    if args.hf_table:
        k,G,c = load_hf_table(args.hf_table);n=len(k)
        if n%2:parser.error('このPfaffian比較には偶数電子数のHF表が必要です。軌道は勝手に間引きません')
        R = np.random.default_rng(7).uniform(.1,.9,(n,2)) @ lattice.T
        S = np.array([1,-1]*(n//2))
        orbitals = make_bloch_orbitals(k,G,c,lam)
        extra_k = np.array([[n+1,1],[1,n+1]]) @ (2*np.pi*np.linalg.inv(lattice).T).T
        extra = make_bloch_orbitals(extra_k,np.zeros((1,2)),np.ones((2,1)),lam)
        print('HF input:',args.hf_table,'(energy・AHC/WC・Chern数はここでは未検証)')
    else:
        R,S,orbitals,extra,k = demo_data(args.n,lam,lattice)
        G,c = np.zeros((1,2)),np.ones((len(k),1))
        print('少数平面波のテスト状態：AHC/WCの基底状態ではありません。')
    # 全軌道が同じ境界twistを持つ場合のみ、一つの多体境界位相が定義できる。
    if (not np.allclose(G@lattice/(2*np.pi),np.rint(G@lattice/(2*np.pi)),atol=1e-8,rtol=0)
        or not np.allclose((k-k[0])@lattice/(2*np.pi),np.rint((k-k[0])@lattice/(2*np.pi)),atol=1e-8,rtol=0)):
        parser.error('指定格子に対してGまたは軌道間のk差が整合しません')
    print('R =',R,'\nS =',S)
    print('u_k =',lower_spinor(k[1],lam),'\n<sigma> =',pseudospin(lower_spinor(k[1],lam)))
    J = dict(a=[.08,.1,.12],b=[.7,.8,.9])
    BF = dict(eta=[.06,.03,.05],b=1.2,mode='smooth')
    variants = [
        ('Slater',{}),
        ('Slater + Jastrow',dict(jastrow_parameters=J)),
        ('Slater + Backflow',dict(backflow_parameters=BF)),
        ('Slater + Jastrow + Backflow',dict(jastrow_parameters=J,backflow_parameters=BF)),
        ('Geminal-Pfaffian',dict(use_pfaffian=True,eta=.2,geminal_orbitals=extra)),
        ('Pfaffian + Jastrow + Backflow',dict(use_pfaffian=True,eta=.2,geminal_orbitals=extra,
                                           jastrow_parameters=J,backflow_parameters=BF)),
    ]
    for name,options in variants:
        psi = wavefunction(R,S,orbitals,lattice=lattice,**options)
        phase = np.angle(psi) if psi != 0 else 'undefined (node)'
        print(f'{name}:\n    amplitude = {abs(psi):.12g}\n    phase     = {phase}')
    reference = slater_wavefunction(R,S,orbitals)
    reduced = wavefunction(R,S,orbitals,use_pfaffian=True)
    print('Pfaffian / Slater (eta=0):',reduced/reference if reference != 0 else 'undefined (node)')
    spinors = make_bloch_orbitals(k,G,c,lam,spinor_output=True)
    print('二成分出力と離散ラベルの差:',abs(slater_wavefunction(R,S,spinors,spinor_output=True)-reference))
    print('Slaterと、下側バンド軌道だけのJ/BFなしPfaffianは厳密投影型です。')
    print('一般のJ/BFや任意スピノルの追加Geminalでは厳密投影を保証しません。')
    if args.plot:
        if len(R)>6:parser.error('断面作図は小規模N<=6に限定します')
        plot_sections(R,S,orbitals,variants,lattice,args.output)


if __name__ == '__main__':
    main()
