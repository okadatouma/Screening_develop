"""Build a Japanese submission PDF and auditable data bundle from saved results.

Reporting-only environment: reportlab, fonttools, matplotlib, numpy, pypdf.
Never updates the experiment's automatic report or scientific artifacts.
"""
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import sys
from xml.sax.saxutils import escape
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'research_framework/results/2026-11-10_diagonal48'
GPU=ROOT/'results/continuous-convergence-relaxed5e-4-20261107'
OUT=RUN/'submission_report_20261112'
ASSETS=Path('/tmp/n36-report-assets')
IDS={'baseline':'A','baseline_plus_diagonal':'B','fourier_l1':'C','fourier_l1_without_diagonal':'D'}
NAMES={'A':'baseline','B':'baseline＋対角成分','C':'learned Fourier＋L1','D':'Cの対角成分を無効化'}
LABELS=list(IDS)


def main():
    OUT.mkdir(exist_ok=False)
    for name in ['figures','data','assets']:(OUT/name).mkdir()
    manifest={}
    def read(p):
        b=p.read_bytes();manifest[str(p)]=dict(sha256=hashlib.sha256(b).hexdigest(),bytes=len(b));return b
    def js(p):return json.loads(read(p))
    def csvrows(p):
        read(p)
        return list(csv.DictReader(p.open()))
    def savejson(name,x): (OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
    def csvsave(name,rows):
        with (OUT/'data'/name).open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    now=datetime.now(timezone(timedelta(hours=9)))
    results=js(RUN/'results.json');protocol=js(RUN/'protocol.json');plan=js(RUN/'plan.json')
    launch=js(RUN/'launch_check.json');clock=js(RUN/'budget_clock.json')
    models=results['models'];contrasts=results['paired_contrasts'];ph=csvrows(RUN/'ph_trajectory.csv')
    assert len(models)==12 and all(v['step']==10000 for v in models)
    assert all(v['status']=='completed' for v in results['workers'])
    assert not any(v['sampling_pass'] for v in models)
    measurements={v['name']:js(RUN/'measurement'/v['name']/'summary.json') for v in models}
    stats={}
    for label in LABELS:
        xs=[r['train_last1000'] for r in models if r['label']==label]
        stats[label]=dict(mean=float(np.mean(xs)),sd=float(np.std(xs,ddof=1)))
    cstats={}
    for label in ['baseline_plus_diagonal','fourier_l1_without_diagonal']:
        xs=[r['training_delta'] for r in contrasts if r['variant']==label]
        obs=abs(np.mean(xs));null=[abs(np.mean(np.array(xs)*s)) for s in itertools.product([-1,1],repeat=3)]
        cstats[label]=dict(mean=float(np.mean(xs)),sd=float(np.std(xs,ddof=1)),
                           p_two_sided_signflip=float(np.mean(np.array(null)>=obs-1e-12)))
    assert np.isclose(cstats['fourier_l1_without_diagonal']['mean'],8.051522841579223)
    assert np.isclose(cstats['baseline_plus_diagonal']['mean'],1.825583713043443)
    temporal=[]
    for lab in LABELS:
        for graph in ['QK','Screening','FeatureVR','Window']:
            for step in protocol['fixed_probe_steps']:
                xs=[float(r['H1_area_0_1']) for r in ph if r['label']==lab and r['graph']==graph and r['variant']=='rank' and int(r['step'])==step]
                assert len(xs)==3
                temporal.append(dict(group=IDS[lab],label=lab,graph=graph,step=step,mean=float(np.mean(xs)),seed_sd=float(np.std(xs,ddof=1))))
    # Validate the static mask throughout training, not just its final value.
    mask_checks=[];all_histories={}
    for m in models:
        folder=next((RUN/'runs'/c['name'] for c in plan if c['name']==m['name']),None)
        if folder is None:folder=Path(next(c['reference']['folder'] for c in plan if c['reference']['name']==m['name']))
        history=[json.loads(x) for x in read(folder/'history.jsonl').splitlines() if x]
        assert len(history)==10000
        all_histories[m['name']]=history
        assert np.isclose(np.mean([r['energy'] for r in history[-1000:]]),m['train_last1000'])
        if m['label']=='fourier_l1_without_diagonal':
            maximum=max(abs(r['fourier_amplitudes'][2]) for r in history)
            assert maximum==0
            mask_checks.append(dict(name=m['name'],steps=10000,maximum_disabled_amplitude=maximum))
    # Freeze the GPU status as an appendix, independently of the CPU conclusions.
    queue=subprocess.run(['squeue','-u','tomax','-o','%.22i %.28j %.10T %.12M %R'],capture_output=True,text=True)
    (OUT/'data/squeue.txt').write_text(queue.stdout+queue.stderr)
    gpu_plan=js(GPU/'plan.json');gpu_count={}
    for c in gpu_plan['cases']:
        d=GPU/'runs'/c['id']
        if (d/'finished.json').exists():status=js(d/'finished.json')['status']
        elif (d/'failure.txt').exists():status='failed'
        elif (d/'case.json').exists():status='in_progress'
        else:status='not_started'
        gpu_count[status]=gpu_count.get(status,0)+1
    savejson('data/gpu_status.json',dict(timestamp=now.isoformat(),counts=gpu_count,
             tolerance_Ha_per_electron=gpu_plan['energy_tolerance_Ha_per_electron']))
    # Japanese fonts are embedded in the PDF; preserve the OFL license.
    for weight in [400,700]:
        shutil.copy2(ASSETS/f'NotoSansJP-{weight}.ttf',OUT/'assets')
    shutil.copy2(ASSETS/'OFL.txt',OUT/'assets/OFL.txt')
    for name,weight in [('JP',400),('JP-Bold',700)]:pdfmetrics.registerFont(TTFont(name,str(OUT/'assets'/f'NotoSansJP-{weight}.ttf')))
    pdfmetrics.registerFontFamily('JP',normal='JP',bold='JP-Bold',italic='JP',boldItalic='JP-Bold')
    font_manager.fontManager.addfont(str(OUT/'assets/NotoSansJP-400.ttf'))
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(OUT/'assets/NotoSansJP-400.ttf')).get_name(),
                         'font.size':9,'axes.grid':True,'grid.alpha':.2,'axes.unicode_minus':False})
    palette=['#254f77','#278979','#b27619','#ae4c53'];seedcolors=['#2f6c9c','#de9344','#4c9270']
    def figure(fig,name):
        fig.tight_layout();fig.savefig(OUT/'figures'/f'{name}.png',dpi=180,bbox_inches='tight')
        fig.savefig(OUT/'figures'/f'{name}.pdf',bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.45))
    for j,lab in enumerate(LABELS):
        g=stats[lab];axes[0].errorbar(j,g['mean'],yerr=g['sd'],fmt='s',color=palette[j],capsize=4)
        for si,seed in enumerate([101,202,303]):
            y=next(m['train_last1000'] for m in models if m['label']==lab and m['seed']==seed)
            axes[0].scatter(j+(si-1)*.09,y,s=17,color=seedcolors[si])
    axes[0].set(xticks=range(4),xticklabels=list(NAMES),ylabel='学習末尾1000更新のE（内部単位）',title='条件別平均とseed標準偏差')
    for j,var in enumerate(['baseline_plus_diagonal','fourier_l1_without_diagonal']):
        for si,seed in enumerate([101,202,303]):
            y=next(v['training_delta'] for v in contrasts if v['variant']==var and v['seed']==seed)
            axes[1].scatter(j+(si-1)*.08,y,color=seedcolors[si],label=str(seed) if j==0 else None)
    axes[1].axhline(0,color='black',lw=.7);axes[1].set(xticks=[0,1],xticklabels=['B−A：単純追加','D−C：無効化'],ylabel='同一seedの学習E差（新−旧）',title='対角成分の無効化は3 seedとも悪化')
    axes[1].legend(title='seed',fontsize=8);figure(fig,'energy_comparison')
    fig,axes=plt.subplots(2,2,figsize=(10.5,5.1),sharex=True,sharey=True)
    for ax,lab in zip(axes.flat,LABELS):
        for si,seed in enumerate([101,202,303]):
            h=all_histories[next(v['name'] for v in models if v['label']==lab and v['seed']==seed)]
            ax.plot(np.arange(100,10001,100),np.array([v['energy'] for v in h]).reshape(100,100).mean(1),color=seedcolors[si],label=str(seed),lw=.9)
        ax.set(title=IDS[lab]+'：'+NAMES[IDS[lab]],xlabel='更新数',ylabel='学習E（内部単位）')
    axes[0,0].legend(fontsize=8);figure(fig,'training_curves')
    fig,axes=plt.subplots(1,3,figsize=(10.5,3.2))
    for ax,graph in zip(axes,['QK','Screening','FeatureVR']):
        for j,lab in enumerate(LABELS):
            vv=[r for r in temporal if r['label']==lab and r['graph']==graph]
            x=np.array([v['step'] for v in vv]);y=np.array([v['mean'] for v in vv]);sd=np.array([v['seed_sd'] for v in vv])
            ax.plot(x,y,'.-',color=palette[j],label=IDS[lab]);ax.fill_between(x,np.maximum(0,y-sd),y+sd,color=palette[j],alpha=.1)
        ax.set(title=graph,xlabel='更新数',ylabel='順位正規化後のH1面積');ax.legend(ncol=2,fontsize=7)
    figure(fig,'ph_trajectories')
    fun={v['name']:{x['intervention']:x for x in v['responses']} for v in results['functional_response']}
    function_table=[]
    for m in models:
        target='remove_diagonal' if m['label']=='baseline_plus_diagonal' else 'remove_fourier_2'
        head=fun[m['name']]['remove_all_heads'];rem=fun[m['name']].get(target)
        function_table.append(dict(group=IDS[m['label']],seed=m['seed'],head_logabs_rms=head['logabs_centered_rms'],
            head_phase_rms=head['phase_centered_rms'],diagonal_logabs_rms=rem['logabs_centered_rms'] if rem else None,
            diagonal_phase_rms=rem['phase_centered_rms'] if rem else None))
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.1))
    for ax,key,title in zip(axes,['diagonal_logabs_rms','head_logabs_rms'],['対角成分を消した場合','全Attention headを消した場合']):
        for j,g in enumerate(NAMES):
            vals=[v for v in function_table if v['group']==g]
            for si,v in enumerate(vals):
                if v[key] is not None:ax.scatter(j+(si-1)*.08,v[key],color=seedcolors[si],label=str(v['seed']) if j==1 else None)
        ax.set(xticks=range(4),xticklabels=list(NAMES),ylabel='波動関数のlog振幅形状変化RMS',title=title)
    axes[0].legend(fontsize=8);figure(fig,'functional_response')
    ordered=sorted(models,key=lambda r:(list(IDS).index(r['label']),r['seed']))
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.6))
    for i,m in enumerate(ordered):
        s=measurements[m['name']]['diagnostics'];means=[v['mean'] for v in s['starts']]
        axes[0].plot([i-.12,i+.12],means,'-',color='gray',lw=.8)
        axes[0].scatter(i-.12,means[0],color='#c17b37',s=18,label='独立初期配置' if i==0 else None)
        axes[0].scatter(i+.12,means[1],color='#326b98',s=18,label='学習末尾配置' if i==0 else None)
    xx=[IDS[m['label']]+str(m['seed']) for m in ordered]
    for ax in axes:ax.set_xticks(range(12),xx,rotation=60,ha='right',fontsize=7)
    axes[0].set(ylabel='開始配置別の評価E（Ha/電子）',title='開始配置によるずれが全条件で残存');axes[0].legend(fontsize=7)
    axes[1].bar(range(12),[m['rhat'] for m in ordered],color='#aa5755');axes[1].axhline(1.05,color='black',ls='--',lw=.8)
    axes[1].set(ylabel='エネルギーRhat',title='全12条件で基準1.05を超過');figure(fig,'measurement_mixing')
    # A structured PDF and Markdown share the exact narrative/tables below.
    styles={
        'body':ParagraphStyle('body',fontName='JP',fontSize=9.1,leading=15,wordWrap='CJK',spaceAfter=6),
        'small':ParagraphStyle('small',fontName='JP',fontSize=7.6,leading=12,wordWrap='CJK',spaceAfter=4),
        'title':ParagraphStyle('title',fontName='JP-Bold',fontSize=20,leading=29,wordWrap='CJK',spaceAfter=13,textColor=colors.HexColor('#204662')),
        'h1':ParagraphStyle('h1',fontName='JP-Bold',fontSize=15,leading=22,wordWrap='CJK',spaceAfter=10,textColor=colors.HexColor('#204662')),
        'h2':ParagraphStyle('h2',fontName='JP-Bold',fontSize=11,leading=17,wordWrap='CJK',spaceBefore=6,spaceAfter=5),
        'cell':ParagraphStyle('cell',fontName='JP',fontSize=8,leading=12,wordWrap='CJK'),
        'head':ParagraphStyle('head',fontName='JP-Bold',fontSize=8,leading=12,wordWrap='CJK',textColor=colors.white)}
    story=[];md=[]
    def para(text,style='body'):
        story.append(Paragraph(escape(text),styles[style]));md.extend([text,''])
    def heading(text,first=False):
        if not first:story.append(PageBreak())
        story.append(Paragraph(escape(text),styles['title' if first else 'h1']));md.extend([('# ' if first else '## ')+text,''])
    def sub(text):story.append(Paragraph(escape(text),styles['h2']));md.extend(['### '+text,''])
    def table(headers,body,widths=None):
        data=[[Paragraph(escape(str(v)),styles['head']) for v in headers]]
        data += [[Paragraph(escape(str(v)),styles['cell']) for v in row] for row in body]
        t=Table(data,colWidths=[x*mm for x in widths] if widths else None,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#254c68')),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
            ('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),
            ('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#254c68')),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f0f4f7'),colors.white]),
            ('LINEBELOW',(0,1),(-1,-1),.2,colors.HexColor('#d5dee5'))]))
        story.extend([t,Spacer(1,6)])
        md.extend(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|'])
        md.extend(['|'+'|'.join(map(str,row))+'|' for row in body])
        md.append('')
    def image(name,caption,width=173):
        from PIL import Image as PILImage
        p=OUT/'figures'/f'{name}.png';w,h=PILImage.open(p).size
        story.append(Image(str(p),width=width*mm,height=width*mm*h/w));para(caption,'small')
        md.extend([f'![{caption}](figures/{name}.png)',''])
    group_body=[[IDS[lab],NAMES[IDS[lab]],f"{stats[lab]['mean']:.2f} ± {stats[lab]['sd']:.2f}"] for lab in LABELS]
    heading('N=36電子系の入力表現と対角低波数の追加対照実験',True)
    para('研究進捗・結果報告書｜2026年11月12日','h2')
    para('対象：新規6学習条件と既存6条件の再評価。主実験の完了日時は2026年11月10日22:46（JST）、本書のGPU付録の確認時刻は'+now.strftime('%Y年%m月%d日 %H:%M')+'（JST）。')
    sub('要旨')
    para('learned Fourier＋L1から候補波数(1,−1)を最初から無効化すると、同じseedの対照に対し学習末尾のエネルギーが3 seedすべてで悪化した。差の平均は＋8.05 ± 2.10（内部単位、±はseed標準偏差）である。一方、baselineへの同成分の単純追加は1 seedで改善、2 seedで悪化し、一貫した改善を示さなかった。')
    para('この結果は、現在の候補集合・モデル・学習予算における対角成分の利用と、学習のしやすさへの関与を支持する。ただし、長い独立評価も全12条件で未収束であり、真の変分エネルギー・基底状態・物理相の優劣は認定しない。')
    table(['達成項目','結果'],[['新規学習','6/6条件で10000更新を完了'],['独立評価','12/12条件を実施。各モデル4時間で未収束終了'],['PH・出力介入','12条件、5 checkpointでPHを解析。最終状態で出力介入'],['所要時間','約21時間21分。計算エラー・全体期限による強制停止なし']],[32,141])
    image('energy_comparison','図1：学習末尾の記述統計と同一seedの対応差。誤差棒は3 seedの標準偏差であり、物理的な信頼区間ではない。')

    heading('1．目的・実験設計')
    para('前回の18条件比較では、learned Fourierの(1,−1)候補が一貫して大きな振幅を持ち、除去による波動関数の振幅応答も大きかった。本追加実験では、共通入力に欠けている方向を補う効果と、Fourier候補を学習する構成内での寄与を切り分ける。')
    table(['ID','条件','扱い・入力の違い'],[['A','baseline','既存3 seedを再利用。位置4成分＋擬スピン1成分'],['B','baseline＋対角成分','新規3 seed。Aにcos(θ1−θ2), sin(θ1−θ2)を追加'],['C','learned Fourier＋L1','既存3 seedを再利用。固定6候補の振幅を学習'],['D','Cの対角成分を無効化','新規3 seed。候補(1,−1)の2特徴を常時ゼロに固定']],[9,53,111])
    para('位置はセルの分数座標(u,v)からθ=(2πu,2πv)へ変換する。Aの共通位置特徴はsinθ1, sinθ2, cosθ1, cosθ2。Cの候補は(1,0),(0,1),(1,−1),(6,0),(0,6),(6,−6)。候補ごとのcos/sin対に同じ学習振幅を掛ける。L1係数は0.01。')
    sub('共通条件と比較可能性')
    table(['項目','設定'],[['物理系','N=36、rs=25、λ=0.72、Δ=25、Coulomb相互作用。同じセル・HF軌道表・Hamiltonian'],['ネットワーク','幅32、4 heads、1層、20成分にゼロ埋めして射影'],['パラメータ数','全条件11439。活性特徴数・関数クラスは条件で異なる'],['学習','seed=101/202/303、10000更新、Adam学習率0.002、64 walkers、2 sweeps/更新'],['初期状態','NN軌道readoutをゼロ化した共通HF状態。Jastrowなし'],['対照確認','新規6条件で初期振幅・位相・burn-in後walker・第1更新Eを既存対照と照合']],[33,140])
    para('Bは既存baselineの5座標と対応する射影行を保持し、対角2成分を末尾に追加した。Dは候補の振幅をゼロ初期化し、固定マスクとゼロ勾配で復活を防いだ。残る候補のL1係数はCと同じである。追加自由度の活性化は介入そのものであり、完全に同じ関数クラスの比較ではない。')
    para('Dの3履歴を再確認し、全10000更新で無効化振幅が厳密にゼロであることを検証した。既存6条件は再学習せず、測定・PH・出力介入のみ同じ手順で再実施した。')

    heading('2．エネルギーの低下と対応比較')
    para('以下の学習エネルギーは、学習中のwalkerで計算した末尾1000更新の平均である。独立した平衡分布からの期待値とは保証されない。数値は内部単位で、独立評価のHa/電子と混同しない。')
    table(['ID','条件','学習E：3 seed平均 ± 標準偏差'],group_body,[9,88,76])
    table(['seed','B−A：単純追加','D−C：最初から無効化'],[[str(seed),f"{next(x['training_delta'] for x in contrasts if x['seed']==seed and x['variant']=='baseline_plus_diagonal'):+.3f}",f"{next(x['training_delta'] for x in contrasts if x['seed']==seed and x['variant']=='fourier_l1_without_diagonal'):+.3f}"] for seed in [101,202,303]]+[['平均 ± seed SD',f"{cstats['baseline_plus_diagonal']['mean']:+.3f} ± {cstats['baseline_plus_diagonal']['sd']:.3f}",f"{cstats['fourier_l1_without_diagonal']['mean']:+.3f} ± {cstats['fourier_l1_without_diagonal']['sd']:.3f}"]],[37,68,68])
    para(f"Bの改善はseed101のみで、単純追加による安定した利点は確認できない。Dは全seedでCより高い学習Eとなり、残る特徴による再学習ではこの予算内の性能を回復できなかった。両側の対応符号反転検定はB−Aがp={cstats['baseline_plus_diagonal']['p_two_sided_signflip']:.2f}、D−Cがp={cstats['fourier_l1_without_diagonal']['p_two_sided_signflip']:.2f}であり、5%有意差の主張には使わない。")
    image('training_curves','図2：100更新ごとの平均学習E。各色はseed。曲線の低下・横ばいは学習過程の観察であり、平衡分布での収束証明ではない。')

    heading('3．パーシステントホモロジー（PH）')
    para('同じ固定8配置・擬スピンを、更新0/1000/3000/6000/10000で評価した。QK・実集約重みScreening・距離Windowは対称化重みwのflag filtration（u=1−w）。FeatureVRは単位化した粒子特徴間のEuclidean距離/2を使う。H0/H1、F2係数で計算した。')
    para('表・図は正の辺重みを順位へ変換したrank指標である。各配置・各headでPHを求めてから平均した。H1面積はu∈[0,1]のBetti数積分であり、∞のbarは保存図に残したまま観測区間だけ積分へ寄与させた。FeatureVRにはhead軸はない。')
    body=[]
    for lab in LABELS:
        b=[]
        for graph in ['QK','Screening','FeatureVR']:
            before=next(v['mean'] for v in temporal if v['label']==lab and v['graph']==graph and v['step']==0)
            after=next(v['mean'] for v in temporal if v['label']==lab and v['graph']==graph and v['step']==10000)
            b.append(f'{before:.3f} → {after:.3f}')
        body.append([IDS[lab]]+b)
    table(['ID','QK：前→後','Screening：前→後','FeatureVR：前→後'],body,[9,55,55,54])
    image('ph_trajectories','図3：PH指標の軌跡。線は3 seed平均、帯はseed標準偏差。異なるグラフ間の面積の絶対値を性能点として比較しない。')
    para('CだけでなくDも3 seedすべてでQKのH1面積が増加した。Dの学習Eは全seedで悪化しているため、「QKのループが増えた」だけでは学習性能の改善を判定できない。BのQK変化はseedで方向が異なった。')
    para('固定WindowのPHは全条件・全checkpointで同一だった。したがって、QK等の変化は固定窓そのものの変化ではない。一方、順位正規化後にも支持辺・次数・入力幾何の影響は残る。PHは内部構造の記述であり、多体波動関数の物理的トポロジーやChern数とは同一視しない。')

    heading('4．波動関数への機能的寄与')
    para('最終checkpointの同じ8配置で、対角成分または全Attention headの出力を除去した。log|ψ|差から全体規格化を除いた標準偏差を振幅形状RMSとし、位相差は全体位相を円周平均で除いてRMSを算出した。固定入力上の応答で、Born重み付きfidelityではない。')
    body=[]
    for group in ['B','C','D']:
        for seed in [101,202,303]:
            v=next(r for r in function_table if r['group']==group and r['seed']==seed)
            body.append([group,str(seed),f"{v['diagonal_logabs_rms']:.3f}",f"{v['diagonal_phase_rms']:.3f}"])
    table(['条件','seed','対角除去：振幅RMS','対角除去：位相RMS（rad）'],body,[14,20,66,73])
    image('functional_response','図4：固定入力上の出力応答。Dの対角除去応答ゼロは、既にその入力を無効化した対照として整合している。')
    para('Bでは対角成分の除去が3 seedとも出力を変えた。したがって追加特徴は実際に利用されたが、それだけでは学習Eの一貫した改善につながらなかった。Cでも対角成分は大きな応答を持つ。Dのゼロ応答は固定マスクの動作確認であり、物理的な重要性ゼロを意味しない。')
    para('Dでは残る候補への依存が観察され、最大の振幅除去応答はseed101で(0,1)、seed202で(6,−6)、seed303で(0,1)だった。特徴の代替利用は起きるが、この予算ではCの学習Eまで回復しなかった。全head除去の応答は全12条件で非ゼロであり、今回の4条件には前回の極端なAttention弱依存例に相当するケースは見られなかった。')

    heading('5．独立評価と収束診断')
    para('最終パラメータを固定し、独立初期配置と学習末尾walkerの2種類から各16鎖、計32鎖を測定した。burn-in2048、thinning4、最小2048・最大8192時点を予定し、各モデル4時間で打ち切った。実際の測定数は6429〜6637時点、約20.6〜21.2万個のlocal-energy値/モデルである。')
    para('判定基準はエネルギーRhat≦1.05、構造因子Rhat≦1.1、ESS≧400、エネルギーSEM≦1e-4 Ha/電子、2開始群の等価性上限≦5e-4 Ha/電子等。すべてを満たした測定だけを比較可能とする。','small')
    table(['ID/seed','時点数','評価E（Ha/電子）','SEM','Rhat'],[[IDS[m['label']]+'/'+str(m['seed']),str(measurements[m['name']]['actual_times']),f"{m['energy_Ha_per_electron']:.6f}",f"{m['energy_sem']:.6f}",f"{m['rhat']:.3f}"] for m in ordered],[21,20,53,40,39])
    image('measurement_mixing','図5：開始配置別の平均と混合診断。学習末尾から開始した鎖のEが全12条件で低く、開始配置への依存が残っている。表示平均・SEMは未収束鎖の記述値。')
    para('全12条件でエネルギーRhatが1.106〜1.521となり、基準1.05を超えた。構造因子の最大Rhatも1.531〜2.157で基準1.1を超えた。両開始群の差が全条件で同方向に残っており、単なる標本数増加だけでは混合問題が解消しなかった。独立評価の順位・差の有意性は本書の結論に採用しない。')

    heading('6．結論・限界・継続GPU計算')
    table(['論点','現時点の判断'],[['対角成分の利用','固定配置上で出力への依存を再確認。係数の大きさだけの議論を超える'],['学習への寄与','D−Cが全3 seedで正。現在の構成・10000更新での寄与を支持'],['単純な入力不足説','B−Aは符号が揃わず、単純追加の安定した効果は確認できない'],['PHの評価','内部の構造変化は測れる。H1増加だけを性能改善の根拠にしない'],['物理的な優劣','独立評価は全12条件で未収束。基底状態・相の優劣は未確定']],[34,139])
    sub('限界と次に必要な確認')
    para('繰り返しは3 seedであり、各配置・head・時刻を独立な学習試行として数えない。活性特徴・関数クラスは条件間で異なる。独立鎖には開始配置依存が残り、SEMは非平衡バイアスを保証しない。固定probeは8配置のみで、出力介入の結果を全Born分布へ一般化できない。')
    para('次の優先課題は、固定した同じ波動関数で開始配置による差が残る原因を調べ、位置・擬スピンの混合を改善することである。今回の結果から、新たな学習や測定を追加投入する操作は行っていない。')
    sub('別系統のGPUジョブ：途中状況')
    names={'converged_to_protocol':'収束基準通過','unconverged_budget_limit':'更新上限・未収束','failed':'実行エラー','in_progress':'実行中','not_started':'待機中'}
    table(['状態','ケース数'],[[names[k],str(gpu_count.get(k,0))] for k in names],[100,73])
    para('GPUは別プロトコルの54ケースで、許容差は0.0005 Ha/電子。CPU追加実験の結果へ混ぜて集計しない。収束基準通過も基底状態・物理相の証明ではない。GPUの状態は本書作成時点のスナップショットである。','small')

    heading('付録：データ・再現性・提出物')
    para('新規学習のSlurm配列IDは3389、監視・自動集計は3390。全6 taskと監視jobがCOMPLETED、終了コード0。学習完了は11月10日14:38〜14:46、その後に各workerが新規1条件と既存1条件を最大各4時間測定した。最終集計は同日22:46。全体48時間の強制停止は発動しなかった。')
    para('比較のソース・HF入力・参照checkpoint等573ファイルを事前に固定した。提出レポート作成時にも元の固定manifestと照合する。学習エネルギー末尾平均を全12履歴から再計算し、既存summaryと一致すること、Dの無効化振幅が全更新でゼロであることを検証した。')
    table(['提出ファイル','内容'],[['本PDF／同名Markdown','同じ本文・表・図の報告書'],['data/group_summary.csv','条件ごとの学習E平均・seed標準偏差'],['data/paired_contrasts.csv','同一seedの対応差。未収束の独立評価差は空欄'],['data/measurement_summary.csv','独立評価・開始配置差・収束診断'],['data/ph_group_summary.csv','PHの条件別・checkpoint別平均とseed標準偏差'],['data/functional_response.csv','対角成分とhead除去による出力形状の応答'],['data/input_manifest.json','本レポートが参照した入力のSHA256'],['data/validation.json','数値・不変性・PDF検証の記録']],[69,104])
    sub('主要な一次保存先')
    for p in [RUN/'protocol.json',RUN/'plan.json',RUN/'results.json',RUN/'paired_contrasts.csv',RUN/'ph_trajectory.csv',RUN/'measurement',RUN/'analysis',ROOT/'research_framework/results/2026-11-06_followup/n36_cluster',GPU]:
        para(str(p.relative_to(ROOT)),'small')
    para('提出一式には集計表・図・測定summaryを同梱する。元の生標本・checkpoint・全ソースは上記の実験保存先に保持している。日本語フォントはNoto Sans JPを埋め込み、SIL Open Font Licenseをassets/OFL.txtに同梱した。外部資料から科学的な主張を追加せず、保存済みの実測結果とコードを根拠とした。','small')
    filename='N36_diagonal_feature_report_20261112'
    def footer(canvas,doc):
        canvas.saveState();canvas.setStrokeColor(colors.HexColor('#ccd6df'));canvas.line(18*mm,16*mm,192*mm,16*mm)
        canvas.setFont('JP',7);canvas.setFillColor(colors.HexColor('#52616c'))
        canvas.drawString(18*mm,11*mm,'N36 対角低波数の追加対照実験｜2026-11-12')
        canvas.drawRightString(192*mm,11*mm,str(doc.page));canvas.restoreState()
    doc=SimpleDocTemplate(str(OUT/f'{filename}.pdf'),pagesize=A4,rightMargin=18*mm,leftMargin=18*mm,
                          topMargin=18*mm,bottomMargin=22*mm,title='N=36電子系の入力表現と対角低波数の追加対照実験',author='研究実験レポート')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    (OUT/f'{filename}.md').write_text('\n'.join(md))
    # Export exact evidence tables and per-model measurement metadata.
    csvsave('group_summary.csv',[dict(group=IDS[k],label=k,**v) for k,v in stats.items()])
    csvsave('measurement_summary.csv',[dict(group=IDS[m['label']],seed=m['seed'],status=measurements[m['name']]['status'],
        actual_times=measurements[m['name']]['actual_times'],energy_Ha_per_electron=m['energy_Ha_per_electron'],sem=m['energy_sem'],rhat=m['rhat'],
        fresh_mean=measurements[m['name']]['diagnostics']['starts'][0]['mean'],trained_mean=measurements[m['name']]['diagnostics']['starts'][1]['mean'],
        start_bound=measurements[m['name']]['diagnostics']['start_equivalence_bound_Ha_per_electron'],sampling_pass=m['sampling_pass']) for m in ordered])
    csvsave('functional_response.csv',function_table);csvsave('ph_group_summary.csv',temporal)
    for name in ['summary.csv','paired_contrasts.csv','ph_trajectory.csv','protocol.json','plan.json','results.json','launch_check.json','budget_clock.json','frozen_sha256.json']:
        shutil.copy2(RUN/name,OUT/'data'/name)
    (OUT/'data/measurement_metadata').mkdir()
    for m in models:shutil.copy2(RUN/'measurement'/m['name']/'summary.json',OUT/'data/measurement_metadata'/(m['name']+'.json'))
    read(Path(__file__));shutil.copy2(Path(__file__),OUT/'build_report.py')
    # Match the pre-run manifest without writing to the experiment.
    fixed=js(RUN/'frozen_sha256.json')
    for p,expected in fixed.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==expected,p
    for p,v in manifest.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==v['sha256'],p
    savejson('data/input_manifest.json',manifest)
    savejson('data/analysis_summary.json',dict(created_at=now.isoformat(),cpu_completed=6,measurement_pass=0,
        total_measurements=12,group_statistics=stats,paired_statistics=cstats,gpu=gpu_count,disabled_feature_checks=mask_checks))
    savejson('data/validation.json',dict(frozen_files_unchanged=len(fixed),report_inputs_unchanged=len(manifest),
        all12_training_averages_reproduced=True,disabled_feature_zero_all30000_updates=True,
        all6_workers_complete=True,no_measurement_qualified_for_physical_ranking=True))
    (OUT/'report_environment.txt').write_text(subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True).stdout)
    print(json.dumps(dict(output=str(OUT),pdf=str(OUT/f'{filename}.pdf'),paired_statistics=cstats,gpu=gpu_count),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
