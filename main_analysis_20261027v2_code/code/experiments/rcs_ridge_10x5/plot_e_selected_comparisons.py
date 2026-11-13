"""Matched outer-test C-index plots; inferential intervals come from Rubin pooling."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / 'results/rcs_ridge_10x5/20261102_v1/final'
COX_RESULT = ROOT / 'results/cox_e_selected_10x5/20261101_v1/final_cindex_and_pvalues.csv'
OUT = RESULT / 'figures'
COLORS = {'ordinary_cox': '#356d98', 'rcs_ridge_cox': '#c88532', 'E_selected': '#287c64'}
NAMES = {'ordinary_cox': '普通 Cox', 'rcs_ridge_cox': 'RCS＋ridge-Cox', 'E_selected': 'E_selected'}
KEY = ['repeat_id', 'fold', 'imputation']


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def prepare():
    if not (RESULT.parent / 'COMPLETE.json').is_file():
        raise ValueError('RCS full experiment is incomplete')
    source = RESULT / 'model_cindex.csv'
    d = pd.read_csv(source)
    expected = {'ordinary_cox', 'rcs_ridge_cox', 'E_selected'}
    if set(d.model) != expected or len(d) != 3000 or d.duplicated(KEY + ['model']).any():
        raise ValueError('Expected three methods with 1000 unique outer models each')
    w = d.pivot(index=KEY, columns='model', values='C_index').sort_index()
    if len(w) != 1000 or w.isna().any().any():
        raise ValueError('Missing paired model C-index')
    combinations = [(r, f, j) for r in range(1, 11) for f in range(1, 6) for j in range(1, 21)]
    if list(w.index) != combinations:
        raise ValueError('Incorrect 10×5×20 model coverage')
    p_rcs = pd.read_csv(RESULT / 'cindex_and_pvalues.csv').set_index('model')
    p_cox = pd.read_csv(COX_RESULT).set_index('model')
    for method in expected.intersection(p_cox.index):
        if abs(p_rcs.loc[method, 'estimate'] - p_cox.loc[method, 'estimate']) > 1e-12:
            raise ValueError(f'Mismatched pooled C-index source: {method}')
    stats = {
        'ordinary_cox': dict(estimate=float(p_cox.loc['E_selected_minus_cox', 'estimate']),
                             lower=float(p_cox.loc['E_selected_minus_cox', 'CI95_lower']),
                             upper=float(p_cox.loc['E_selected_minus_cox', 'CI95_upper']),
                             p=float(p_cox.loc['E_selected_minus_cox', 'p_Rubin'])),
        'rcs_ridge_cox': dict(estimate=-float(p_rcs.loc['rcs_ridge_cox_minus_E_selected', 'estimate']),
                              lower=-float(p_rcs.loc['rcs_ridge_cox_minus_E_selected', 'CI95_upper']),
                              upper=-float(p_rcs.loc['rcs_ridge_cox_minus_E_selected', 'CI95_lower']),
                              p=float(p_rcs.loc['rcs_ridge_cox_minus_E_selected', 'p_Rubin'])),
    }
    for comparator, s in stats.items():
        delta = w['E_selected'] - w[comparator]
        if abs(s['estimate'] - (p_rcs.loc['E_selected', 'estimate'] - p_rcs.loc[comparator, 'estimate'])) > 1e-10:
            raise ValueError('Paired pooled difference does not equal method difference')
        if not (s['lower'] < s['estimate'] < s['upper'] and s['lower'] > 0):
            raise ValueError('Unexpected CI orientation')
        s['model_mean_difference'] = float(delta.mean())
        s['positive_models'] = int((delta > 0).sum())
        s['n_models'] = len(delta)
    return w, stats


def font_setup():
    font = Path('C:/Windows/Fonts/msyh.ttc')
    if font.is_file():
        font_manager.fontManager.addfont(str(font))
        name = font_manager.FontProperties(fname=str(font)).get_name()
        plt.rcParams['font.sans-serif'] = [name, 'DejaVu Sans']
    plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.facecolor': 'white'})


def main():
    w, stats = prepare()
    font_setup()
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(15.4, 9.4), constrained_layout=False)
    grid = GridSpec(2, 3, figure=fig, width_ratios=(1.0, 1.12, .88),
                    left=.065, right=.975, top=.86, bottom=.16, wspace=.25, hspace=.48)
    fig.suptitle('E_selected 与普通 Cox、RCS＋ridge-Cox 的配对外层测试比较',
                 fontsize=17, fontweight='bold', y=.965)
    fig.text(.5, .91, '10 次重复 × 5 折 × 20 份插补；每种方法 1000 个同折同插补模型',
             ha='center', fontsize=11, color='#4b5861')
    for row, comparator in enumerate(('ordinary_cox', 'rcs_ridge_cox')):
        label = NAMES[comparator]
        x = w[comparator].to_numpy(float)
        y = w['E_selected'].to_numpy(float)
        delta = y - x
        s = stats[comparator]
        scatter = fig.add_subplot(grid[row, 0])
        scatter.scatter(x, y, s=11, color=COLORS['E_selected'], alpha=.23,
                        edgecolors='none', rasterized=True)
        scatter.plot([.61, .86], [.61, .86], color='#58616a', linestyle='--', linewidth=1.1)
        scatter.set(xlim=(.62, .85), ylim=(.62, .85), xlabel=f'{label} 的 C-index',
                    ylabel='E_selected 的 C-index')
        scatter.set_aspect('equal', adjustable='box')
        scatter.set_title(f'{chr(65 + row)}  同一模型单元的配对结果', loc='left', fontweight='bold')
        scatter.text(.03, .97, f'E_selected 更高：{s["positive_models"]}/1000',
                     transform=scatter.transAxes, va='top', color=COLORS['E_selected'],
                     bbox=dict(boxstyle='round,pad=.32', facecolor='white', edgecolor='#dce4e9'))
        scatter.grid(alpha=.12)

        histogram = fig.add_subplot(grid[row, 1])
        limits = (-.08, .10)
        counts, edges = np.histogram(delta, bins=np.linspace(*limits, 37))
        mids = (edges[1:] + edges[:-1]) / 2
        bars = histogram.bar(mids, counts, width=np.diff(edges) * .91, align='center')
        for bar, mid in zip(bars, mids):
            bar.set_facecolor(COLORS['E_selected'] if mid >= 0 else '#b7c2c9')
        histogram.axvline(0, color='#333b42', linewidth=1.35)
        histogram.axvline(s['model_mean_difference'], color=COLORS['E_selected'],
                          linestyle='--', linewidth=1.6)
        histogram.set(xlim=limits, xlabel=f'模型级差值：E_selected − {label}', ylabel='模型数')
        histogram.set_title('1000 个配对差值的分布', loc='left', fontweight='bold')
        histogram.grid(axis='y', alpha=.15)

        effect = fig.add_subplot(grid[row, 2])
        effect.axvspan(0, .032, color='#eef7f3', zorder=0)
        effect.axvline(0, color='#333b42', linewidth=1.35)
        effect.plot([s['lower'], s['upper']], [0, 0], color=COLORS['E_selected'], linewidth=3)
        effect.plot([s['lower'], s['lower']], [-.09, .09], color=COLORS['E_selected'], linewidth=2)
        effect.plot([s['upper'], s['upper']], [-.09, .09], color=COLORS['E_selected'], linewidth=2)
        effect.scatter([s['estimate']], [0], s=90, color=COLORS['E_selected'], zorder=3)
        effect.set(xlim=(-.005, .032), ylim=(-.6, .8), yticks=[],
                   xlabel=f'汇总差值：E_selected − {label}')
        effect.set_title('患者聚类＋Rubin 合并', loc='left', fontweight='bold')
        effect.text(.03, .85,
                    f'Δ C = {s["estimate"]:.5f}\n95% CI  {s["lower"]:.5f}～{s["upper"]:.5f}\np = {s["p"]:.6f}',
                    transform=effect.transAxes, va='top', fontsize=11,
                    bbox=dict(boxstyle='round,pad=.45', facecolor='white', edgecolor='#dce4e9'))
        effect.grid(axis='x', alpha=.12)
    fig.text(.065, .075,
             '左、中图展示相关的 1000 个模型结果；右图的区间和 p 值按患者聚类并对 20 份插补做 Rubin 合并。',
             fontsize=10, color='#4b5861')
    fig.text(.065, .048,
             '两项差值的 95% CI 均未跨 0；同一队列曾被用于模型开发，不能将这些结果视为独立外部验证。',
             fontsize=10, color='#4b5861')
    png = OUT / 'e_selected_paired_significance.png'
    pdf = OUT / 'e_selected_paired_significance.pdf'
    fig.savefig(png, dpi=240)
    fig.savefig(pdf)
    plt.close(fig)
    meta = {'created_at': datetime.now(timezone.utc).isoformat(),
            'plot_code_sha256': sha(Path(__file__)),
            'source_model_cindex_sha256': sha(RESULT / 'model_cindex.csv'),
            'source_rcs_pvalues_sha256': sha(RESULT / 'cindex_and_pvalues.csv'),
            'source_cox_pvalues_sha256': sha(COX_RESULT),
            'comparison': stats, 'plot_files': {p.name: sha(p) for p in (png, pdf)},
            'inference_scope': 'patient-clustered Rubin pooled; 1000 models are correlated'}
    (OUT / 'figure_manifest.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({k: {'pooled_difference': v['estimate'], 'CI95': [v['lower'], v['upper']],
                          'Rubin_p': v['p'], 'positive_models': v['positive_models']}
                      for k, v in stats.items()}, ensure_ascii=False, indent=2))
    print(png)


if __name__ == '__main__':
    main()
