import json, collections, numpy as np, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(ROOT, 'results', 'phase2_ablation_strong.json')))
recs = d['records']
agg = collections.defaultdict(lambda: collections.defaultdict(lambda: {'dJ': [], 'ds': []}))
for r in recs:
    agg[(r['n'], r['lam'])][r['method']]['dJ'].append(r['dJ'])
    agg[(r['n'], r['lam'])][r['method']]['ds'].append(r['dsvm'])
order = ['Random-mRMR', 'Greedy-mRMR', 'SA', 'MeanField(cl)',
         'VQC(no-ent)', 'VQC(ring)', 'VQC(all)', 'QAOA-mRMR']
out = [f"device {d['device']}  records {len(recs)}  cells {len(agg)}"]
# also pooled over all cells
pooled = collections.defaultdict(lambda: {'dJ': [], 'ds': []})
for cell in sorted(agg):
    out.append(f"=== n,lam = {cell} ===")
    for m in order:
        v = agg[cell][m]; dJ = np.array(v['dJ']); ds = np.array(v['ds'])
        pooled[m]['dJ'] += v['dJ']; pooled[m]['ds'] += v['ds']
        out.append(f"  {m:<15} dJ={dJ.mean():+.3f} (>0:{(dJ>1e-6).mean():.2f})  dSVM={ds.mean():+.3f}")
out.append("=== POOLED over all cells ===")
for m in order:
    dJ = np.array(pooled[m]['dJ']); ds = np.array(pooled[m]['ds'])
    out.append(f"  {m:<15} dJ={dJ.mean():+.4f} (>0:{(dJ>1e-6).mean():.2f})  dSVM={ds.mean():+.4f} (n={len(dJ)})")
open(os.path.join(ROOT, 'results', 'ablation_summary.txt'), 'w').write('\n'.join(out))
print('\n'.join(out))
