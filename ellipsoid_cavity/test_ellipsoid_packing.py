from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from .ellipsoid import EllipsoidTool
from .confinement_dih import standardize, embed_dih, get_E, harmonic_potential, dist_vector_ellipsoid
from .resources_monitor import run_and_monitor_status
import logging
import pandas as pd
from itertools import product
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(message)s')

s,p = 'CC[C@@H](C)CCCCCC[C@H](C)CCCCCO>>CC[C@H]1COC[C@@H]2C[C@H]3C[C@@H](C)CCC[C@H]3C[C@@H]12'.split('>>')
p = Chem.AddHs(Chem.MolFromSmiles(p))
AllChem.EmbedMolecule(p)

center, axes, orientation = EllipsoidTool().getMinVolEllipse(P=p.GetConformer(0).GetPositions(), tolerance=0.01)
logging.info(f'target ellipsoid: center={center.round(3)} axes={axes.round(3)}')

standardize(p, 0)
center2, axes2, orientation2 = EllipsoidTool().getMinVolEllipse(P=p.GetConformer(0).GetPositions(), tolerance=0.01)
logging.info(f'standarization check: center: {center2.round(3)}, axes_dev={abs(axes2-axes).round(3)}, orientation: {orientation2.round(3)}')

def test_embedding(smiles, r=axes, n_conformers=50, randomize=False, exp=False, method='cobyla'): 
    conformers, results = embed_dih(smiles, n_conformers=n_conformers, r=r, method=method,
                             randomize=randomize, use_scipy=True, exp=exp, random_step=60, use_tqdm=True)
    #min_e, min_dst
    result = []
    for i,c in enumerate(conformers):
        m = Chem.MolFromMolBlock(c, removeHs=False)

        e= get_E(m,0)
        x = dist_vector_ellipsoid(m.GetConformer(0).GetPositions(), r)
        x = np.linalg.norm(x, axis=1).max()
        result.append((c,x,e,results[i].message))
    if result!=[]:
        result.sort(key=lambda x: x[1:])
        result = result[0]
    return result

axes_rev = np.array(list(reversed(axes)))

results = []
for met, r,ex in product(['cobyla', 'cg'], [True, False], [True, False]):
    logging.info(f'random={r}, exp={ex}')
    resp, stats = run_and_monitor_status(test_embedding, args=(s,), kwargs=dict(method=met, r=axes_rev+1.5, randomize=r, exp=ex), interval=2)
    if resp:
        structure, d, e, msg = resp
    else:
        structure, d, e, msg = '', None, None, ''
    datum = {'structure':structure.replace('\n','EOL'), 'max_dst':d, 'energy':e, 'time':stats['wall_time'], 'random':r, 'exp':ex,
            'method':met, 'message':msg}
    for k in ['mean','max']:
        cpu, mem = stats['process'][k] - stats['background'][k]
        datum[f'cpu_{k}'] = cpu
        datum[f'mem_{k}'] = mem
    results.append(datum)

pd.DataFrame(results).to_csv('ellipsoid_dih2.csv', sep=';')
