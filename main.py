from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from ellipsoid_cavity import EllipsoidTool, embed_dih, get_E, dist_vector_ellipsoid, standardize
import logging
import typer 

app = typer.Typer()

@app.command()
def main(smiles:str, semiaxis_x:float=5, semiaxis_y:float=5, semiaxis_z:float=7, num_conf:int=50, output:str='confined.mol'):
    r = [semiaxis_x, semiaxis_y, semiaxis_z]
    r.sort(key=lambda x:-x)
    r = np.array(r)
    conformers, results = embed_dih(smiles, n_conformers=num_conf, r=r,
                                    method='cobyla', use_scipy=True, exp=False, use_tqdm=True)
    #min_e, min_dst
    result = []
    for i,c in enumerate(conformers):
        m = Chem.MolFromMolBlock(c, removeHs=False)
        e= get_E(m, 0)
        x = dist_vector_ellipsoid(m.GetConformer(0).GetPositions(), r)
        x = np.linalg.norm(x, axis=1).max()
        result.append((c, x, e, results[i].message))

    if result!=[]:
        result.sort(key=lambda x: x[1:])
        conf, dist, energy, message = result[0]
        m = Chem.MolFromMolBlock(conf, removeHs=False)
        _, axes, _ = EllipsoidTool().getMinVolEllipse(P=m.GetConformer(0).GetPositions()) 
        axes.sort()
        r.sort()
        print('minimum energy:', round(energy,1))
        print('maximum distance outside ellipsoid:', dist)
        print('requested semiaxes:', r.round(1))
        print('semiaxes of final MVEE:', axes.round(1))
        print('fits?', (axes<=r).all())
        with open(output, 'w') as f:
            f.write(conf)
        

if __name__=='__main__':
    app()
