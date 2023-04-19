from rdkit import Chem
from rdkit.Chem import AllChem
from copy import deepcopy
from rdkit.Chem import rdDistGeom
from rdkit import DistanceGeometry
from scipy.optimize import minimize
from .embed_patches import correct, ALLIL
import numpy as np
import traceback

def add_conformer_from_xyz(mol, XYZ, clear=False):
    if clear:
        mol.RemoveAllConformers()
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for i, (x,y,z) in enumerate(XYZ):
        conformer.SetAtomPosition(i, [x,y,z])
    return mol.AddConformer(conformer, assignId=True)

def embed_within_sphere(mol, radius, nconfs=100, seed=0xf00d, num_threads=1):
    D = 2*radius
    bounds = rdDistGeom.GetMoleculeBoundsMatrix(mol)
    ps = rdDistGeom.EmbedParameters()
    ps.useExpTorsionAnglePrefs = False
    ps.useBasicKnowledge = False #??True??
    ps.randomSeed = seed
    ps.useRandomCoords=True
    ps.pruneRmsThresh=0.1
    ps.enforceChirality=True
    ps.forceTol=0.01

    N = bounds.shape[0]
    for i in range(N):
        for j in range(i+1,N):
            bounds[i,j] = min([bounds[i,j], 11])

    smoothing_status = DistanceGeometry.DoTriangleSmoothing(bounds)
    assert smoothing_status
    ps.SetBoundsMat(bounds)
    ps.useRandomCoords = True
    ps.numThreads = num_threads
    cids = rdDistGeom.EmbedMultipleConfs(mol, nconfs, ps)
    return cids

def confinement_func_generator(copied_mol, r, harmonic=False, k=1, A=1):
    prp = AllChem.MMFFGetMoleculeProperties(copied_mol)
    
    def mmff_with_confinement_V(flatten_coords):
        X = flatten_coords.reshape(-1,3)
        cid= add_conformer_from_xyz(copied_mol, X, True)
        ff = AllChem.MMFFGetMoleculeForceField(copied_mol, prp, confId=cid)
        ene, grad = ff.CalcEnergy(), np.array(ff.CalcGrad())
        if r!=None:
            rs = X - X.mean(axis=0)
            rs_mod = np.sqrt((rs * rs).sum(axis=1))
            rs_norm = -rs/rs_mod.reshape(-1,1)
            delta = rs_mod - r
            if harmonic:
                Es = np.where(delta>0, delta, 0)
                Fs_mod = Es.reshape(-1,1)*k
                Es = Es*Es*k/2
            else:
                Es = np.exp(delta*k)*A
                Fs_mod = (Es*k).reshape(-1,1)
            Fs_mod = Fs_mod
            ene += Es.sum()
            grad += (rs_norm*Fs_mod).reshape(-1)
        return ene, grad
    
    return mmff_with_confinement_V

def optimize_within_sphere(mol, method='bfgs', r=None, k=1, A=1):
    conformers_flatten = [conf.GetPositions().reshape(-1) for conf in mol.GetConformers()]
    assert conformers_flatten!=[]
    copied_mol = deepcopy(mol)
    func = confinement_func_generator(copied_mol, r, k=k, A=A)
    results = [minimize(lambda x:func(x)[0], x0, jac=lambda x:func(x)[1], method=method)
               for x0 in conformers_flatten]
    results.sort(key=lambda x: x.fun)
    copied_mol.RemoveAllConformers()
    ids = []
    for r in results:
        xyz = r.x.reshape(-1,3)
        new_id = add_conformer_from_xyz(copied_mol, xyz)
        ids.append(new_id)
    
    return copied_mol, ids, results

def get_confined_conformers(smiles, radius, mol=None, num_confs=100, numThreads=50, opt=True):
    '''embeds smiles in sphere of given radius; returns list of (molBlcoks, energy) tuples and status string'''
    try:
        mol = mol if not (mol is None) else Chem.AddHs(Chem.MolFromSmiles(smiles))
        cids = embed_within_sphere(mol, radius, nconfs=num_confs, num_threads=numThreads)
        mol2 = deepcopy(mol)
        if opt:
            mol2, ids, results = optimize_within_sphere(mol2, r=radius)
            results = [(Chem.MolToMolBlock(mol2, confId=i), x.fun) for i, x in zip(ids, results)]
            results.sort(key=lambda x: x[1])
        else:
            results = [(Chem.MolToMolBlock(mol2, confId=i), None) for i in cids]

        status = 'ok'
    except:
        results = []
        status = traceback.format_exc().replace('\n','EOL')

    return results, status

def opt_ff_with_sphere(smiles, mol=None, radius=6, num_confs=100, numThreads=50):
    '''embeds smiles in sphere of given radius; returns list of (molBlcoks, energy) tuples and status string'''
    try:
        mol = mol if not (mol is None) else Chem.AddHs(Chem.MolFromSmiles(smiles))
        cids = embed_within_sphere(mol, radius, nconfs=num_confs, num_threads=numThreads)
        mol_e = Chem.EditableMol(mol)
        idx = mol_e.AddAtom(Chem.Atom(6))
        mol2 = mol_e.GetMol()
        Chem.SanitizeMol(mol2)
        results = []
        prp = AllChem.MMFFGetMoleculeProperties(mol, mmffVerbosity=0)
        prp2 = AllChem.MMFFGetMoleculeProperties(mol2, mmffVerbosity=0)
        for cid in cids:
            c = mol2.GetConformer(cid).GetPositions()[:-1].mean(axis=0)
            mol2.GetConformer(cid).SetAtomPosition(idx, list(c))
            ff = AllChem.MMFFGetMoleculeForceField(mol2, prp2, confId=cid)
            ff.AddFixedPoint(idx)
            for i in range(mol.GetNumAtoms()):
                ff.AddDistanceConstraint(i, idx, 0., radius, 5000.0)
            status = ff.Minimize(maxIts=1000)
            conf = mol.GetConformer(cid)
            conf2 = mol2.GetConformer(cid)
            for i in range(mol.GetNumAtoms()):
                conf.SetAtomPosition(i, conf2.GetAtomPosition(i))
            correct(mol, cid, pattern=ALLIL)
            ff = AllChem.MMFFGetMoleculeForceField(mol, prp, confId=cid)
            e = ff.CalcEnergy()
            results.append((Chem.MolToMolBlock(mol, confId=cid), e))

        results.sort(key=lambda x: x[1])
        status = 'ok'
    except:
        results = []
        status = traceback.format_exc().replace('\n','EOL')

    return results, status
