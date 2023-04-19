from rdkit import Chem
from rdkit.Chem import AllChem
from copy import deepcopy
from rdkit.Chem import rdDistGeom
from rdkit import DistanceGeometry
from scipy.optimize import minimize
import numpy as np
import traceback

def sphere(bounds, mol, radius):
    if hasattr(radius, '__iter__'):
        D = 2*max(radius)
    else:
        D = 2*radius
    N = bounds.shape[0]
    for i in range(N):
        for j in range(i+1,N):
            bounds[i,j] = min([bounds[i,j], 11])
    return bounds

def smooth(bounds, mol):
    smoothing_status = DistanceGeometry.DoTriangleSmoothing(bounds)
    assert smoothing_status
    return bounds

def raw_embed(mol, nconfs=100, seed=0xf00d, num_threads=1, modifiers=[]):
    bounds = rdDistGeom.GetMoleculeBoundsMatrix(mol)
    ps = rdDistGeom.EmbedParameters()
    ps.useExpTorsionAnglePrefs = False
    ps.useBasicKnowledge = False#??True??
    ps.randomSeed = seed
    for modifier in modifiers:
        bounds = modifier(bounds, mol)

    ps.SetBoundsMat(bounds)
    ps.useRandomCoords = True
    ps.numThreads = num_threads
    cids = rdDistGeom.EmbedMultipleConfs(mol, nconfs, ps)
    return cids

def filter_nei(mol, idx, forbidden=[], min_len=0):
    result = []
    for atom in mol.GetAtomWithIdx(idx).GetNeighbors():
        idx =atom.GetIdx()
        if idx not in forbidden:
            result.append(idx)
    if len(result)<min_len:
        result.extend(None for _ in range(min_len-len(result)))
    return result

DEBUG=False

def get_pseudo_cip(mol, idx, debug=DEBUG, recalc=True):
    if idx==None:
        result = -1
    else:
        atom = mol.GetAtomWithIdx(idx)
        if atom.HasProp('_CIPRank'):
            result = int(atom.GetProp('_CIPRank'))
        else:
            if recalc:
                Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
                return get_pseudo_cip(mol, idx, debug=debug, recalc=False)
            if debug:
                print('NO CIPRANKS ON', idx)
            result = atom.GetIdx()
    return result

def order_indices(mol, db_idx, assert_stereo=None, debug=DEBUG):
    'convention: [C:1]\C([C:2])=C([C:3])\[C:4]'
    assert assert_stereo in [None, 'STEREOE', 'STEREOZ']
    bond = mol.GetBondBetweenAtoms(*db_idx)
    stereo_flag = bond.GetStereo().name
    if debug:
        print('bond_stereo', stereo_flag, 'bond_idx', db_idx)
    
    idx_a = filter_nei(mol, db_idx[0], forbidden=db_idx, min_len=2)
    idx_b = filter_nei(mol, db_idx[1], forbidden=db_idx, min_len=2)
    if debug:
        print('atom', db_idx[0], 'nei', idx_a)
        print('atom', db_idx[1], 'nei', idx_b)
    assert idx_a.count(None)!=2
    assert idx_b.count(None)!=2
    
    p1, p2 = idx_a
    
    # is in ring?
    rings = [r for r in mol.GetRingInfo().AtomRings() if 
                all(x in r for x in db_idx) and len(r)<11]
    if rings!=[]:
        rings_here = [r for r in rings if p1 in r]
        rings_there = [r for r in rings if idx_b[0] in r]
        
        if rings_here==rings_there:
             p3, p4 = idx_b
        else:
             p4, p3 = idx_b
        if debug:
            print('RINGS')
            print('idx_b', idx_b, 'p3,p4:', p3,p4)
            print('rings here, there:', rings_here, rings_there)
    else:
        if stereo_flag == 'STEREONONE' and assert_stereo!=None:
             stereo_flag = assert_stereo
        p3, p4 = idx_b
        cip1, cip2, cip3, cip4 = [get_pseudo_cip(mol, x) for x in idx_a+idx_b]
        if debug:
            print('CIP Ranks', cip1, cip2, cip3, cip4)
        a1_is_max = cip1 == max([cip1, cip2])
        p3_is_max = cip3 == max([cip3, cip4])
        flip = a1_is_max==p3_is_max
        if stereo_flag=='STEREOZ':
             flip = not flip
        if flip:
             p3, p4 = p4, p3
             
    return [p1,p2,p3,p4], stereo_flag


DB = Chem.MolFromSmarts('C=C')
ALLIL = Chem.MolFromSmarts('[CX3+]-[$(C=C)]')

def contact_iterator(mol, indices):
    pairs = {'G':[[0,1], [2,3]],
             'E':[[0,3], [1,2]],
             'Z':[[0,2], [1,3]]}
    symbols = [x if x==None else mol.GetAtomWithIdx(x).GetSymbol() for x in indices]
    for k in 'GEZ':
        for i,j in pairs[k]:
            si, sj = symbols[i], symbols[j]
            i, j = indices[i], indices[j]

            if None in [i,j]:
                continue
            if si=='H' and sj=='H':
                contact_els = 'HH'
            elif 'H' in [si, sj]:
                contact_els = 'CH'
            else:
                contact_els = 'CC'

            yield k, i, j, contact_els
    
def patch_db(bound_mtx, mol, pattern=DB, assert_stereo=None):
    params = {'CC': {'G':2.5, 'E':3.5, 'Z':2.9},
              'HH': {'G':1.8, 'E':3.0, 'Z':2.4},
              'CH': {'G':2.2, 'E':3.4, 'Z':2.6}}
    delta = 0.1
    dbs = mol.GetSubstructMatches(pattern)
    # print(dbs)
    for db in dbs:
        indices, flag = order_indices(mol, db, assert_stereo)
        for contact_type, i, j, contact_els in contact_iterator(mol, indices):
            i, j = sorted([i,j])
            v = params[contact_els][contact_type]
            bound_mtx[i,j] = v+delta
            bound_mtx[j,i] = v-delta
    return bound_mtx

def dih_iterator(mol, confid=0, pattern=DB, assert_stereo=None):
    dbs = mol.GetSubstructMatches(pattern)
    for db in dbs:
        indices, flag = order_indices(mol, db, assert_stereo)
        for ctype, i, j, _ in contact_iterator(mol, indices):
            if ctype=='G':
                continue
            elif ctype=='E':
                val = 180
            elif ctype=='Z':
                val = 0
            else: #Z
                val = Chem.rdMolTransforms.GetDihedralDeg(mol.GetConformer(confid),
                                                i, db[0], db[1], j)
                if abs(val)<abs(abs(val)-180):
                    val = 0
                else:
                    val = 180


            if mol.GetBondBetweenAtoms(*db).IsInRing():
                continue

            yield i, db[0], db[1], j, val

def get_max_dev(mol, confid=0, pattern=DB, assert_stereo=None):
    devs = []
    for i,j,k,l, v in dih_iterator(mol, confid, pattern, assert_stereo):
        actual = Chem.rdMolTransforms.GetDihedralDeg(mol.GetConformer(confid),
                                                i, j, k, l)
        devs.append((i,j,k,l,abs(actual-v)))
    if devs!=[]:
        devs.sort(key=lambda x:x[-1])
        return devs[-1]


def correct(mol, confid=0, pattern=DB, assert_stereo=None):
    for i,j,k,l,v in dih_iterator(mol, confid, pattern, assert_stereo):
        Chem.rdMolTransforms.SetDihedralDeg(mol.GetConformer(confid),
                                                i, j, k, l, v)

def optimize_hydrogens(mol, confid=0):
    prm = AllChem.MMFFGetMoleculeProperties(mol)
    ff = AllChem.MMFFGetMoleculeForceField(mol, prm, confId=confid)
    for a in mol.GetAtoms():
        if a.GetSymbol!='H':
            ff.MMFFAddPositionConstraint(a.GetIdx(), 0.1, 100)
    ff.Minimize()

def sphere_potential(xyz, r=5.5):
    radii_v = -(xyz - xyz.sum(axis=0))
    radii = np.sqrt((radii_v*radii_v).sum(axis=1))
    radii_v/=radii.reshape(-1,1)
    magnitude = np.exp(np.clip(radii-r, -r, 2*r))
    return magnitude.reshape(-1,1)*radii_v

def project_forces_onto_axis(force_array, positions_array, start_x, v):
    '''v has to be normal!'''
    xx0 = positions_array - start_x
    xxp = xx0 - xx0.dot(v).reshape(-1,1)*v
    
    forces_projected = force_array - force_array.dot(v).reshape(-1,1)*v
    momenta = np.cross(xxp, forces_projected)
    effective_moment = momenta.sum(axis=0).dot(v)
    return effective_moment

def get_bond_data(mol, confid=0, forbidden=[]):
    results = []
    for b in mol.GetBonds():
        order = b.GetBondTypeAsDouble()
        in_ring = b.IsInRing()
        if in_ring or order!=1:
            continue
            
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        if 'H' in [x.GetSymbol() for x in [a1, a2]]:
            continue
        a1i, a2i = a1.GetIdx(), a2.GetIdx()
        if a1i in forbidden and a2i in forbidden:
            continue
        n1 = [x.GetIdx() for x in a1.GetNeighbors() if x.GetIdx() not in [a1i, a2i]] 
        n2 = [x.GetIdx() for x in a2.GetNeighbors() if x.GetIdx() not in [a1i, a2i]]
        
        if [] in [n1,n2]:
            continue
        n1i = n1[0]
        n2i = n2[0]
        
        a1v, a2v =[mol.GetConformer(confid).GetAtomPosition(x) for x in [a1i, a2i]]
        v = a2v-a1v
        v /= np.linalg.norm(v)
        results.append((a1v, v, (n1i, a1i, a2i, n2i)))
    return results

def adjust_dihedral_step(mol, confid=0, potential=sphere_potential, bond_data=None, 
                         step_deg=10, forbidden=[]):
    conformer = mol.GetConformer(confid)
    xyz = conformer.GetPositions()
    forces = potential(xyz)
    
    if bond_data is None:
        bond_data  = get_bond_data(mol, confid, forbidden=forbidden)
    if bond_data ==[]:
        return
    
    momenta = np.array([project_forces_onto_axis(forces, xyz, s, v) for s, v, _ in bond_data])
    angles = step_deg*momenta/abs(momenta).max()
    
    for i, a in enumerate(angles):
        dih_def = bond_data[i][-1]
        dih = Chem.rdMolTransforms.GetDihedralDeg(conformer, *dih_def)
        Chem.rdMolTransforms.SetDihedralDeg(conformer, *dih_def, dih+a)

def gently_adjust_to_sphere(mol, confid=0, radius=5.5, bond_data=None, maxit=12, 
                            forbidden=[], step_deg=10):
    curr_radius = Chem.Get3DDistanceMatrix(mol, confid).max()/2
    potential = lambda x: sphere_potential(x, r=radius)
    i=0
    while curr_radius>radius and i<maxit:
        adjust_dihedral_step(mol, confid, potential=potential, forbidden=forbidden,
                             step_deg=step_deg)
        i += 1
        curr_radius = Chem.Get3DDistanceMatrix(mol, confid).max()/2
