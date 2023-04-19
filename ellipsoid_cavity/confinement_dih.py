import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.optimize import minimize
import networkx as nx
from .confinement import opt_ff_with_sphere
from .embed_patches import project_forces_onto_axis, raw_embed, sphere, optimize_hydrogens, smooth, patch_db, correct, ALLIL
from .distance_to_ellipse import DistToEllipsoid
from .ellipsoid import EllipsoidTool
import tqdm
from scipy.spatial.transform import Rotation

def get_is_inside(x, radii):
    return (x*x/(radii*radii)).sum() < 1

def dist_vector_ellipsoid(xyz, radii):
    #print(xyz)
    sgn_matrix = np.sign(xyz)
    xyz = abs(xyz)
    result = np.zeros(xyz.shape)
    for i,x in enumerate(xyz):
        p = DistToEllipsoid(radii, x)
        if not get_is_inside(x, radii):
            result[i] = p-x
    return result*sgn_matrix

def get_selection(mol, idx1, idx2, G=None):
    if G is None:
        G = nx.from_numpy_array(Chem.GetAdjacencyMatrix(mol))
    G.remove_edge(idx1, idx2)
    results = [nx.node_connected_component(G, x) for x in [idx1, idx2]]
    G.add_edge(idx1, idx2)
    return results

def make_projection_vectors(mol, dih_def_list):
    result = []
    result_indices = []
    G = nx.from_numpy_array(Chem.GetAdjacencyMatrix(mol))
    for _, i, j, _ in dih_def_list:
        a, b = get_selection(mol, i, j, G=G)
        b = list(b)
        n1, n2 = len(a), len(b)
        N = n1 + n2
        resp = np.zeros((N, N))
        c = n2/N
        indices = []
        for idx in range(N):
            if idx in b:
                resp[idx, b] = c
                indices.append(idx)
        result.append(resp)
        result_indices.append(indices)
    return result, result_indices

def get_dihedral_data(mol, excludeHs=False):
    dih_list = []
    for b in mol.GetBonds():
        if (not b.IsInRing()) and b.GetBondTypeAsDouble()==1:
            j,k = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            j_nei = [x.GetIdx() for x in mol.GetAtomWithIdx(j).GetNeighbors()]
            k_nei = [x.GetIdx() for x in mol.GetAtomWithIdx(k).GetNeighbors()]
            j_nei = [x for x in j_nei if x!=k]
            k_nei = [x for x in k_nei if x!=j]
            if j_nei==[] or k_nei==[]:
                continue
            i = j_nei[0]
            l = k_nei[0]
            if excludeHs and mol.GetAtomWithIdx(l).GetSymbol()=='H':
                continue
            dih_list.append((i,j,k,l))
    in_arr = np.zeros((len(dih_list), mol.GetNumAtoms()))
    out_arr = np.zeros((len(dih_list), mol.GetNumAtoms()))
    for i, (_, j, k, _) in enumerate(dih_list):
        in_arr[i,j] = 1
        out_arr[i,k] = 1
    return dih_list, in_arr, out_arr


def set_dihedrals(conformer, dih_def, dih_v):
    for i, dih in enumerate(dih_def):
        if np.isnan(dih_v[i]):
            continue
        Chem.rdMolTransforms.SetDihedralDeg(conformer, *dih, dih_v[i])  

def get_dihedrals(conformer, dih_def):
    return [ Chem.rdMolTransforms.GetDihedralDeg(conformer, *dih) for dih in dih_def]

def get_ff(mol, confid=0):
    prp = AllChem.MMFFGetMoleculeProperties(mol)
    ff = AllChem.MMFFGetMoleculeForceField(mol, prp, confId=confid)
    return ff

def get_E(mol, confid=0):
    return get_ff(mol, confid).CalcEnergy()


def get_F(mol, confid=0):
    return get_ff(mol, confid).CalcGrad()

def harmonic_potential(xyz, r, force=False, k=1000, exp=True):
    if hasattr(r, '__iter__'):
        Xn = dist_vector_ellipsoid(xyz, r)
        R = np.linalg.norm(Xn, axis=1)
        Xn =Xn/np.where(R==0,0, 1/R).reshape(-1,1)
        R = np.clip(R, 0, 2*r[-1])
    else:
        X = xyz - xyz.mean(axis=0)
        R = np.linalg.norm(X, axis=1)
    #     print(X)
    #     print(R)
        Xn = -X/R.reshape(-1,1)
        lower = -r if exp else 0
        R = np.clip(R-r, lower, 2*r)
    if force:
        result = Xn*R.reshape(-1,1)*k 
        if exp:
            result = np.exp(R).reshape(-1,1)*Xn
    else:
        result = k*(R*R).sum()/2
        if exp:
            result = np.exp(R).sum()
    return result
    
def get_confinement_potential_and_force(mol, confid, r=5.5, which='both', remove_sum=False, exp=False, removeHs=False):
    dih_def_list, dih_in, dih_out = get_dihedral_data(mol, excludeHs=removeHs)
    conformer = mol.GetConformer(confid)
    d0 = get_dihedrals(conformer, dih_def_list)
    #print(d0)
    scale = np.pi/180
    projection, non_null_space = make_projection_vectors(mol, dih_def_list)
    
    def pot(dih_angles):
        set_dihedrals(conformer, dih_def_list, dih_angles)
        E = get_E(mol, confid)
        Ec = harmonic_potential(conformer.GetPositions(), r, force=False, exp=exp)
        return E + Ec
    
    def force(dih_angles):
        set_dihedrals(conformer, dih_def_list, dih_angles)
        xyz = conformer.GetPositions()
        F = np.array(list(get_F(mol, confid))).reshape(-1, 3)
        Fc = harmonic_potential(xyz, r, force=True, exp=exp)
#         print(F)
#         print(Fc)
        
        if which=='both':
            F = F + Fc
        elif which=='confinement':
            F = Fc
        elif which=='ff':
            F=F
        else:
            raise ValueError(f'Unknown: {which}')
            
        if remove_sum:
            F -= F.sum(axis=0)
        start_points = dih_in.dot(xyz)
        vecs = dih_out.dot(xyz) - start_points
        vecs /= np.linalg.norm(vecs, axis=1).reshape(-1,1)
        torques = []
        for s, v, p, indices in zip(start_points, vecs, projection, non_null_space):
            raw_torque = project_forces_onto_axis(F, xyz, s, v)
            if indices!=[]:
                Fcom = F[indices]
                XYZcom = p.dot(xyz)[indices]
                torque_com = project_forces_onto_axis(Fcom, XYZcom, s, v)
                raw_torque -= torque_com
                
            torques.append(raw_torque)
            
        torques = np.array(torques)*scale
        
        return torques
        
    return pot, force, d0

rotY = Rotation.from_euler('y', np.pi/2).as_matrix()

def standardize(mol, confid, use_rot_y=False):
    conf = mol.GetConformer(confid)
    pos = conf.GetPositions()
    center, radii, orientation = EllipsoidTool().getMinVolEllipse(pos)
    pos -= center
    pos = pos.dot(orientation.T)
    if use_rot_y:
        pos = pos.dot(rotY)
    for i, x in enumerate(pos):
        conf.SetAtomPosition(i, list(x))

def embed_dih(smiles, n_conformers=10, r=6, method='cobyla', randomize=True, use_scipy=True, exp=False, random_step=60, use_tqdm=False):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if hasattr(r, '__iter__'):
        opt_ff_with_sphere('', mol=mol, radius=max(r), num_confs=n_conformers)
        cids = [x.GetId() for x in mol.GetConformers()]
    else:
        patches = [lambda x,y: sphere(x,y,r), smooth, patch_db]
        cids = raw_embed(mol, n_conformers, modifiers=patches)
        cids = list(cids)
    results = []
    if use_tqdm:
        iterator = tqdm.tqdm(list(cids))
    else:
        iterator = cids
    for confid in iterator:
        #optimize_hydrogens(mol, confid)
        correct(mol, confid, pattern=ALLIL)
        if hasattr(r, '__iter__'):
            standardize(mol, confid, use_rot_y=True)
            if confid==cids[0]:
                tc,ta,tr=EllipsoidTool().getMinVolEllipse(mol.GetConformer(confid).GetPositions())
                print(f'test: center={tc.round(3)} axes={ta.round(3)}  orienation={tr.round(3)}')
        else:
            optimize_hydrogens(mol, confid)


        if use_scipy:
            p, f, d0 = get_confinement_potential_and_force(mol, confid, r=r, exp=exp, removeHs=True)
            if randomize:
                d0 = np.array(d0)
                t0 = f(d0)
                t0 = t0/abs(t0.sum())
                d0 += np.random.random(len(d0))*random_step*t0
            result = minimize(p, d0, jac=f, method=method)#, options=dict(maxiter=5000))
        else:
            result = None
        results.append(result)
    return [Chem.MolToMolBlock(mol, confId=c) for c in cids], results
