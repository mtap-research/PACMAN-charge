import os
import re
import json
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import pymatgen.core as mg
from ase.io import read
from ase import neighborlist
from pymatgen.analysis import local_env
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.graphs import StructureGraph


# check atom dict, H2O molec, closed dista
def check_all(root_cif_dir, cutoff = 0.75):
    metals = ['Li','Na','K','Rb','Cs','Fr',
	        'Be','Mg','Ca','Sr','Ba','Ra',
	        'Sc','Y','La','Ac','Ti','Zr',
			'Hf','Mn','Fe','Co','Ni','Cu',
			'Ag','Zn','Cd','Al','Ga','In',
			'Tl']
    bad_list_distance = []
    bad_list_oxo = []
    bad_list_lone_atom = []
    for cif in os.listdir(root_cif_dir):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            mof = read(os.path.join(root_cif_dir, cif))
        d = mof.get_all_distances()
        upper_diag = d[np.triu_indices_from(d, k=1)]
        for entry in upper_diag:
            if entry < cutoff:
                print('Interatomic distance issue:' + cif.split('.')[0])
                bad_list_distance.append(cif)
                break
        bad = False
        syms = np.array(mof.get_chemical_symbols())
        if not any(item in syms for item in metals):
            continue
        cutoff = neighborlist.natural_cutoffs(mof)
        nl = neighborlist.NeighborList(cutoff,self_interaction=False,bothways=True)
        nl.update(mof)
        for i, sym in enumerate(syms):
            if sym not in metals:
                continue
            bonded_atom_indices = nl.get_neighbors(i)[0]
            if bonded_atom_indices is None:
                continue
            bonded_atom_symbols = syms[bonded_atom_indices]
            for j, bonded_atom_symbol in enumerate(bonded_atom_symbols):
                if bonded_atom_symbol != 'O':
                    continue
                cn = len(nl.get_neighbors(bonded_atom_indices[j])[0])
                if cn == 1:
                    bad = True
                    print('Missing H on terminal oxo: ' + cif)
                    bad_list_oxo.append(cif)
                if bad:
                    break
                if bad:
                    break
        nn = local_env.CrystalNN()
        graph = StructureGraph.with_local_env_strategy(mof, nn)
        for j in range(len(mof)):
            nbr = graph.get_connected_sites(j)
            if not nbr:
                print('Lone atom issue:' + cif+'\n')
                bad_list_lone_atom.append(cif)
                break
    with open('bad_distance.txt','w') as w:
        for bad_cif in bad_list_distance:
            w.write(bad_cif+'\n')
    with open('bad_oxo.txt','w') as w:
        for bad_cif in bad_list_oxo:
            w.write(bad_cif + '\n')
    with open('bad_lone_atom.txt','w') as w:
	    for bad_cif in bad_list_lone_atom:
		    w.write(bad_cif+'\n')

# define periodic table
periodic_table_symbols = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg',
    'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr',
    'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf',
    'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po',
    'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm',
    'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs',
    'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
    ]

#  from cif to json: atom features and bond features, it can save time because do not need to calculate graph infomation every time
def CIF2json(root_cif_dir,data_csv,save_path):
        mofs = pd.read_csv(data_csv)["name"]
        for mof in tqdm(mofs):
            structure = read(root_cif_dir + mof + ".cif")
            struct = AseAtomsAdaptor.get_structure(structure)
            _c_index, _n_index, _, n_distance = struct.get_neighbor_list(r=6, numerical_tol=0, exclude_self=True)
            _nonmax_idx = []
            for i in range(len(structure)):
                idx_i = (_c_index == i).nonzero()[0]
                idx_sorted = np.argsort(n_distance[idx_i])[: 200] # 20
                _nonmax_idx.append(idx_i[idx_sorted])
            _nonmax_idx = np.concatenate(_nonmax_idx)
            index1 = _c_index[_nonmax_idx]
            index2 = _n_index[_nonmax_idx]
            dij = n_distance[_nonmax_idx]
            numbers = []
            s_data = mg.Structure.from_file(root_cif_dir + mof + '.cif')
            elements = [str(site.specie) for site in s_data.sites]
            for i in range(len(elements)):
                ele = elements[i]
                atom_index = periodic_table_symbols.index(ele)
                numbers.append(int(int(atom_index)+1))
            # numbers = np.load(atom_dir + mof + ".npy")
            nn_num = []
            for i in range(len(structure)):
                j = 0
                for idx in range(len(index1)):
                    if index1[idx] == i:
                         j += 1
                    else:
                         pass
                nn_num.append(j)
            data = {"rcut": 6.0,
                    "numbers": numbers,
                    "index1": index1.tolist(),
                    "index2":index2.tolist(),
                    "dij": dij.tolist(),
                    "nn_num": nn_num}
            with open(save_path + mof + "_" + ".json", 'w') as file:
                json.dump(data, file)

# main paart, get cell and position of each atom data and save
def pre4opt(csv, root_cif_dir, save_pos_dir):
    mofs = pd.read_csv(csv)["name"]
    print("Processing the following MOFs:")
    print(mofs)
    for row in mofs:
        mof = row
        try:
            structure = mg.Structure.from_file(root_cif_dir + mof + '.cif')
            coords = structure.frac_coords
            #coords = structure.cart_coords
            #density = structure.density
            #charge = structure.charge
            elements = [str(site.specie) for site in structure.sites]
            pos = []
            #atom = []
            # lattice = structure.lattice.matrix
            #lattice = structure.lattice
            # np.save(save_cell_dir + mof + '.npy', lattice)
            # np.save(save_cell_dir + mof + '.npy', [lattice.abc,lattice.angles])
            for i in range(len(elements)):
                #ele = elements[i]
                #atom_index = periodic_table_symbols.index(ele)
                #atom.append(int(int(atom_index)+1))
                x = coords[i][0]
                y = coords[i][1]
                z = coords[i][2]
                pos.append([float(x),float(y),float(z)])
            np.save(save_pos_dir + mof + '.npy', pos)
            #np.save(save_pos_dir + mof + '.npy', atom)
            print(f"Processed {mof} successfully.")
        except Exception as e:
            print(f"An error occurred while processing {mof}: {e}")

# get numbers of atoms for each structure
def n_atom(data_csv,root_cif_dir):
    mofs = pd.read_csv(data_csv)["name"]
    atom_number = []
    for mof in tqdm(mofs):
        structure = mg.Structure.from_file(root_cif_dir + mof + '.cif')
        elements = [str(site.specie) for site in structure.sites]
        print(mof,len(elements))
        atom_number.append([mof,len(elements)])
    df_atom_number = pd.DataFrame(atom_number)
    df_atom_number.to_csv(root_cif_dir + "atom_number.csv")

# in order to get DDEC charge from relaxed structure
def get_ddec_data(root_cif_dir,dataset_csv,save_ddec_dir):
    mofs = pd.read_csv(dataset_csv)["name"]
    for mof in mofs:
        ddec_data = []
        length = []
        with open(root_cif_dir + mof + ".cif", 'r') as f:
            lines = f.readlines()
            idex_x = []
            idex_ddec = []
            for i, line in enumerate(lines):
                if " _atom_site_pbe_ddec_charge" in line:
                    idex_ddec.append(i + 1)
                elif "_atom_site_fract_x" in line:
                    idex_x.append(i + 1)
            diff_x_ddec = int(idex_ddec[0]) - int(idex_x[0])
            for line in lines:
                length.append(len(line.split()))
            for line in lines:
                if len(line.split()) == max(length):
                    split_line = re.split(r"[ ]+", line)
                    ddec = float(split_line[diff_x_ddec + 3])
                    ddec_data.append(ddec)
            np.save(save_ddec_dir + mof + '.npy', ddec_data)          
        f.close()

def get_bader_data(root_cif_dir,dataset_csv,save_bader_dir):
    mofs = pd.read_csv(dataset_csv)["name"]
    for mof in mofs:
        try:
            bader_data = []
            length = []
            with open(root_cif_dir + mof + ".cif", 'r') as f:
                lines = f.readlines()
                idex_x = []
                idex_bader = []
                for i, line in enumerate(lines):
                    if " _atom_site_pbe_bader_charge" in line:
                        idex_bader.append(i + 1)
                    elif "_atom_site_fract_x" in line:
                        idex_x.append(i + 1)
                diff_x_bader = int(idex_bader[0]) - int(idex_x[0])
                for line in lines:
                    length.append(len(line.split()))
                for line in lines:
                    if len(line.split()) == max(length):
                        split_line = re.split(r"[ ]+", line)
                        bader = float(split_line[diff_x_bader + 3])
                        bader_data.append(bader)
                np.save(save_bader_dir + mof + '.npy', bader_data)          
            f.close()
        except:
            pass
        
def get_cm5_data(root_cif_dir,dataset_csv,save_cm5_dir):
    mofs = pd.read_csv(dataset_csv)["name"]
    for mof in mofs:
        try:
            cm5_data = []
            length = []
            with open(root_cif_dir + mof + ".cif", 'r') as f:
                lines = f.readlines()
                idex_x = []
                idex_cm5 = []
                for i, line in enumerate(lines):
                    if " _atom_site_pbe_cm5_charge" in line:
                        idex_cm5.append(i + 1)
                    elif "_atom_site_fract_x" in line:
                        idex_x.append(i + 1)
                diff_x_cm5 = int(idex_cm5[0]) - int(idex_x[0])
                for line in lines:
                    length.append(len(line.split()))
                for line in lines:
                    if len(line.split()) == max(length):
                        split_line = re.split(r"[ ]+", line)
                        cm5 = float(split_line[diff_x_cm5 + 3])
                        cm5_data.append(cm5)
                np.save(save_cm5_dir + mof + '.npy', cm5_data)          
            f.close()
        except:
            pass

def get_repeat_data(root_cif_dir,save_repeat_dir):
    mofs = glob.glob(os.path.join(root_cif_dir, '*.cif'))
    for mof in tqdm(mofs[:]):
        mof = mof.replace(".cif","").split("/")[-1]
        try:
            repeat_data = []
            with open(root_cif_dir + mof + ".cif", 'r') as f:
                lines = f.readlines()
                
                for line in lines:
                    if len(re.split(r"[ ]+", line))==7:
                    
                        repeat=re.split(r"[ ]+", line)[-1]
                        repeat_data.append(repeat.replace("\n",""))
                np.save(save_repeat_dir + mof + '.npy', repeat_data)          
            f.close()
			
        except:
           pass