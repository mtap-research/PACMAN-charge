from __future__ import print_function, division
import os
import csv
import json
import functools
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data import random_split

def get_train_val_test_loader(dataset, collate_fn=default_collate,
                              batch_size=64, train_idx=None,val_idx=None, test_idx=None,
                              num_workers=0, pin_memory=False):
    generator = torch.Generator().manual_seed(1234)
    dataset_train, dataset_val, dataset_test = random_split(dataset, [len(train_idx), len(val_idx), len(test_idx)],
                                                            generator=generator)
    train_loader = DataLoader(dataset_train, batch_size=batch_size,
                            num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=pin_memory)
    val_loader = DataLoader(dataset_val, batch_size=batch_size,
                            num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=pin_memory,shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=batch_size,
                          num_workers=num_workers,
                          collate_fn=collate_fn, pin_memory=pin_memory,shuffle=False)
    return train_loader, val_loader, test_loader

def collate_pool(dataset_list):
    batch_atom_fea = []
    batch_nbr_fea = []
    batch_nbr_fea_idx1 = []
    batch_nbr_fea_idx2 = []
    batch_num_nbr = []
    batch_dij = []
    crystal_atom_idx = []
    batch_target = []
    batch_cif_ids = []
    base_idx = 0
    for i, ((atom_fea, nbr_fea, nbr_fea_idx1, nbr_fea_idx2, num_nbr, dij), target, cif_id) in enumerate(dataset_list):
        n_i = atom_fea.shape[0]
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        tt1 = np.array(nbr_fea_idx1)+base_idx
        tt2 = np.array(nbr_fea_idx2)+base_idx
        batch_nbr_fea_idx1.append(torch.LongTensor(tt1.tolist()))
        batch_nbr_fea_idx2.append(torch.LongTensor(tt2.tolist()))
        batch_num_nbr.append(num_nbr)
        crystal_atom_idx.append(torch.LongTensor([i]*n_i))
        batch_target.append(target)
        batch_cif_ids.append(cif_id)
        batch_dij.append(dij)
        base_idx += n_i
    return (torch.cat(batch_atom_fea, dim=0),torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx1, dim=0),torch.cat(batch_nbr_fea_idx2, dim=0),
            torch.cat(batch_num_nbr, dim=0),torch.cat(crystal_atom_idx,dim=0),torch.cat(batch_dij,dim=0)),\
        		torch.stack(batch_target, dim=0),batch_cif_ids

class GaussianDistance(object):
    def __init__(self, dmin, dmax, step, var=None):
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var
    def expand(self, distances):
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 / self.var**2)

class AtomInitializer(object):
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}
    def get_atom_fea(self, atom_type):
        return self._embedding[atom_type]
    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}
    def state_dict(self):
        return self._embedding
    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]

class AtomCustomJSONInitializer(AtomInitializer):
		def __init__(self, elem_embedding_file):
				elem_embedding = json.load(open(elem_embedding_file))
				elem_embedding = {int(key): value for key, value in elem_embedding.items()}
				atom_types = set(elem_embedding.keys())
				super(AtomCustomJSONInitializer, self).__init__(atom_types)
				for key in range(101):
						zz = np.zeros((101,))
						zz[key] = 1.0
						self._embedding[key] = zz.reshape(1,-1)

class CIFData(Dataset):
    def __init__(self,root_dir,radius=6,dmin=0,step=0.2,random_seed=110):
        self.root_dir = root_dir
        self.radius = radius
        # id_prop_file = os.path.join(self.root_dir, 'id_prop_pbe.csv')
        id_prop_file = os.path.join(self.root_dir, 'id_prop_bandgap.csv')
        with open(id_prop_file, encoding='UTF-8-sig') as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]
        np.random.seed(random_seed)
        np.random.shuffle(self.id_prop_data)
        atom_init_file = os.path.join(self.root_dir, 'atom_init.json')
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)
    def __len__(self):
        return len(self.id_prop_data)
    @functools.lru_cache(maxsize=None) 
    def __getitem__(self,idx):
        cif_id, target = self.id_prop_data[idx] 
        with open(os.path.join(self.root_dir,cif_id+'.json')) as f:
            crystal_data = json.load(f)
        nums = crystal_data['numbers']
        atom_fea = np.vstack([self.ari.get_atom_fea(nn) for nn in nums])
        index1 = np.array(crystal_data['index1'])
        nbr_fea_idx = np.array(crystal_data['index2'])
        dij = np.array(crystal_data['dij']) 
        nbr_fea = self.gdf.expand(dij)
        num_nbr = np.array(crystal_data['nn_num'])
        atom_fea = torch.Tensor(atom_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx1 = torch.LongTensor(index1)
        nbr_fea_idx2 = torch.LongTensor(nbr_fea_idx)
        num_nbr = torch.Tensor(num_nbr)
        dij = torch.Tensor(dij)
        target = torch.Tensor([float(target)])
        return (atom_fea, nbr_fea, nbr_fea_idx1, nbr_fea_idx2, num_nbr, dij), target, cif_id
    