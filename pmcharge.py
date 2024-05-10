import os
import glob
import json
import torch
import pickle
import argparse
from tqdm import tqdm
from model4pre.GCN_E import GCN
from model4pre.GCN_charge import SemiFullGN
from model4pre.data import collate_pool, get_data_loader, CIFData
from model4pre.cif2data import ase_format, CIF2json, pre4pre, write4cif   #,n_atom

def main():
    parser = argparse.ArgumentParser(description="Run PACMaN with the specified configurations")
    parser.add_argument('folder_name', type=str, help='Relative path to a folder with cif files without partial atomic charges')
    parser.add_argument('--charge_type', type=str, default='DDEC6', choices=['DDEC6', 'Bader', 'CM5'], help='Type of charges to use')
    parser.add_argument('--digits', type=int, default=6, help='Number of decimal places to print for partial atomic charges')
    parser.add_argument('--atom_type', action='store_true', default=False, help='Keep the same partial atomic charge for the same atom types')
    parser.add_argument('--neutral', action='store_true', default=False, help='Keep the net charge is zero')
    args = parser.parse_args()

    path = args.folder_name
    charge_type =  args.charge_type
    digits = args.digits
    atom_type = args.atom_type
    neutral = args.neutral
    if os.path.isfile(path):
        print("please input a folder, not a file")
    elif os.path.isdir(path):
        pass
    else:
        print("Can not find your file, please check is it exit or correct?")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # model_pbe_name = "./pth/best_pbe/pbe-atom.pth"
    # model_bandgap_name = "./pth/best_bandgap/bandgap.pth"
    
    if charge_type=="DDEC6":
        model_charge_name = "./pth/best_ddec/ddec.pth"
        # pbe_nor_name = "./pth/best_pbe/normalizer-pbe.pkl"
        # bandgap_nor_name = "./pth/best_bandgap/normalizer-bandgap.pkl"
        charge_nor_name = "./pth/best_ddec/normalizer-ddec.pkl"
    elif charge_type=="Bader":
        model_charge_name = "./pth/best_bader/bader.pth"
        charge_nor_name = "./pth/best_bader/normalizer-bader.pkl"
    elif charge_type=="CM5":
        model_charge_name = "./pth/best_cm5/cm5.pth"
        charge_nor_name = "./pth/best_cm5/normalizer-cm5.pkl"
    # with open(pbe_nor_name, 'rb') as f:
    #     pbe_nor = pickle.load(f)
    # f.close()
    # with open(bandgap_nor_name, 'rb') as f:
    #     bandgap_nor = pickle.load(f)
    # f.close()
    with open(charge_nor_name, 'rb') as f:
        charge_nor = pickle.load(f)
    f.close()


    print("Folder Name: " + str(path))
    print("Charge Type: "+ str(charge_type))
    print("Digits: " + str(digits))
    print("Atom Type:" + str(atom_type))
    print("Neutral: " + str(neutral))

    cif_files = glob.glob(os.path.join(path, '*.cif'))
    print("writing cif: ***_pacman.cif")

    # dic = {}
    fail = {}
    i = 0
    for cif in tqdm(cif_files):
        try:
            ase_format(cif)
            cif_data = CIF2json(cif)
            pos = pre4pre(cif)
            # num_atom = n_atom(cif)
            batch_size = 1
            num_workers = 0
            pin_memory = False
            pre_dataset = CIFData(cif_data, pos,6,0,0.2)
            collate_fn = collate_pool
            pre_loader= get_data_loader(pre_dataset,collate_fn,batch_size,num_workers,pin_memory)
            for batch in pre_loader:
                chg_1 = batch[0].shape[-1]+3
                chg_2 = batch[1].shape[-1]

            # pbe1 = structures[0].shape[-1]
            # pbe2 = structures[1].shape[-1]
            # checkpoint = torch.load(model_pbe_name, map_location=torch.device(device))
            # x = checkpoint['model_args']
            # atom_fea_len = x['atom_fea_len']
            # h_fea_len = x['h_fea_len']
            # n_conv = x['n_conv']
            # n_h = x['n_h']
            # model_pbe = GCN(pbe1,pbe2,atom_fea_len,n_conv,h_fea_len,n_h)
            # model_pbe.cuda() if torch.cuda.is_available() else model_pbe.to(device)
            # model_pbe.load_state_dict(checkpoint['state_dict'])
            # model_pbe.eval()
            # bandgap1 = structures[0].shape[-1]
            # bandgap2 = structures[1].shape[-1]
            # checkpoint = torch.load(model_bandgap_name, map_location=torch.device(device))
            # x = checkpoint['model_args']
            # atom_fea_len = x['atom_fea_len']
            # h_fea_len = x['h_fea_len']
            # n_conv = x['n_conv']
            # n_h = x['n_h']
            # model_bandgap = GCN(bandgap1,bandgap2,atom_fea_len,n_conv,h_fea_len,n_h)
            # model_bandgap.cuda() if torch.cuda.is_available() else model_bandgap.to(device)
            # model_bandgap.load_state_dict(checkpoint['state_dict'])
            # model_bandgap.eval()
        
            gcn = GCN(chg_1-3, chg_2, 128, 7, 256,5) 
            chkpt = torch.load(model_charge_name, map_location=torch.device(device))
            model4chg = SemiFullGN(chg_1,chg_2,128,8,256)
            model4chg.to(device)
            model4chg.load_state_dict(chkpt['state_dict'])
            model4chg.eval()
            for _, (input) in enumerate(pre_loader):
                with torch.no_grad():
                    input_var = (input[0].to(device),
                                    input[1].to(device),
                                    input[2].to(device),
                                    input[3].to(device),
                                    input[4].to(device),
                                    input[5].to(device))
                    encoder_feature = gcn.Encoding(*input_var)
                    atoms_fea = torch.cat((input[0],input[7]),dim=-1)
                    input_var2 = (atoms_fea.to(device),
                                    input[1].to(device),
                                    input[2].to(device),
                                    input[3].to(device),
                                    input[4].to(device),
                                    input[5].to(device),
                                    encoder_feature.to(device))
                        
                    # pbe = model_pbe(*input_var)
                    # pbe = pbe_nor.denorm(pbe.data.cpu()).item()*num_atom
                    # bandgap = model_bandgap(*input_var)
                    # bandgap = bandgap_nor.denorm(bandgap.data.cpu()).item()
                    # print("PBE energy and Bandgap of "+ cif_ids[0] + ": " + str(pbe) + " and " + str(bandgap) + " / ev")
                    # dic[cif] = [pbe,bandgap]

                    chg = model4chg(*input_var2)
                    chg = charge_nor.denorm(chg.data.cpu())
                    write4cif(cif,chg,digits,atom_type,neutral,charge_type)          
        except:
            print("Fail predict: " + cif)
            fail[str(i)]=[cif]
            i += 1
        # with open(path + "/preE.json",'w') as f:
        #     json.dump(dic,f)
        # f.close()
        if i==0:
            pass
        else:
            with open(path + "fail.json",'w') as f:
                json.dump(fail,f)
            f.close()
    print("Fail list: ", fail)
if __name__ == "__main__":
    main()
