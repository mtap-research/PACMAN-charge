import os
import time
import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from model.GCN_E import GCN
from torch.optim.lr_scheduler import ExponentialLR
from model.utils import Normalizer,sampling,save_checkpoint,AverageMeter,mae
from model.data_E import collate_pool, get_train_val_test_loader, CIFData,GaussianDistance

def main():
    model_folder = './pth/'
    os.makedirs(model_folder, exist_ok=True)
    chk_name = model_folder+'chk_bandgap/checkpoint.pth'
    best_name = model_folder+'best_bandgap/bandgap.pth'
    root_dir = './data/json/'
    radius = 6.0
    dmin = 0
    step = 0.2
    random_seed = 1123
    batch_size = 16
    N_tot = 16781
    N_tr = int(N_tot*0.8)
    N_val = int(N_tot*0.1)
    N_test = N_tot - N_tr - N_val
    train_idx = list(range(N_tr))
    val_idx = list(range(N_tr,N_tr+N_val))
    test_idx = list(range(N_tr+N_val,N_tot))
    num_workers = 0
    pin_memory = False
    atom_fea_len = 128
    h_fea_len = 256
    n_conv = 7
    n_h = 5
    lr_decay_rate = 0.99
    lr = 0.001
    weight_decay = 1e-6
    noise = 1e-5
    gdf = GaussianDistance(dmin=0.0, dmax=6.0, step=0.2)
    model_args = {'radius':radius,'dmin':dmin,'step':step,'batch_size':batch_size,
							  'random_seed':random_seed,'N_tr':N_tr,'N_val':N_val,'N_test':N_test,
								'atom_fea_len':atom_fea_len,'h_fea_len':h_fea_len,
								'n_conv':n_conv,'n_h':n_h,'lr':lr,'lr_decay_rate':lr_decay_rate,'weight_decay':weight_decay}
    best_mae_error = 1e10
    epochs = 500
    dataset = CIFData(root_dir,radius,dmin,step,is_unrelaxed=False,random_seed=random_seed)
    collate_fn = collate_pool
    train_loader, val_loader, test_loader = get_train_val_test_loader(dataset,collate_fn,batch_size,
                                                          train_idx,val_idx,test_idx,num_workers,pin_memory)
    sample_target = sampling(root_dir+'id_prop_bandgap.csv')
    normalizer = Normalizer(sample_target)
    with open(model_folder + 'best_bandgap/normalizer-bandgap.pkl', 'wb') as f:
        pickle.dump(normalizer, f)
    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    model = GCN(orig_atom_fea_len,nbr_fea_len,atom_fea_len,n_conv,h_fea_len,n_h)
    model.cuda()
    model_args['orig_atom_fea_len'] = orig_atom_fea_len
    model_args['nbr_fea_len'] = nbr_fea_len
    model_args['noise'] = noise
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr,weight_decay=weight_decay)
    scheduler = ExponentialLR(optimizer, gamma=lr_decay_rate)
    t0 = time.time()
    for epoch in range(epochs):
        train(train_loader,model,criterion,optimizer,epoch,normalizer,gdf,noise)
        mae_error = validate(val_loader,model,criterion,normalizer)
        scheduler.step()
        is_best = mae_error < best_mae_error
        best_mae_error = min(mae_error, best_mae_error)
        save_checkpoint({'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'best_mae_error': best_mae_error,
                         'optimizer': optimizer.state_dict(),
                         'normalizer': normalizer.state_dict(),
                         'model_args':model_args},is_best,chk_name,best_name)
    t1 = time.time()
    print('--------Training time in sec-------------')
    print(t1-t0)
    print('---------Best Model on Validation Set---------------')
    best_checkpoint = torch.load(best_name)
    print(best_checkpoint['best_mae_error'].cpu().numpy())
    print('---------Evaluate Model on Test Set---------------')
    model.load_state_dict(best_checkpoint['state_dict'])
    validate(test_loader, model, criterion, normalizer)

def train(train_loader, model, criterion, optimizer, epoch, normalizer,gdf,e):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()
    model.train()
    end = time.time()
    for i, (input,target,_) in enumerate(train_loader):
        data_time.update(time.time() - end)
        input6 = input[6]
        noise = torch.Tensor(float(e)*np.random.normal(size=input6.shape))
        input6 += noise
        input6 = np.array(input6)
        input1_noise = torch.Tensor(gdf.expand(input6))
        input_var = (input[0].cuda(),
                    input[1].cuda(),
                    input[2].cuda(),
                    input[3].cuda(),
                    input[4].cuda(),
                    input[5].cuda())
        input_var_noise = (input[0].cuda(),
                            input1_noise.cuda(),
                            input[2].cuda(),
                            input[3].cuda(),
                            input[4].cuda(),
                            input[5].cuda())
        target_normed = normalizer.norm(target)
        target_var = target_normed.cuda()
        output = model(*input_var)
        output_noise = model(*input_var_noise)
        loss = criterion(output, target_var) + criterion(output_noise, target_var)
        mae_error = mae(normalizer.denorm(output.data.cpu()), target)
        losses.update(loss.item(), target.size(0))
        mae_errors.update(mae_error, target.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        print('Epoch: [{0}][{1}/{2}]\t'
          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
          'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
          epoch, i, len(train_loader), batch_time=batch_time,
          data_time=data_time, loss=losses, mae_errors=mae_errors))

def validate(test_loader,model,criterion,normalizer):
    batch_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()
    model.eval()
    end = time.time()
    # cifsids_all=[]
    for i, (input, target, _) in enumerate(test_loader):
        with torch.no_grad():
            input_var = (input[0].cuda(),
                        input[1].cuda(),
                        input[2].cuda(),
                        input[3].cuda(),
                        input[4].cuda(),
                        input[5].cuda())
            target_normed = normalizer.norm(target)
            target_var = target_normed.cuda()
            output = model(*input_var)
            loss = criterion(output, target_var)
            mae_error = mae(normalizer.denorm(output.data.cpu()), target)
            losses.update(loss.item(), target.size(0))
            mae_errors.update(mae_error, target.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
        print('Test: [{0}/{1}]\t'
          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
          'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
          i, len(test_loader), batch_time=batch_time, loss=losses,
          mae_errors=mae_errors))
        star_label = '*'
    print(' {star} MAE {mae_errors.avg:.3f}'.format(star=star_label,mae_errors=mae_errors))
    # for cif in cifids:
        # cifsids_all.append(cif)
        # print(cif)
    # np.savetxt("./predicted_data/pbe/pbe_test_name.txt",cifsids_all, fmt='%s')

    return mae_errors.avg

if __name__ == '__main__':
	main()
