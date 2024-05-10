import time
import torch
import pickle
import torch.nn as nn
from random import sample
from model.GCN_E import GCN
import torch.optim as optim
from model.GCN_charge import SemiFullGN
from torch.optim.lr_scheduler import ExponentialLR
from model.data_charge import collate_pool, get_data_loader, CIFData
from model.utils import Normalizer,save_checkpoint,AverageMeter,mae

def main():
    model_folder = './pth/'
    chk_name = model_folder+'chk_cm5/checkpoint.pth'
    best_name = model_folder+'best_cm5/cm5.pth'
    root_dir ='./data/json/'
    root_dir_pos ='./data/npy/pos/'
    root_dir_cm5 ='./data/npy/cm5/'
    radius = 6
    dmin = 0
    step = 0.2
    random_seed = 1126
    batch_size = 32
    num_workers = 0
    pin_memory = False
    atom_fea_len = 128
    n_feature = 256
    n_conv = 8
    lr_decay_rate = 0.99
    lr = 0.0005
    weight_decay = 0
    best_mae_error = 1e10
    start_epoch = 0
    epochs = 500
    
    train_csv = root_dir+'id_prop_train_cm5.csv'
    val_csv = root_dir+'id_prop_val_cm5.csv'
    test_csv = root_dir+'id_prop_test_cm5.csv'

    train_dataset = CIFData(root_dir,root_dir_pos,root_dir_cm5,train_csv,radius,dmin,step,random_seed)
    val_dataset = CIFData(root_dir,root_dir_pos,root_dir_cm5,val_csv,radius,dmin,step,random_seed)
    test_dataset = CIFData(root_dir,root_dir_pos,root_dir_cm5,test_csv,radius,dmin,step,random_seed)
    collate_fn = collate_pool
    train_loader = get_data_loader(train_dataset,collate_fn,batch_size,num_workers,pin_memory,False)
    val_loader = get_data_loader(val_dataset,collate_fn,batch_size,num_workers,pin_memory,True)
    test_loader= get_data_loader(test_dataset,collate_fn,batch_size,num_workers,pin_memory,True)
    print('# of trainset: ',len(train_loader.dataset))
    print('# of valset: ',len(val_loader.dataset))
    print('# of testset: ',len(test_loader.dataset))
    sample_data_list = [train_dataset[i] for i in sample(range(len(train_dataset)), 500)]
    _,sample_target_charge, _ = collate_pool(sample_data_list)
    normalizer = Normalizer(sample_target_charge)
    with open(model_folder + 'best_cm5/normalizer-cm5.pkl', 'wb') as f:
        pickle.dump(normalizer, f)
    structures, _,_,_ = train_dataset[0]
    orig_atom_fea_len = structures[0].shape[-1] + 3
    nbr_fea_len = structures[1].shape[-1]
   
    gcn = GCN(orig_atom_fea_len-3, nbr_fea_len, 128, 7, 256,5) 
    gcn.cuda()
    model = SemiFullGN(orig_atom_fea_len,nbr_fea_len,atom_fea_len,n_conv,n_feature)
    model.cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr,weight_decay=weight_decay)
    scheduler = ExponentialLR(optimizer, gamma=lr_decay_rate)
    t0 = time.time()
    for epoch in range(start_epoch,epochs):
        train(train_loader,model,gcn,criterion,optimizer,epoch,normalizer)
        mae_error = validate(val_loader,model,gcn,criterion,normalizer)
        scheduler.step()
        is_best = mae_error < best_mae_error
        best_mae_error = min(mae_error, best_mae_error)
        save_checkpoint({'epoch': epoch,'state_dict': model.state_dict(),'best_mae_error': best_mae_error,
                         'optimizer': optimizer.state_dict(),'normalizer': normalizer.state_dict()},
                         is_best,chk_name,best_name)
    t1 = time.time()
    print('--------Training time in sec-------------')
    print(t1-t0)
    print('---------Best Model on Validation Set---------------')
    best_checkpoint = torch.load(best_name)
    print(best_checkpoint['best_mae_error'].cpu().numpy())
    print('---------Evaluate Model on Test Set---------------')
    model.load_state_dict(best_checkpoint['state_dict'])
    validate(test_loader,model,gcn,criterion,normalizer)

def train(train_loader, model, gcn, criterion, optimizer, epoch, normalizer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()
    model.train()
    end = time.time()
    for i, (input,target_charge,_) in enumerate(train_loader):
        data_time.update(time.time() - end)
        with torch.no_grad():
            input_var = (input[0].cuda(),
                        input[1].cuda(),
                        input[2].cuda(),
                        input[3].cuda(),
                        input[4].cuda(),
                        input[5].cuda())
            structure_feature = gcn.Encoding(*input_var)
        atoms_fea = torch.cat((input[0],input[7]),dim=-1)
        input_var2 = (atoms_fea.cuda(),
                        input[1].cuda(),
                        input[2].cuda(),
                        input[3].cuda(),
                        input[4].cuda(),
                        input[5].cuda(),
                        structure_feature
                        ) ; target = target_charge
        target_normed = normalizer.norm(target)
        target_var = target_normed.cuda()
        output = model(*input_var2)
        loss = criterion(output, target_var)
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

def validate(val_loader,model,gcn,criterion,normalizer):
    batch_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()
    model.eval()
    end = time.time()
    for i, (input,target_charge,_) in enumerate(val_loader):
        with torch.no_grad():
            input_var = (input[0].cuda(),
                        input[1].cuda(),
                        input[2].cuda(),
                        input[3].cuda(),
                        input[4].cuda(),
                        input[5].cuda())
            structure_feature = gcn.Encoding(*input_var)
            atoms_fea = torch.cat((input[0],input[7]),dim=-1)
            input_var2 = (atoms_fea.cuda(),
                            input[1].cuda(),
                            input[2].cuda(),
                            input[3].cuda(),
                            input[4].cuda(),
                            input[5].cuda(),
                            structure_feature
                            ) ; target = target_charge
            target_normed = normalizer.norm(target)
            target_var = target_normed.cuda()
            output = model(*input_var2)
            loss = criterion(output, target_var)
            mae_error = mae(normalizer.denorm(output.data.cpu()),target)
            losses.update(loss.item(), target.size(0))
            mae_errors.update(mae_error, target.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            print('Test: [{0}/{1}]\t'
          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
          'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
          i, len(val_loader), batch_time=batch_time, loss=losses,
          mae_errors=mae_errors)) 
    star_label = '*'
    print(' {star} MAE {mae_errors.avg:.3f}'.format(star=star_label,mae_errors=mae_errors))
    print(normalizer.denorm(output)[0:10])
    print(target[0:10])
    return mae_errors.avg

if __name__ == '__main__':
	main()
