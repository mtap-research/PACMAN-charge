import torch
import shutil

class Normalizer(object):
	def __init__(self, tensor):
		self.mean = torch.mean(tensor)
		self.std = torch.std(tensor)
	def norm(self, tensor):
		return (tensor - self.mean) / self.std
	def denorm(self, normed_tensor):
		return normed_tensor * self.std + self.mean
	def state_dict(self):
		return {'mean': self.mean,'std': self.std}
	def load_state_dict(self, state_dict):
		self.mean = state_dict['mean']
		self.std = state_dict['std']

def mae(prediction, target):
	return torch.mean(torch.abs(target - prediction))

def sampling(csv_path):
    import csv
    with open(csv_path,'r') as f:
        reader = csv.reader(f)
        x = [row for row in reader]
    result = []
    for i in range(len(x)):
        temp = x[i]
        result.append(float(temp[1]))
    return torch.Tensor(result)

class AverageMeter(object):
	def __init__(self):
		self.reset()
	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0
	def update(self,val,n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def save_checkpoint(state,is_best,chk_name,best_name):
	torch.save(state, chk_name)
	if is_best:
		shutil.copyfile(chk_name,best_name)
