import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.utils import Accumulator, update_lr


def train(net: nn.Module, train_iter: DataLoader, test_iter: DataLoader, num_epochs: int, lr, optimizer: torch.optim.Optimizer, save_dir: str='./models', log_id: str='alchemy'):
	os.makedirs(save_dir, exist_ok=True)

	# set up devices
	if not torch.cuda.is_available():
		device = torch.device('cpu')
	else:
		device = torch.device('cuda')
	net.to(device)

	num_batches = len(train_iter)
	loss = nn.CrossEntropyLoss(reduction='none')

	for epoch in range(num_epochs):
		# training
		net.train()
		# define metrics: loss, sample
		metrics = Accumulator(3)
		for i, (X, y) in enumerate(train_iter):
			# move to device
			X = X.to(device)
			y = y.to(device)
			# calculate yhat
			yhat = net(X)
			# calculate loss
			loss_ = loss(yhat, y)
			# backward gradient
			(loss_.sum() / X.shape[0]).backward()
			# update learning rate if available
			if callable(lr):
				update_lr(optimizer, lr(epoch))
			# step params
			optimizer.step()
			optimizer.zero_grad()
			# update metrics
			metrics.add(loss_.sum(), X.shape[0])
			# print progress
			print(f'epoch {epoch + 1}/{num_epochs} batch {i + 1}/{num_batches} loss: {metrics[0] / metrics[1]}')
		
		print('========= Testing =========')
		# testing
		net.eval()
		# define metrics: correct, sample
		metrics = Accumulator(2)
		with torch.no_grad():
			for i, (X, y) in enumerate(test_iter):
				# move to device
				X = X.to(device)
				y = y.to(device)
				# calculate yhat
				yhat = net(X)
				# calculate accur
				_, idx = torch.max(yhat, dim=1)
				# print(y)
				# print(idx)
				# print(yhat)
				correct = (idx == y)
				metrics.add(correct.sum(), X.shape[0])
		print(f'RAW: epoch {epoch + 1}/{num_epochs} accuracy: {metrics[0] / metrics[1]}')
		metrics = Accumulator(2)
		with torch.no_grad():
			for _ in range(10):
				for i, (X, y) in enumerate(train_iter):
					# move to device
					X = X.to(device)
					y = y.to(device)
					# calculate yhat
					yhat = net(X)
					# calculate accur
					_, idx = torch.max(yhat, dim=1)
					# print(y)
					# print(idx)
					# print(yhat)
					correct = (idx == y)
					metrics.add(correct.sum(), X.shape[0])
		print(f'AUG: epoch {epoch + 1}/{num_epochs} accuracy: {metrics[0] / metrics[1]}')
		print('===========================')
		
		# save model
		torch.save(net.state_dict(), os.path.join(save_dir, f'./{log_id}-model-{epoch}.pth'))
		# save optim
		torch.save(optimizer.state_dict(), os.path.join(save_dir, f'./{log_id}-optim-{epoch}.pth'))
