import random
import numpy
import torch
from utils.utils import CosineAnnealingScheduler, LearningRateSchedulerComposer, LinearWarmupScheduler
from utils.winit import weight_init
from alchemy import data_pipeline, model, trainer


seed = 8613845 # 直播间 ID
torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)
random.seed(seed)
numpy.random.seed(seed)

batch_size = 64
img_size_scale = 3
num_epoch = 400
warmup_epoch = 30
max_lr = 0.1
weight_decay = 0.0005
momentum = 0.9

lr = LearningRateSchedulerComposer([
	LinearWarmupScheduler(max_lr, warmup_epoch),
	CosineAnnealingScheduler(max_lr, num_epoch - warmup_epoch)
])

if __name__ == '__main__':
	# data loader
	train_iter, test_iter = data_pipeline.load_icon_data('icons', 32 * img_size_scale, batch_size)

	# define network
	net = model.ResNet18Classifier(num_classes=data_pipeline.get_icon_count('icons'))

	# weight init
	net.apply(weight_init)

	# define optimizer
	optimizer = torch.optim.SGD(net.parameters(), lr=lr(0), weight_decay=weight_decay, momentum=momentum)

	trainer.train(net, train_iter, test_iter, num_epoch, lr, optimizer, './models', 'resnet18-adamw')
