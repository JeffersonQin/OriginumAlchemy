import os
from random import random
import torchvision
import torchvision.transforms as transforms
from typing import Tuple
from torch.utils.data import DataLoader, Dataset
from PIL import Image


def remove_alpha(png):
	background = Image.new('RGBA', png.size, (0, 0, 0))
	alpha_composite = Image.alpha_composite(background, png)
	alpha_composite_3 = alpha_composite.convert('RGB')
	return alpha_composite_3


class ArknightsSkillIconDataset(Dataset):
	def __init__(self, path: str, img_size: int, data_aug: bool):
		self.data_aug = data_aug
		self.transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Resize((img_size, img_size))])
		self.images = []
		for _, _, fs in os.walk(path):
			for f in fs:
				img = Image.open(os.path.join(path, f))
				self.images.append(img)


	def __len__(self) -> int:
		return len(self.images)


	def __getitem__(self, idx) -> Tuple[Image.Image, int]:
		img = self.images[idx]
		# data augmentation
		img = remove_alpha(img)		
		# transform to tensor
		img = self.transform(img)
		# adjust brightness
		if random() > 0.5 and self.data_aug:
			rand_brightness = random() * 0.5 + 0.5
			img = torchvision.transforms.functional.adjust_brightness(img, rand_brightness)
		return img, idx


def get_icon_count(path: str):
	cnt = 0
	for _, _, fs in os.walk(path):
		cnt += len(fs)
	return cnt


def load_icon_data(path: str, img_size: int, batch_size, **kwargs) -> DataLoader:
	assert img_size % 32 == 0, "image size must be a multiple of 32, because resnet will downsample image by 32."
	return (DataLoader(ArknightsSkillIconDataset(path, img_size, True), batch_size=batch_size, **kwargs), 
			DataLoader(ArknightsSkillIconDataset(path, img_size, False), batch_size=batch_size, **kwargs))
