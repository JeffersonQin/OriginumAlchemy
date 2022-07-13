import torch
import torch.nn as nn
import torchvision.models
from torchvision.models.feature_extraction import create_feature_extractor


__all__ = ['ConvUnit', 'ResNet18Classifier']


class ConvUnit(nn.Module):
	"""Convolutional Unit, consists of conv2d, batchnorm, leaky_relu"""

	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
		super(ConvUnit, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
		self.bn = nn.BatchNorm2d(out_channels)
		self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)

	def forward(self, X: torch.Tensor):
		X = self.conv(X)
		X = self.bn(X)
		X = self.leaky_relu(X)
		return X


class ResNet18Classifier(nn.Module):
	def __init__(self, num_classes: int):
		super(ResNet18Classifier, self).__init__()

		# feature extractor
		self.resnet = torchvision.models.resnet18()
		self.backbone = create_feature_extractor(self.resnet, return_nodes={
			'layer4': 'main'
		})

		self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
		self.linear1 = nn.Linear(512, 1024)
		self.linear2 = nn.Linear(1024, num_classes)
		self.head = nn.Sequential(self.avg, nn.Flatten(), self.linear1, self.linear2)

	def forward(self, X: torch.Tensor):
		b = self.backbone(X)['main']
		return self.head(b)
