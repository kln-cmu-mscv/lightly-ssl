# from lightly.data.collate import BaseCollateFunction
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from typing import List
from . import video_transforms as transforms

class BaseSSLCollateFunction(nn.Module):

	def __init__(self, transform: torchvision.transforms.Compose):

		super(BaseSSLCollateFunction, self).__init__()
		self.transform = transform

	def forward(self, batch: List[tuple]):

		batch_size = len(batch)

		transforms = [self.transform(batch[i % batch_size][0][0]).unsqueeze(0)
						for i in range(2 * batch_size)]

		transforms_enhanced = [self.transform(batch[i % batch_size][0][1]).unsqueeze(0)
						for i in range(2 * batch_size)]

		# list of labels
		labels = torch.LongTensor([item[1] for item in batch])
		# list of filenames
		fnames = [item[2] for item in batch]

		# tuple of transforms
		# transforms = (
		# 	torch.cat(transforms[:batch_size], 0),
		# 	torch.cat(transforms[batch_size:], 0)
		# )

		transforms = (
			torch.cat(transforms, 0),
			torch.cat(transforms_enhanced, 0)
		)

		return transforms, labels, fnames

class ARIDSSLCollateFunction(BaseSSLCollateFunction):
	def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
		normalize = transforms.Normalize(mean=mean, std=std)
		self.transform = transforms.Compose([
								transforms.RandomScale(make_square=True, aspect_ratio=[0.8, 1./0.8], slen=[224, 288]),
								transforms.RandomCrop((224, 224)), # insert a resize if needed
								transforms.RandomHorizontalFlip(),
								transforms.RandomHLS(vars=[15, 35, 25]),
								transforms.ToTensor(),
								normalize,
							])
		super(ARIDSSLCollateFunction, self).__init__(self.transform)


class BaseCLSCollateFunction(nn.Module):

	def __init__(self, transform: torchvision.transforms.Compose):

		super(BaseCLSCollateFunction, self).__init__()
		self.transform = transform

	def forward(self, batch: List[tuple]):

		batch_size = len(batch)

		transforms = [self.transform(batch[i % batch_size][0][0]).unsqueeze(0)
						for i in range(batch_size)]

		# list of labels
		labels = torch.LongTensor([item[1] for item in batch])
		# list of filenames
		fnames = [item[2] for item in batch]

		# tuple of transforms
		transforms = torch.cat(transforms, 0)

		return transforms, labels, fnames

class ARIDCLSCollateFunction(BaseCLSCollateFunction):
	def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
		normalize = transforms.Normalize(mean=mean, std=std)
		self.transform = transforms.Compose([
								# transforms.RandomScale(make_square=True, aspect_ratio=[0.8, 1./0.8], slen=[224, 288]),
								# transforms.RandomCrop((224, 224)), # insert a resize if needed
                                transforms.Resize((224, 224)),
								transforms.RandomHorizontalFlip(),
								# transforms.RandomHLS(vars=[15, 35, 25]),
								transforms.ToTensor(),
								normalize,
							])
		super(ARIDCLSCollateFunction, self).__init__(self.transform)


class BaseSUPCollateFunction(nn.Module):

	def __init__(self, transform: torchvision.transforms.Compose):

		super(BaseSUPCollateFunction, self).__init__()
		self.transform = transform

	def forward(self, batch: List[tuple]):

		batch_size = len(batch)

		transforms = [self.transform(batch[i % batch_size][0]).unsqueeze(0)
						for i in range(batch_size)]

		# list of labels: 
		labels = torch.LongTensor([item[1] for item in batch])
		# one_hot_labels = torch.zeros((batch_size, 11))
		# one_hot_labels[torch.arange(batch_size),labels] = 1

		# list of filenames
		fnames = [item[2] for item in batch]

		# tuple of transforms
		transforms = torch.cat(transforms, 0)

		return transforms, labels, fnames

class ARIDSUPCollateFunction(BaseSUPCollateFunction):
	def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
		normalize = transforms.Normalize(mean=mean, std=std)
		self.transform = transforms.Compose([
								transforms.RandomScale(make_square=True, aspect_ratio=[0.8, 1./0.8], slen=[224, 288]),
								transforms.RandomCrop((224, 224)), # insert a resize if needed
								transforms.RandomHorizontalFlip(),
								transforms.RandomHLS(vars=[15, 35, 25]),
								transforms.ToTensor(),
								normalize,
							])
		super(ARIDSUPCollateFunction, self).__init__(self.transform)
