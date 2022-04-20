import os
import logging
import torch
import torch.nn as nn

from . import video_sampler as sampler
from . import video_transforms as transforms
from .video_iterator import VideoIter
from typing import List

import torchvision
import torchvision.transforms as T

# from lightly.data.collate import BaseCollateFunction
class BaseCollateFunction(nn.Module):

	def __init__(self, transform: torchvision.transforms.Compose):

		super(BaseCollateFunction, self).__init__()
		self.transform = transform

	def forward(self, batch: List[tuple]):

		batch_size = len(batch)

		transforms = [self.transform(batch[i % batch_size][0]).unsqueeze(0)
						for i in range(2 * batch_size)]

		# list of labels
		labels = torch.LongTensor([item[1] for item in batch])
		# list of filenames
		fnames = [item[2] for item in batch]

		# tuple of transforms
		transforms = (
			torch.cat(transforms[:batch_size], 0),
			torch.cat(transforms[batch_size:], 0)
		)

		return transforms, labels, fnames

class ARIDCollateFunction(BaseCollateFunction):
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
		super(ARIDCollateFunction, self).__init__(self.transform)

def get_arid(data_root='./dataset/ARID', clip_length=8, train_interval=2,
			mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
			seed=0, return_item_subpath=False, **kwargs):
	""" data iter for ARID
	"""
	logging.debug("VideoIter:: clip_length = {}, interval = [train: {}], seed = {}".format(clip_length, train_interval, seed))

	train_sampler = sampler.RandomSampling(num=clip_length, interval=train_interval, speed=[1.0, 1.0], seed=(seed+0))
	train = VideoIter(video_prefix=os.path.join(data_root, 'raw', 'train_data'),
					  csv_list=os.path.join(data_root, 'raw', 'list_cvt', 'ARID1.1_t1_train_pub.csv'),
					  sampler=train_sampler,
					  force_color=True,
					  video_transform=None,
					  name='train',
					  shuffle_list_seed=(seed+2),
					  return_item_subpath=return_item_subpath
					  )

	return train


def creat(name, batch_size, num_workers=16, **kwargs):

	if name.upper() == 'ARID':
		train = get_arid(**kwargs)
	else:
		assert NotImplementedError("iter {} not found".format(name))


	train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)

	return train_loader
