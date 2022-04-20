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
