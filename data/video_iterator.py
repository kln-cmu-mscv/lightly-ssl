"""
Original Author: Yunpeng Chen
Adaptation Author: Yuecong Xu
"""
import os
import cv2
import csv
import numpy as np

import torch.utils.data as data
import logging


class Video(object):
	"""basic Video class"""

	def __init__(self, vid_path):
		self.open(vid_path)

	def __del__(self):
		self.close()

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_value, traceback):
		self.__del__()

	def reset(self):
		self.close()
		self.vid_path = None
		self.frame_count = -1
		self.faulty_frame = None
		return self

	def open(self, vid_path):
		assert os.path.exists(vid_path), "VideoIter:: cannot locate: `{}'".format(vid_path)

		# close previous video & reset variables
		self.reset()

		# try to open video
		cap = cv2.VideoCapture(vid_path)
		if cap.isOpened():
			self.cap = cap
			self.vid_path = vid_path
		else:
			raise IOError("VideoIter:: failed to open video: `{}'".format(vid_path))

		return self

	def count_frames(self, check_validity=False):
		offset = 0
		if self.vid_path.endswith('.flv'):
			offset = -1
		unverified_frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) + offset
		if check_validity:
			verified_frame_count = 0
			for i in range(unverified_frame_count):
				self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
				if not self.cap.grab():
					logging.warning("VideoIter:: >> frame (start from 0) {} corrupted in {}".format(i, self.vid_path))
					break
				verified_frame_count = i + 1
			self.frame_count = verified_frame_count
		else:
			self.frame_count = unverified_frame_count
		assert self.frame_count > 0, "VideoIter:: Video: `{}' has no frames".format(self.vid_path)
		return self.frame_count

	def extract_frames(self, idxs, force_color=True):
		frames = self.extract_frames_fast(idxs, force_color)
		if frames is None:
			# try slow method:
			frames = self.extract_frames_slow(idxs, force_color)
		return frames

	def extract_frames_fast(self, idxs, force_color=True):
		assert self.cap is not None, "No opened video."
		if len(idxs) < 1:
			return []

		frames = []
		pre_idx = max(idxs)
		for idx in idxs:
			assert (self.frame_count < 0) or (idx < self.frame_count), \
				"idxs: {} > total valid frames({})".format(idxs, self.frame_count)
			if pre_idx != (idx - 1):
				self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
			res, frame = self.cap.read() # in BGR/GRAY format
			pre_idx = idx
			if not res:
				self.faulty_frame = idx
				return None
			if len(frame.shape) < 3:
				if force_color:
					# Convert Gray to RGB
					frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
			else:
				# Convert BGR to RGB
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			frames.append(frame)
		return frames

	def extract_frames_slow(self, idxs, force_color=True):
		assert self.cap is not None, "No opened video."
		if len(idxs) < 1:
			return []

		frames = [None] * len(idxs)
		idx = min(idxs)
		self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
		while idx <= max(idxs):
			res, frame = self.cap.read() # in BGR/GRAY format
			if not res:
				# end of the video
				self.faulty_frame = idx
				return None
			if idx in idxs:
				# fond a frame
				if len(frame.shape) < 3:
					if force_color:
						# Convert Gray to RGB
						frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
				else:
					# Convert BGR to RGB
					frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				pos = [k for k, i in enumerate(idxs) if i == idx]
				for k in pos:
					frames[k] = frame
			idx += 1
		return frames

	def close(self):
		if hasattr(self, 'cap') and self.cap is not None:
			self.cap.release()
			self.cap = None
		return self


class VideoIter(data.Dataset):

	def __init__(self, video_prefix, video_prefix_enhanced, csv_list, sampler,
				 video_transform=None, name="<NO_NAME>", force_color=True,
				 cached_info_path=None, return_item_subpath=True, shuffle_list_seed=None, check_video=False, tolerant_corrupted_video=None):
		super(VideoIter, self).__init__()
		# load params
		self.sampler = sampler
		self.force_color = force_color
		self.video_prefix = video_prefix
		self.video_prefix_enhanced = video_prefix_enhanced
		self.video_transform = video_transform
		self.return_item_subpath = return_item_subpath
		self.backup_item = None
		if (not check_video) and (tolerant_corrupted_video is None):
			logging.warning("VideoIter:: >> `check_video' is off, `tolerant_corrupted_video' is automatically activated.")
			tolerant_corrupted_video = True
		self.tolerant_corrupted_video = tolerant_corrupted_video
		self.rng = np.random.RandomState(shuffle_list_seed if shuffle_list_seed else 0)

		# load video list
		self.video_list = self._get_video_list(video_prefix=video_prefix, csv_list=csv_list, check_video=check_video, cached_info_path=cached_info_path)

		if shuffle_list_seed is not None:
			self.rng.shuffle(self.video_list)
		logging.info("VideoIter:: iterator initialized (phase: '{:s}', num: {:d})".format(name, len(self.video_list)))

	def getitem_from_raw_video(self, index):
		# get current video info
		v_id, label, vid_subpath, frame_count = self.video_list[index]
		video_path = os.path.join(self.video_prefix_enhanced, vid_subpath)
		
		video_id = video_path.split('/')[-1]
		vid_class = video_path.split('/')[-2]
		video_path_enhanced = os.path.join(self.video_prefix, vid_class, video_id)

		faulty_frames = []
		successfule_trial = False
		try:
			with Video(vid_path=video_path) as video:
				video_enhanced = Video(vid_path=video_path_enhanced)
				if frame_count < 0:
					frame_count = video.count_frames(check_validity=False)
				for i_trial in range(20):
					# dynamic sampling
					sampled_idxs = self.sampler.sampling(range_max=frame_count, v_id=v_id, prev_failed=(i_trial>0))
					if set(sampled_idxs).intersection(faulty_frames):
						continue
					prev_sampled_idxs = sampled_idxs
					# extracting frames
					sampled_frames = video.extract_frames(idxs=sampled_idxs, force_color=self.force_color)
					sampled_frames_enhanced = video_enhanced.extract_frames(idxs=sampled_idxs, force_color=self.force_color)
					if sampled_frames is None:
						faulty_frames.append(video.faulty_frame)
					else:
						successfule_trial = True
						break
		except IOError as e:
			logging.warning(">> I/O error({0}): {1}".format(e.errno, e.strerror))

		if not successfule_trial:
			assert (self.backup_item is not None), \
				"VideoIter:: >> frame {} is error & backup is inavailable. [{}]'".format(faulty_frames, video_path)
			# logging.warning(">> frame {} is error, use backup item! [{}]".format(faulty_frames, video_path))
			with Video(vid_path=self.backup_item['video_path']) as video:
				sampled_frames = video.extract_frames(idxs=self.backup_item['sampled_idxs'], force_color=self.force_color)
		elif self.tolerant_corrupted_video:
			# assume the error rate less than 10%
			if (self.backup_item is None) or (self.rng.rand() < 0.1):
				self.backup_item = {'video_path': video_path, 'sampled_idxs': sampled_idxs}

		clip_input = np.concatenate(sampled_frames, axis=2)
		clip_input_enhanced = np.concatenate(sampled_frames_enhanced, axis=2)
		# apply video augmentation
		if self.video_transform is not None:
			clip_input = self.video_transform(clip_input)
			clip_input_enhanced = self.video_transform(clip_input_enhanced)
		return [clip_input, clip_input_enhanced], label, vid_subpath


	def __getitem__(self, index):
		succ = False
		attempts = 0
		while not succ and attempts < 5:
			try:
				clip_input, label, vid_subpath = self.getitem_from_raw_video(index)
				succ = True
			except Exception as e:
				index = self.rng.choice(range(0, self.__len__()))
				attempts = attempts + 1
				logging.warning("VideoIter:: ERROR!! (Force using another index:\n{})\n{}".format(index, e))

		if self.return_item_subpath:
			return clip_input, label, vid_subpath
		else:
			return clip_input, label


	def __len__(self):
		return len(self.video_list)


	def _get_video_list(self, video_prefix, csv_list, check_video=False, cached_info_path=None):
		# format:
		# [videoID, videoPath, classID] --> [v_id, label, video_subpath, frame_count(-1)]
		assert os.path.exists(video_prefix), "VideoIter:: failed to locate: `{}'".format(video_prefix)
		assert os.path.exists(csv_list), "VideoIter:: failed to locate: `{}'".format(csv_list)

		# building dataset
		video_list = []
		new_video_info = {}
		logging_interval = 100

		with open(csv_list, newline="") as f:
			reader = csv.DictReader(f)
			lines = len(list(reader))
			logging.info("VideoIter:: found {} videos in `{}'".format(lines, csv_list))
		f.close()

		with open(csv_list, newline="") as f:
			reader = csv.DictReader(f)
			for line in reader:
				v_id = line['VideoID']
				try:
					label = line['ClassID']
				except:
					label = -1
				video_subpath = line['Video']
				
				video_path = os.path.join(video_prefix, video_subpath)
				if not os.path.exists(video_path):
					# logging.warning("VideoIter:: >> cannot locate `{}'".format(video_path))
					continue
				if check_video:
					frame_count = self.video.open(video_path).count_frames(check_validity=True)
					new_video_info.update({video_subpath: frame_count})
				else:
					frame_count = -1
				info = [int(v_id), int(label), video_subpath, frame_count]
				video_list.append(info)
				if check_video and (i % logging_interval) == 0:
					logging.info("VideoIter:: - Checking: {:d}/{:d}, \tinfo: {}".format(i, len(lines), info))
		
		return video_list
