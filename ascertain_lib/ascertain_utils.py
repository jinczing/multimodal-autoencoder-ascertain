import os
import scipy.io
import random
import csv
import numpy as np
import matplotlib.pyplot as plt
import sys, traceback, getpass, time, re
from abc import ABC, abstractmethod
from itertools import groupby
from pandas import DataFrame

"""
A dataset design for multimodal dataset and specified to facilitate high-level experiment configuration,
e.g. modality selection, which-wise train-test split, csv conversion, pandas dataframe conversion, configured shuffle
"""
class MultimodalDataset(ABC):
	def __init__(self, csv_file='', selected_modalities=[], selected_features=[], split_ratio='80-20', split_mode='across-subject'
		, debug=False):
		"""
		Constructor

		Args:
			csv_file: the source csv file which initiate dataset, rigid format must be followed
				columes: subject_id, within-subject instance id(case_id), (label_name)_label naming style for labels,
				 		 modality_features naming style for features, 

			selected_modalities: the selected modalities specified in modality name, can't
								 used with selected_features in parallel, can't be named with
								 'id' or 'label' in them

			selected_features: the selected features specified in feature index number, can't used
							   with selected_modalities in parallel, can't be named with 'id' or
							   'label' in them

			split_ratio: ratio for train-test split
				format: e.g. 80-20: 80% train, 20% test, 73-27: 73% train, 27% test

			split_mode: specify which model to use for train-test split: within-subject and across-subject

			debug: whether to print debugging inforamtion

		"""
		self.csv_file = csv_file
		self.selected_modalities = selected_modalities
		self.selected_features = selected_features
		self.split_ratio = split_ratio
		self.split_mode = split_mode
		self.data = None # type: ndarray, data, contain not column names
		self.columns = None # type: ndarray
		self.train = None # type: list
		self.test = None # type: list

		self.debug = debug

		# check invalid initialization parameters
		if not csv_file or (not selected_features and not selected_modalities):
			raise Exception("some paremeter must not be empty")

		# initialization
		self.from_csv_file()
		self.select_modalities()
		self.train_test_split()


	def from_csv_file(self):
		"""
		convert csv to np array
		"""
		with open(self.csv_file) as file:
			reader = csv.reader(file)
			data = list(reader)
			self.columns = np.asarray(data[0])
			self.data = np.asarray(data[1:]).astype(float)

		self.configure_column_index()

	def configure_column_index(self):
		try:
			self.subject_id_index = np.where(self.columns == 'subject_id')[0][0]
			self.case_id_index = np.where(self.columns == 'case_id')[0][0]
			print(self.case_id_index)
		except Exception as e:
			print("incorrect csv file format: ", e)

	def select_modalities(self):
		selected_index = [i for i in range(len(self.columns)) 
							if (not self.is_column_feature(i)) or (self.is_column_feature(i) and (self.columns[i].split('_')[0] in self.selected_modalities)) ]
		self.data = self.data[:,selected_index]
		self.columns = [self.columns[i] for i in selected_index]

	def select_features(self):
		pass

	def shuffle_subjects(self):
		pass

	def shuffle_features(self):
		pass

	def shuffle_within_subject_instance(self):
		pass

	def train_test_split(self):
		if(self.split_mode=='across-subject'):
			self.train_test_split_across_subject()
		else:
			self.train_test_split_within_subject()

	def train_test_split_across_subject(self):
		"""
		split train and test dataset subject-wise based on self.split_ratio
		group data upon subject id
		"""
		data_grouped = [list(it) for k, it in groupby(self.data.tolist(), lambda x: x[self.subject_id_index])]
		split_index = int((len(data_grouped))*(float(self.split_ratio.split('-')[0])/100))
		self.train = data_grouped[:split_index]
		self.train = [item for sublist in self.train for item in sublist] # flatten
		self.test = data_grouped[split_index:]
		self.test = [item for sublist in self.test for item in sublist] # flatten

	def train_test_split_within_subject(self):
		"""
		split train and test dataset within-subject-wise based on self.split_ratio
		group data upon case id
		"""
		within_subjects_indexed_data = sorted(self.data.tolist(), key=lambda item: int(item[self.case_id_index]))
		data_grouped = [list(it) for k, it in groupby(within_subjects_indexed_data, lambda item: item[self.case_id_index])]
		split_index = int((len(data_grouped))*(float(self.split_ratio.split('-')[0])/100))
		self.train = data_grouped[:split_index]
		self.train = [item for sublist in self.train for item in sublist] # flatten
		self.test = data_grouped[split_index:]
		self.test = [item for sublist in self.test for item in sublist] # flatten

	def to_csv(self, train_file_path='./', test_file_path='./'):
		"""
		export split train and test dataset as csv file
		"""
		assert (self.train is not None and self.test is not None)

		self.train.insert(0, self.columns)
		with open(train_file_path, 'w', newline='') as csv_file:
			writer = csv.writer(csv_file)
			writer.writerows(self.train)

		self.test.insert(0, self.columns)
		with open(test_file_path, 'w', newline='') as csv_file:
			writer = csv.writer(csv_file)
			writer.writerows(self.test)

	def to_pandas(self):
		"""
		convert train and test dataset to pandas dataframe
		"""
		return DataFrame(self.train), DataFrame(self.test)

	def _debug(self):
		if(not self.debug):
			return
		print(len(self.train), len(self.test), self.train[0], self.test[0])

	def is_column_feature(self, column_index):
		return ('label' not in self.columns[column_index] and 'id' not in self.columns[column_index])