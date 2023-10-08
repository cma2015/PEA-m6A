# -*- coding: utf-8 -*-
# Copyright 2022 Minggui Song.
# All rights reserved.
#

"""Represent a collect flnc information.

What's here:

Extract sample sequences features.
-------------------------------------------

Classes:
    - FeaturesEncoding
"""
import src.models
import src.weaknets
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.keras import Input

from distutils.command.config import config
from logging import getLogger
from src.sys_output import Output
import numpy as np
from src.utils import embed, asc2one
from pathlib import Path
from src.config import config
import subprocess

logger = getLogger(__name__)  # pylint: disable=invalid-name



class FeaturesEncoding(object):
    """.

    Attributes:
        - args: Arguments.
        - output: Output info, warning and error.

    """

    def __init__(self, arguments) -> None:
        """Initialize CollectFlncInfo."""
        self.args = arguments
        self.output = Output()
        self.output.info(
            f'Initializing {self.__class__.__name__}: (args: {arguments}.')
        logger.debug(
            f'Initializing {self.__class__.__name__}: (args: {arguments}.')

    def onehot(self) -> None:
        with open(f'{self.args.input}', 'r') as f1:
            pos_lines = f1.readlines()

        for i in range(1, len(pos_lines), 2):
            pos_lines[i] = pos_lines[i].rstrip('\n')
            pos_lines[i] = pos_lines[i].lstrip('>')
        
        pos_token = []

        for i in range(1, len(pos_lines), 2):
            sub_array = asc2one(np.fromstring(pos_lines[i], dtype=np.int8))
            pos_token.append(sub_array)
            pos_tokens = np.array(pos_token)

        inst_len = self.args.instance_length
        inst_stride = self.args.stride_length
        out_bags = []
        for seq in pos_tokens:
            ont_hot_bag = embed(seq, inst_len, inst_stride)
            out_bags.append(ont_hot_bag)

        out_bags = np.asarray(out_bags)
        np.save(f'{self.args.output}/{self.args.species}_onehot_features.npy', out_bags)

    def statistics(self) -> None:
        subprocess.call(["Rscript", "Statistics.R", f'self.args.input'])
        with open(f'{self.args.input}', 'r') as fin:
            lines = fin.readlines()
        kmer_features = []
        for iter in range(1, len(lines)):
            in_array = lines[iter].rstrip('\n').split(',')
            kmer_features.append(in_array[0:])
        np.save(f'{self.args.output}/{self.args.species}_ST_features.npy', kmer_features)

    def deeplearning(self) -> None:
        with open(f'{self.args.input}', 'r') as f1:
            pos_lines = f1.readlines()

        for i in range(1, len(pos_lines), 2):
            pos_lines[i] = pos_lines[i].rstrip('\n')
            pos_lines[i] = pos_lines[i].lstrip('>')
        
        pos_token = []

        for i in range(1, len(pos_lines), 2):
            sub_array = asc2one(np.fromstring(pos_lines[i], dtype=np.int8))
            pos_token.append(sub_array)
            pos_tokens = np.array(pos_token)

        inst_len = self.args.instance_length
        inst_stride = self.args.stride_length
        out_bags = []
        for seq in pos_tokens:
            ont_hot_bag = embed(seq, inst_len, inst_stride)
            out_bags.append(ont_hot_bag)

        out_bags = np.asarray(out_bags)

        model = src.models.SingleWeakRM()
        model.extractor.build(input_shape=(1, None, self.args.instance_length, 4))
        model.extractor.call(Input(shape=(None, self.args.instance_length, 4)))
        model.extractor.load_weights(f'{self.args.model_name}')

        output_list = []
        for iterj in range(0, out_bags.shape[0]):
            bag_features, _ = model.extractor(out_bags[iterj].reshape(1, -1, self.args.instance_length, 4).astype(np.float32), training=False)

            output_list.append(bag_features[0])

        output_list = np.array(output_list)
        np.save(f'{self.args.output}/{self.args.species}_DL_features.npy', output_list)


    def process(self) -> None:
        if self.args.encoding == "onehot":
            self.output.info('Starting encoding data by one-hot.')
            self.onehot()
            self.output.info('Completed encoding data by one-hot.')
        elif self.args.encoding == "statistics":
            self.output.info('Starting encoding data by statistics-based.')
            self.statistics()
            self.output.info('Completed encoding data by statistics-based.')
        else:
            self.output.info('Starting encoding data by deep learning.')
            self.deeplearning()
            self.output.info('Completed encoding data by deep learning.')