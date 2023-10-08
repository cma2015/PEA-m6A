# -*- coding: utf-8 -*-
# Copyright 2022 Minggui Song.
# All rights reserved.
#

"""Represent a collect flnc information.

What's here:

Preprocess m6A-Seq data through SnakeMake scripts.
-------------------------------------------

Classes:
    - DataProcess
"""

from distutils.command.config import config
from logging import getLogger
from src.sys_output import Output
import numpy as np
from src.utils import create_folder, embed
from pathlib import Path
from src.config import config

logger = getLogger(__name__)  # pylint: disable=invalid-name


class DataProcess(object):
    """The preprocess positive and negative data sets process.

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

    def token2npy(self, input_dir, output_dir) -> None:
        data_dir = input_dir
        target_dir = output_dir
        #create_folder(target_dir)

        inst_len = self.args.instance_length
        inst_stride = self.args.stride_length

        train_token = np.load(data_dir / f'train_token.npy', allow_pickle=True)
        valid_token = np.load(data_dir / f'valid_token.npy', allow_pickle=True)
        test_token = np.load(data_dir / f'test_token.npy', allow_pickle=True)

        train_label = np.load(data_dir / f'train_label.npy', allow_pickle=True)
        valid_label = np.load(data_dir / f'valid_label.npy', allow_pickle=True)
        test_label = np.load(data_dir / f'test_label.npy', allow_pickle=True)

        train_bags = []
        for seq in train_token:
            ont_hot_bag = embed(seq, inst_len, inst_stride)
            train_bags.append(ont_hot_bag)

        train_bags = np.asarray(train_bags)

        valid_bags = []
        for seq in valid_token:
            ont_hot_bag = embed(seq, inst_len, inst_stride)
            valid_bags.append(ont_hot_bag)

        valid_bags = np.asarray(valid_bags)

        test_bags = []
        for seq in test_token:
            ont_hot_bag = embed(seq, inst_len, inst_stride)
            test_bags.append(ont_hot_bag)

        test_bags = np.asarray(test_bags)

        np.save(target_dir / f'train_data.npy', train_bags)
        np.save(target_dir / f'valid_data.npy', valid_bags)
        np.save(target_dir / f'test_data.npy', test_bags)

        np.save(target_dir / f'train_label.npy', train_label)
        np.save(target_dir / f'valid_label.npy', valid_label)
        np.save(target_dir / f'test_label.npy', test_label)

    def checkdir(self) -> None:
        """"Check output positive and negative directory."""
        self.output.info('Creating output directory.')
        self.ouput_dirs = []
        self.input_dirs = []
        for iter in config.sample_names:
            self.output.info(iter)
            output_dir = Path(self.args.output_dir)
            output_sp_dir = Path(self.args.output_dir) / iter
            output_processed_dir = Path(self.args.output_dir) / iter / 'processed'
            input_sp_dir = Path(self.args.input_dir) / iter
            self.input_dirs.append(input_sp_dir)
            self.ouput_dirs.append(output_processed_dir)

            if not output_dir.is_dir():
                self.output.info('Creating output directory.')
                output_dir.mkdir()

            if not output_sp_dir.is_dir():
                self.output.info(f'Creating output dircotry of {iter}.')
                output_sp_dir.mkdir()

            if not output_processed_dir.is_dir():
                self.output.info(f'Creating output processed directory of {iter}.')
                output_processed_dir.mkdir()


    def process(self) -> None:
        """Call the preprocessing data process object."""
        self.output.info('Starting preprocessing data Process.')
        logger.debug('Starting preprocessing data Process.')

        self.checkdir()
        for iter in range(config.TRAIN.CLASSES):
            self.token2npy(self.input_dirs[iter], self.ouput_dirs[iter])

        self.output.info('Completed preprocessing data Process.')
        logger.debug('Completed preprocessing data Process.')
