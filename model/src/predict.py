# -*- coding: utf-8 -*-
# Copyright 2022 Minggui Song.
# All rights reserved.

"""Represent a collect flnc information.

What's here:

Train positive and negative data sets.
-------------------------------------------

Classes:
    - Predict
"""
from logging import getLogger
from statistics import mode
from src.sys_output import Output
import time

import numpy as np
from pathlib import Path
from src.config import config
from catboost import CatBoostClassifier

logger = getLogger(__name__)  # pylint: disable=invalid-name
class Predict(object):
    """
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

    def load_data(self) -> None:
        self.output.info('Loading data ...')

        input_sp_dir = Path(self.args.input_dir)
        features_set = set(self.args.matrix_name)

        self.train_data = np.array([])
        self.valid_data = np.array([])
        self.test_data = np.array([])

        if 'DL' in features_set:
            DL_test_data = np.load(input_sp_dir / 'DL_predict_data.npy', allow_pickle=True)
            if not any(self.train_data):
                self.test_data = DL_test_data
        if 'ST' in features_set:
            ST_test_data = np.load(input_sp_dir / 'ST_predict_data.npy', allow_pickle=True)
            if not any(self.train_data):
                self.test_data = ST_test_data
            else:
                self.test_data = np.vstack((self.test_data, ST_test_data))       
        if 'OT' in features_set:
            OT_test_data = np.load(input_sp_dir / 'OT_predict_data.npy', allow_pickle=True)
            if not any(self.train_data):
                self.test_data = OT_test_data
            else:
                self.test_data = np.vstack((self.test_data, OT_test_data))

        np.save(input_sp_dir / 'PEAm6A_pridict_data.npy', self.test_data)
        

        self.output.info('Finish loading data !!!')

    def predict_model(self) -> None:
        cbc = CatBoostClassifier()
        cbc.load_model(self.args.model_name)
        cbc_pred_test_array = cbc.predict(self.test_data)
        cbc_pred_test_proba = cbc.predict_proba(self.test_data)
        np.savetxt(f'{self.args.ouput}/predict_proba.txt',cbc_pred_test_proba)
        np.savetxt(f'{self.args.ouput}/predict_class.txt',cbc_pred_test_array)

    def process(self) -> None:
        self.output.info('Starting predicting data Process.')
        logger.debug('Starting predicting data Process.')

        self.load_data()
        self.predict_model()

        self.output.info('Completed predicting data Process.')
        logger.debug('Completed predicting data Process.')