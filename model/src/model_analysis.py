# -*- coding: utf-8 -*-
# Copyright 2022 Minggui Song.
# All rights reserved.
#

"""Represent a collect flnc information.

What's here:

Preprocess m6A-Seq data through SnakeMake scripts.
-------------------------------------------

Classes:
    - ModelAnalysis
"""
import pandas as pd
import shap
import catboost
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as pl
from distutils.command.config import config
from logging import getLogger
from src.sys_output import Output
import numpy as np
from src.utils import create_folder, embed
from pathlib import Path
from src.config import config
shap.initjs()

logger = getLogger(__name__)  # pylint: disable=invalid-name


class ModelAnalysis(object):
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

    def summary(self) -> None:
        with open(f'{self.args.input}/col_names', 'r') as fin:
            lines = fin.readlines()
        col = [i.rstrip('\n')for i in lines]
        train_data = np.load(f'{self.args.input}/PEAm6A_train_data.npy', allow_pickle=True)
        train_label = np.load(f'{self.args.input}/train_label.npy', allow_pickle=True)
        cbc = CatBoostClassifier()
        cbc.load_model(self.args.model)
        cbc_prob = cbc.predict_proba(train_data)

        data_df = pd.DataFrame(train_data,  columns=col, dtype=float)
        data_df['label'] = train_label
        data_df['prob'] = cbc_prob[:,1]

        shap_values = cbc.get_feature_importance(Pool(data_df, train_labels_array), type='ShapValues')

        expected_value = shap_values[0,-1]
        shap_values = shap_values[:,:-1]


        newCmap = LinearSegmentedColormap.from_list("", ['#4e79a7','#A74E79','#e25b5c'])

        shap.summary_plot(shap_values, data_df,cmap= newCmap,max_display=10,show=False)
        pl.savefig(f"{self.args.output}/{self.args.output_name}_shap_summary.pdf",dpi=1000)

    def dependence(self) -> None:
        with open(f'{self.args.input}/col_names', 'r') as fin:
            lines = fin.readlines()
        col = [i.rstrip('\n')for i in lines]
        train_data = np.load(f'{self.args.input}/PEAm6A_train_data.npy', allow_pickle=True)
        train_label = np.load(f'{self.args.input}/train_label.npy', allow_pickle=True)
        cbc = CatBoostClassifier()
        cbc.load_model(self.args.model)
        cbc_prob = cbc.predict_proba(train_data)

        data_df = pd.DataFrame(train_data,  columns=col, dtype=float)
        data_df['label'] = train_label
        data_df['prob'] = cbc_prob[:,1]

        shap_values = cbc.get_feature_importance(Pool(data_df, train_labels_array), type='ShapValues')

        expected_value = shap_values[0,-1]
        shap_values = shap_values[:,:-1]

        newCmap = LinearSegmentedColormap.from_list("", ['#4e79a7','#A74E79','#e25b5c'])

        shap.dependence_plot(self.args.features1,shap_values, data_df, cmap = newCmap,interaction_index=self.args.features2,show=False)
        pl.savefig(f"{self.args.output}/{self.args.output_name}_shap_dependence.pdf",dpi=1000) 

    def process(self) -> None:
        """Call the preprocessing data process object."""
        self.output.info('Starting preprocessing data Process.')
        if self.args.plot == 'summary':
            self.summary()
        else:
            self.dependence()
        self.output.info('Completed preprocessing data Process.')

