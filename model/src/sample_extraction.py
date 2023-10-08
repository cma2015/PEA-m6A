# -*- coding: utf-8 -*-
# Copyright 2021 Minggui Song.
# All rights reserved.

"""Represent a collect flnc information.

What's here:

Get positive and negative data sets.
-------------------------------------------

Classes:
    - SampleExtraction
"""
from logging import getLogger
from statistics import mode
from src.sys_output import Output
import time
import numpy as np
import pandas as pd
from pathlib import Path
from src.config import config
import random
import os
import re
import pybedtools

logger = getLogger(__name__)  # pylint: disable=invalid-name

class SampleExtraction(object):
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

    def get_pos(self) -> None:
        # trim peak length <100nt or >sample
        
        data = pd.read_csv(self.args.peak, sep='\t', names=['chr', 'peak_start', 'peak_end','peak_id', 'un', 'strand'])
        data['peak_length'] = data['peak_end'] -  data['peak_start']
        data_cleaned = data[(data.peak_length >=100) & (data.peak_length <= self.args.length)].reset_index(drop=True)
        data_cleaned.to_csv(f'{self.args.output}/01_raw_data/{self.args.species}_pos_cleaned.bed',sep='\t', header=None, index=None)

        peak = pybedtools.BedTool(f'{self.args.output}/01_raw_data/{self.args.species}_pos_cleaned.bed')
        fasta = pybedtools.example_filename(f'{self.args.ann_dic}/{self.args.species}.fa')
        a = peak.sequence(fi=fasta,fo=f'{self.args.output}/01_raw_data/{self.args.species}_pos_cleaned.fa.out', name=True, s=True)
        # trim peak without RRACH
        with open(f'{self.args.output}/01_raw_data/{self.args.species}_pos_cleaned.fa.out', 'r') as fin:
            lines = fin.readlines()

        all_count_list = []
        for i in range(1, len(lines), 2):
            #进行正则匹配
            pattern = re.compile(r'[GA][GA]AC[ACT]',re.I)
            #获取postion A的位置
            a = [m.start() + 2  for m in pattern.finditer(lines[i].rstrip('\n'))]
            string_a = ",".join('%s' %id for id in a)
            all_count_list.append(len(a))

        data_cleaned['A_counts'] = all_count_list
        pos_data = data_cleaned[data_cleaned.A_counts != 0].reset_index(drop=True)
        pos_data.to_csv(f'{self.args.output}/02_peak_data/{self.args.species}_pos_peak.bed', sep='\t', header=None, index=None)

        gene_bed = pd.read_csv(f'{self.args.ann_dic}/{self.args.species}_gene_info.bed', sep='\t', names=['chr', 'gene_start', 'gene_end', 'strand', 'info', 'gene_name', 'length'])

        start_list = []
        end_list = []
        for iterj in range(0, pos_data.shape[0]):
            peak_start = pos_data.loc[iterj, 'peak_start']
            peak_end = pos_data.loc[iterj, 'peak_end']
            peak_id = pos_data.loc[iterj, 'peak_id']
            peak_length = pos_data.loc[iterj, 'peak_length']

            gene_info = gene_bed[gene_bed['gene_name'] == peak_id].reset_index(drop=True)
            if gene_info.shape[0] == 0:
                expand_peak_start = 'None'
                expand_peak_end = 'None'
            else:
                gene_start = gene_info.loc[0, 'gene_start']
                gene_end = gene_info.loc[0, 'gene_end']
                gene_length = gene_info.loc[0, 'length']

            #if gene_length < 1000:
            #    print(f'{peak_id}_{gene_length}')
            expand_length = self.args.length - peak_length
            left_expand = expand_length // 2
            right_expand = expand_length - left_expand

            expand_peak_start = peak_start - left_expand
            expand_peak_end = peak_end + right_expand

            start_list.append(expand_peak_start)
            end_list.append(expand_peak_end)

        expand_peak_data = pos_data.copy(deep=True)
        expand_peak_data['peak_start'] = start_list
        expand_peak_data['peak_end'] = end_list

        expand_peak_data[expand_peak_data.peak_start != 'None'].to_csv(f'{self.args.output}/03_sample_data/{self.args.species}_pos_sample_peak.bed', sep='\t', header=None, index=None)

    def get_neg(self) -> None:
        gene_bed = pd.read_csv(f'{self.args.ann_dic}/{self.args.species}_gene_info.bed', sep='\t', names=['chr', 'gene_start', 'gene_end',  'strand','info', 'gene_name','length'])
        gene_cleaned = gene_bed[gene_bed['length'] >= self.args.length].reset_index(drop=True)
        exons = pd.read_csv(f'{self.args.ann_dic}/{self.args.species}_exon.bed', sep='\t', names=['chr', 'type', 'exon_start', 'exons_end', 'strand', 'info','gene', 'gene_name','length'])
        pos_sample_peak = pd.read_csv(f'{self.args.output}/03_sample_data/{self.args.species}_pos_sample_peak.bed', sep='\t', names=['chr', 'peak_start', 'peak_end', 'peak_id', 'un', 'strand', 'peak_length', 'A_counts'])

        pos_length = pos_sample_peak.shape[0]

        peak_id_set = set(pos_sample_peak['peak_id'])
        non_peak_region = []
        for iter in range(0, gene_cleaned.shape[0]):
            chr = gene_cleaned.loc[iter, 'chr']
            gene_start = int(gene_cleaned.loc[iter, 'gene_start'])
            gene_end = int(gene_cleaned.loc[iter, 'gene_end'])
            strand = gene_cleaned.loc[iter, 'strand']
            gene_name = gene_cleaned.loc[iter, 'gene_name']

            if gene_name in peak_id_set:
                peak_dataframe = pos_sample_peak[pos_sample_peak['peak_id'] == gene_name].reset_index(drop=True)
                if peak_dataframe.shape[0] > 2:
                    next
                elif peak_dataframe.shape[0] == 1:
                    peak_start = int(peak_dataframe.loc[0, 'peak_start'])
                    peak_end = int(peak_dataframe.loc[0, 'peak_end'])
                    if peak_start < gene_start  and peak_end < gene_end:
                        dic = {'chr':chr, 'start': peak_end, 'end': gene_end, 'gene_name': gene_name, 'un': '.', 'strand': strand}
                        non_peak_region.append(dic)
                    elif peak_start > gene_start  and peak_end > gene_end:
                        dic = {'chr':chr, 'start': gene_start, 'end': peak_start, 'gene_name': gene_name, 'un': '.', 'strand': strand}
                        non_peak_region.append(dic)
                    elif peak_start > gene_start  and peak_end < gene_end:
                        dic = {'chr':chr, 'start': gene_start, 'end': peak_start, 'gene_name': gene_name, 'un': '.', 'strand': strand}
                        non_peak_region.append(dic)
                        dic = {'chr':chr, 'start': peak_end, 'end': gene_end, 'gene_name': gene_name, 'un': '.', 'strand': strand}
                        non_peak_region.append(dic)
            else:
                dic = {'chr':chr, 'start': gene_start, 'end': gene_end, 'gene_name': gene_name, 'un': '.', 'strand': strand}
                non_peak_region.append(dic)

        non_peak_region = pd.DataFrame(non_peak_region)
        non_peak_region['length'] = non_peak_region['end'] -  non_peak_region['start']
        non_peak_region = non_peak_region[non_peak_region['length'] >= 500]
        non_peak_region = non_peak_region.reset_index(drop=True)

        non_peak_region.to_csv(f'{self.args.output}/01_raw_data/{self.args.species}_neg_cleaned.bed', sep='\t', index=None, header=None)

        peak = pybedtools.BedTool(f'{self.args.output}/01_raw_data/{self.args.species}_neg_cleaned.bed')
        fasta = pybedtools.example_filename(f'{self.args.ann_dic}/{self.args.species}.fa')
        a = peak.sequence(fi=fasta,fo=f'{self.args.output}/01_raw_data/{self.args.species}_neg_cleaned.fa.out', name=True, s=True)

        with open(f'{self.args.output}/01_raw_data/{self.args.species}_neg_cleaned.fa.out', 'r') as fin:
            lines = fin.readlines()

        all_count_list = []
        all_posA_list = []

        for iter in range(1, len(lines), 2):
            non_peak_region_start = non_peak_region.loc[iter // 2, 'start']
            gene_name = non_peak_region.loc[iter // 2, 'gene_name']
            #进行正则匹配
            pattern = re.compile(r'[GA][GA]AC[ACT]',re.I)
            #获取postion A的位置
            a = [m.start() + 2 + non_peak_region_start  for m in pattern.finditer(lines[iter].rstrip('\n'))]
            a_in_exons = []

            exons_tmp = exons[exons['gene_name'] == gene_name].reset_index(drop=True)
            for tmp_a in a:
                for exon_iter in range(0, exons_tmp.shape[0]):
                    exon_start = exons_tmp.loc[exon_iter, 'exon_start']
                    exon_end = exons_tmp.loc[exon_iter, 'exons_end']
                    if tmp_a > exon_start and tmp_a < exon_end:
                        a_in_exons.append(tmp_a)
                        break
            all_posA_list.append(",".join('%s' %id for id in a))
            all_count_list.append(len(a))
        non_peak_region['posA'] = all_posA_list
        non_peak_region['A_counts'] = all_count_list

        non_peak_region = non_peak_region[non_peak_region['A_counts'] > 0].reset_index(drop=True)

        neg_peak_list = []

        for iter in range(0, non_peak_region.shape[0]):
            posA = non_peak_region.loc[iter, 'posA'].split(',')
            chr = non_peak_region.loc[iter, 'chr']
            non_peak_start = non_peak_region.loc[iter, 'start']
            non_peak_end = non_peak_region.loc[iter, 'end']
            gene_name = non_peak_region.loc[iter, 'gene_name']
            strand = non_peak_region.loc[iter, 'strand']

            random_posA = int(random.choice(posA))

            out_start = random_posA - (self.args.length // 2)
            out_end = random_posA + (self.args.length // 2)

            if out_start < non_peak_start:
                out_end = out_end + non_peak_start - out_start
                out_start = non_peak_start
            elif out_end > non_peak_end:
                out_start = out_start - (out_end - non_peak_end)
                out_end = non_peak_end

            dic = {'chr': chr, 'start': out_start, 'end': out_end, 'gene_name': gene_name, 'un': '.', 'strand': strand}

            neg_peak_list.append(dic)

        neg_peak_dataframe = pd.DataFrame(neg_peak_list)

        pos = pd.read_csv(f'{self.args.output}/03_sample_data/{self.args.species}_pos_sample_peak.bed', sep='\t', names=['chr', 'start', 'end', 'gene', 'un', 'strand', 'length', 'count'])
        pos_gene = set(pos['gene'])
        flag = []
        for iter in range(0, neg_peak_dataframe.shape[0]):
            if neg_peak_dataframe.loc[iter, 'gene_name'] in pos_gene:
                flag.append('a')
            else:
                flag.append('b')
        neg_peak_dataframe['type'] = flag
        pos2 = neg_peak_dataframe[neg_peak_dataframe.type == 'a'].reset_index(drop=True)
        tmp = neg_peak_dataframe[neg_peak_dataframe.type == 'b'].reset_index(drop=True)
        if pos_length > pos2.shape[0]:
            neg_peak = tmp.sample(n=pos_length - pos2.shape[0]).reset_index(drop=True)
            frame = [pos2, neg_peak]
            pd.concat(frame).to_csv(f'{self.args.output}/03_sample_data/{self.args.species}_neg_sample_peak.bed', sep='\t', index=None, header=None)
        else:
            neg_peak = pos2.sample(n=pos_length).reset_index(drop=True)
            neg_peak.to_csv(f'{self.args.output}/03_sample_data/{self.args.species}_neg_sample_peak.bed', sep='\t', index=None, header=None)
    def checkdir(self) -> None:
        """"Check output positive and negative directory."""
        self.output.info('Creating output directory.')

        raw_data = Path(self.args.output) / '01_raw_data'
        peak_data = Path(self.args.output) / '02_peak_data'
        sample_data = Path(self.args.output) / '03_sample_data'

        if not raw_data.is_dir():
            self.output.info('Creating output directory.')
            raw_data.mkdir()

        if not peak_data.is_dir():
            self.output.info(f'Creating output dircotry of {iter}.')
            peak_data.mkdir()

        if not sample_data.is_dir():
            self.output.info(f'Creating output processed directory of {iter}.')
            sample_data.mkdir()

    def get_fasta(self) -> None:
        
        fasta = pybedtools.example_filename(f'{self.args.ann_dic}/{self.args.species}.fa')
        
        self.output.info('Getting positive sample fasta data ...')
        peak = pybedtools.BedTool(f'{self.args.output}/03_sample_data/{self.args.species}_pos_sample_peak.bed')
        a = peak.sequence(fi=fasta,fo=f'{self.args.output}/03_sample_data/{self.args.species}_pos_sample_peak.fa_out', name=True, s=True)
        self.output.info('Finish getting positive data !!!')

        self.output.info('Getting negative sample fasta data ...')
        peak = pybedtools.BedTool(f'{self.args.output}/03_sample_data/{self.args.species}_neg_sample_peak.bed')
        a = peak.sequence(fi=fasta,fo=f'{self.args.output}/03_sample_data/{self.args.species}_neg_sample_peak.fa_out', name=True, s=True)
        self.output.info('Finish getting negative data !!!')
        

    def process(self) -> None:
        self.checkdir()

        self.output.info('Starting getting positive sample Process.')
        self.get_pos()

        self.output.info('Starting getting positive sample Process.')
        self.get_neg()

        self.get_fasta()
