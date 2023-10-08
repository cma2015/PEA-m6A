# -*- coding: utf-8 -*-
# Copyright 2021 Minggui Song.
# All rights reserved.

"""Represent a collect flnc information.

What's here:

Train positive and negative data sets.
-------------------------------------------

Classes:
    - Train
"""
from logging import getLogger
from statistics import mode
from src.sys_output import Output
import time
import itertools
import src.models
import src.weaknets
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
from pathlib import Path
from src.config import config
import random
import os
from catboost import CatBoostClassifier
#from tensorflow.keras.utils import plot_model
def setup_seed(seed):
    random.seed(seed)  # 为python设置随机种子
    np.random.seed(seed)  # 为numpy设置随机种子
    tf.random.set_seed(seed)  # tf cpu fix seed
    os.environ['TF_DETERMINISTIC_OPS'] = '1'  # tf gpu fix seed, please `pip install tensorflow-determinism` first

setup_seed(42)


tfk = tf.keras
tfkl = tf.keras.layers
tfdd = tf.data.Dataset
tfkc = tf.keras.callbacks

logger = getLogger(__name__)  # pylint: disable=invalid-name

class Train(object):
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

    def DL_data_provider(self) -> None:

        self.output.info('Loading data ...')

        input_sp_dir = Path(self.args.input_dir)
        train_data = np.load(input_sp_dir / 'train_data.npy', allow_pickle=True)
        valid_data = np.load(input_sp_dir / 'valid_data.npy', allow_pickle=True)
        test_data = np.load(input_sp_dir / 'test_data.npy', allow_pickle=True)

        train_label = np.load(input_sp_dir / 'train_label.npy', allow_pickle=True)
        valid_label = np.load(input_sp_dir / 'valid_label.npy', allow_pickle=True)
        test_label = np.load(input_sp_dir / 'test_label.npy', allow_pickle=True)


        train_dataset = tfdd.from_generator(lambda: itertools.zip_longest(train_data, train_label),
                                        output_types=(tf.float32, tf.int32),
                                        output_shapes=(tf.TensorShape([None, self.args.instance_length, 4]),
                                        tf.TensorShape([None])))
        valid_dataset = tfdd.from_generator(lambda: itertools.zip_longest(valid_data, valid_label),
                                        output_types=(tf.float32, tf.int32),
                                        output_shapes=(tf.TensorShape([None, self.args.instance_length, 4]),
                                        tf.TensorShape([None])))
        test_dataset = tfdd.from_generator(lambda: itertools.zip_longest(test_data, test_label),
                                       output_types=(tf.float32, tf.int32),
                                        output_shapes=(tf.TensorShape([None, self.args.instance_length, 4]),
                                        tf.TensorShape([None])))

        self.train_dataset = train_dataset.shuffle(8).batch(1)
        self.valid_dataset = valid_dataset.batch(1)
        self.test_dataset = test_dataset.batch(1)
        self.test_label = test_label

        self.output.info('Finish loading data !!!')

    def PEAm6A_data_provider(self) -> None:

        self.output.info('Loading data ...')
        input_sp_dir = Path(self.args.input_dir)
        features_set = set(self.args.matrix_name)

        self.train_data = np.array([])
        self.valid_data = np.array([])
        self.test_data = np.array([])

        if 'DL' in features_set:
            DL_train_data = np.load(input_sp_dir / 'DL_train_data.npy', allow_pickle=True)
            DL_valid_data = np.load(input_sp_dir / 'DL_valid_data.npy', allow_pickle=True)
            DL_test_data = np.load(input_sp_dir / 'DL_test_data.npy', allow_pickle=True)
            if not any(self.train_data):
                self.train_data = DL_train_data
                self.valid_data = DL_valid_data
                self.test_data = DL_test_data

        if 'ST' in features_set:
            ST_train_data = np.load(input_sp_dir / 'ST_train_data.npy', allow_pickle=True)
            ST_valid_data = np.load(input_sp_dir / 'ST_valid_data.npy', allow_pickle=True)
            ST_test_data = np.load(input_sp_dir / 'ST_test_data.npy', allow_pickle=True)
            if not any(self.train_data):
                self.train_data = ST_train_data
                self.valid_data = ST_valid_data
                self.test_data = ST_test_data
            else:
                self.train_data = np.vstack((self.train_data, ST_train_data))
                self.valid_data = np.vstack((self.valid_data, ST_valid_data))
                self.test_data = np.vstack((self.test_data, ST_test_data))
            
        if 'OT' in features_set:
            OT_train_data = np.load(input_sp_dir / 'OT_train_data.npy', allow_pickle=True)
            OT_valid_data = np.load(input_sp_dir / 'OT_valid_data.npy', allow_pickle=True)
            OT_test_data = np.load(input_sp_dir / 'OT_test_data.npy', allow_pickle=True)

            if not any(self.train_data):
                self.train_data = OT_train_data
                self.valid_data = OT_valid_data
                self.test_data = OT_test_data
            else:
                self.train_data = np.vstack((self.train_data, OT_train_data))
                self.valid_data = np.vstack((self.valid_data, OT_valid_data))
                self.test_data = np.vstack((self.test_data, OT_test_data))

        np.save(input_sp_dir / 'PEAm6A_train_data.npy', self.train_data)
        np.save(input_sp_dir / 'PEAm6A_valid_data.npy', self.valid_data)
        np.save(input_sp_dir / 'PEAm6A_test_data.npy', self.test_data)
        
        self.train_label = np.load(input_sp_dir / 'train_label.npy', allow_pickle=True)
        self.valid_label = np.load(input_sp_dir / 'valid_label.npy', allow_pickle=True)
        self.test_label = np.load(input_sp_dir / 'test_label.npy', allow_pickle=True)

    def DL_train_model(self) -> None:
        self.output.info('Start building model ...')
        model = src.models.SingleWeakRM()
        #model = src.models.SingleReWeakRM()
        self.output.info('Finish building model ...')

        opt_extractor = tf.keras.optimizers.Adam(learning_rate=self.args.lr_init, decay=self.args.lr_decay)
        opt_one = tf.keras.optimizers.Adam(learning_rate=self.args.lr_init, decay=self.args.lr_decay)

        train_loss = tf.keras.metrics.Mean()
        valid_loss = tf.keras.metrics.Mean()
        train_auROC = tf.keras.metrics.AUC()
        valid_auROC = tf.keras.metrics.AUC()

        train_step_signature = [
            tf.TensorSpec(shape=(1, None, self.args.instance_length, 4), dtype=tf.float32),
            tf.TensorSpec(shape=(1, 1), dtype=tf.int32),
        ]


        @tf.function(input_signature=train_step_signature)
        def train_step(train_seq, train_label):
            with tf.GradientTape(persistent=True) as tape:
                bag_features, _ = model.extractor(train_seq, training=True)
                out_prob = model.classifer(bag_features, training=True)
                ####需要修改----添加一部分
                loss = tfk.losses.BinaryCrossentropy(from_logits=False)(y_true=train_label, y_pred=out_prob)
                total_loss = loss + tf.reduce_sum(model.extractor.losses)
            gradients_extractor = tape.gradient(total_loss, model.extractor.trainable_variables)
            gradients_one = tape.gradient(total_loss, model.classifer.trainable_variables)
            opt_extractor.apply_gradients(zip(gradients_extractor, model.extractor.trainable_variables))
            opt_one.apply_gradients(zip(gradients_one, model.classifer.trainable_variables))
            train_loss(loss)
            train_auROC(y_true=train_label, y_pred=out_prob)

        @tf.function(input_signature=train_step_signature)
        def valid_step(valid_seq, valid_label):
            #inf_prob, _ = model(valid_seq, training=False)
            bag_features, _ = model.extractor(valid_seq, training=False)
            inf_prob = model.classifer(bag_features, training=False)
            #####同训练部分
            vloss = tf.keras.losses.BinaryCrossentropy()(y_true=valid_label, y_pred=inf_prob)
            valid_loss(vloss)
            valid_auROC(y_true=valid_label, y_pred=inf_prob)


        num_epoch = self.args.epoch
        current_monitor = np.inf
        patient_count = 0

        self.output.info('Start training model ...')
        for epoch in tf.range(1, num_epoch + 1):
            train_loss.reset_states()
            valid_loss.reset_states()

            train_auROC.reset_states()
            valid_auROC.reset_states()

            epoch_start_time = time.time()
            for tdata in self.train_dataset:
                train_step(tdata[0], tdata[1])
            print('Training of epoch {} finished! Time cost is {}s'.format(epoch, round(time.time() - epoch_start_time, 2)))

            valid_start_time = time.time()
            for vdata in self.valid_dataset:
                valid_step(vdata[0], vdata[1])

            new_valid_monitor = np.round(valid_loss.result().numpy(), 4)
            if new_valid_monitor < current_monitor:
                print('val_loss improved from {} to {}, saving model to {}'.
                        format(str(current_monitor), str(new_valid_monitor), self.args.checkpoint_directory))
                out_dir = Path(self.args.checkpoint_directory)
                features_name = 'features_' + self.args.checkpoint_name + '.h5'
                model.extractor.save_weights(out_dir / features_name)
                class_names = self.args.checkpoint_name + '.h5'
                model.classifer.save_weights(out_dir / class_names)
                current_monitor = new_valid_monitor
                patient_count = 0
            else:
                print('val_loss did not improved from {}'.format(str(current_monitor)))
                patient_count += 1

            if patient_count == 5:
                break


            template = "Epoch {}, Time Cost: {}s, TL: {}, TROC: {}, VL:{}, VROC: {}"
            print(template.format(epoch, str(round(time.time() - valid_start_time, 2)),
                                str(np.round(train_loss.result().numpy(), 4)),
                                str(np.round(train_auROC.result().numpy(), 4)),
                                str(np.round(valid_loss.result().numpy(), 4)),
                                str(np.round(valid_auROC.result().numpy(), 4)),
                                )
                )


        if config.eval_after_train:
            out_dir = Path(self.args.checkpoint_directory)
            features_name = 'features_' + self.args.checkpoint_name  +'.h5'
            model.extractor.load_weights(out_dir / features_name)
            class_names = self.args.checkpoint_name + '.h5'
            model.classifer.load_weights(out_dir / class_names)
            predictions = []
            for tdata in self.test_dataset:
                bag_features, _ = model.extractor(tdata[0], training=False)
                pred = model.classifer(bag_features)
                predictions.append(pred.numpy())
            predictions = np.concatenate(predictions, axis=0)

            auc = roc_auc_score(self.test_label, predictions)
            ap = average_precision_score(self.test_label, predictions)

            pos_num, neg_num = sum(self.test_label==1)[0], sum(self.test_label==0)[0]
            class_weight = {1: (pos_num+neg_num)/2/pos_num, 0: (pos_num+neg_num)/2/neg_num}

            predictions_lable = predictions
            predictions_lable[predictions_lable > 0.5] = 1
            predictions_lable[predictions_lable <= 0.5] = 0

            tn, fp, fn, tp = confusion_matrix(self.test_label, predictions_lable).ravel()
            tn, fp, fn, tp = tp * class_weight[1], fp * class_weight[0], fn * class_weight[1], tn * class_weight[0]

            acc = (tp + tn) / ( tn + fp + fn + tp )
            tpr = tp / ( tp + fn )
            tnr = tn / ( tn + fp )
            mcc = (tp*tn-fp*fn)/((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**0.5
            f1 = tp * 2 / ( tp * 2 + fp + fn )

            print(self.args.species_name + ' Test ACC: ', acc)
            print(self.args.species_name + ' Test TPR: ', tpr)
            print(self.args.species_name + ' Test TNR: ', tnr)
            print(self.args.species_name + ' Test MCC: ', mcc)
            print(self.args.species_name + ' Test F1:' , f1)
            print(self.args.species_name + ' Test AUC: ', auc)
            print(self.args.species_name + ' Test PRC: ', ap)

    def PEAm6A_train_model(self) -> None:
        cbc = CatBoostClassifier(auto_class_weights='Balanced')
        cbc.fit(self.train_data, self.train_label, eval_set=(self.valid_data, self.valid_label))
        out_dir = Path(self.args.checkpoint_directory)
        features_name = 'PEA-m6A' + self.args.checkpoint_name + '.h5'
        cbc.save_model(out_dir / features_name)

        if config.eval_after_train:
            out_dir = Path(self.args.checkpoint_directory)
            features_name = 'PEA-m6A' + self.args.checkpoint_name + '.h5'
            cbc = CatBoostClassifier()
            cbc.load_model(out_dir / features_name)
            cbc_pred_test_array = cbc.predict(self.test_data)
            cbc_pred_test_proba = cbc.predict_proba(self.test_data)

            tn, fp, fn, tp = confusion_matrix(self.test_label, cbc_pred_test_array).ravel()
            print(f'Accuracy:{(tp+tn)/(tn+fp+fn+tp)}\n')
            print(f'TPR(Recall):{tp/(tp+fn)}\n')
            print(f'TNR(Specificity):{tn/(tn+fp)}\n')
            print(f'MCC:{(tp*tn-fp*fn)/((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**0.5}\n')
            print(f'F1:{tp*2/(tp*2+fp+fn)}\n')
            print(f'ROC_AUC:{roc_auc_score(self.test_label, cbc_pred_test_proba[:,1])}\n')
            print(f'PRC:{average_precision_score(self.test_label, cbc_pred_test_proba[:,1])}\n')

    def process(self) -> None:
        self.output.info('Starting training data Process.')
        logger.debug('Starting training data Process.')

        if self.args.model_name == 'WeakRM':
            self.DL_data_provider()
            self.DL_train_model()
        else:
            self.PEAm6A_data_provider()
            self.PEAm6A_train_model()
        self.output.info('Completed training data Process.')
        logger.debug('Completed training data Process.')