# -*- encoding:utf-8 -*-
import os
import logging
from multiprocessing import Process

from db_models.datasets import Dataset
from db_models.models import Model
import deeplearning.prepare.prepare_for_imagenet
import deeplearning.prepare.prepare_for_lstm
import deeplearning.train.train_imagenet
import deeplearning.train.train_lstm

logger = logging.getLogger(__name__)


def run_imagenet_train(
    prepared_data_root,
    output_dir_root,
    dataset_id,
    model_id,
    epoch,
    pretrained_model,
    gpu_num,
    resize_mode,
    channels,
    avoid_flipping
):
    dataset = Dataset.query.get(dataset_id)
    model = Model.query.get(model_id)
    model.dataset = dataset
    model.epoch = epoch
    model.resize_mode = resize_mode
    model.channels = channels
    model.avoid_flipping = avoid_flipping
    model.gpu = gpu_num
    model, train_image_num = deeplearning.prepare.prepare_for_imagenet.do(model, prepared_data_root)
    if model.framework == 'chainer':
        train_process = Process(
            target=deeplearning.train.train_imagenet.do_train_by_chainer,
            args=(
                model,
                output_dir_root,
                32,   # bachsize
                250,  # val_batchsize
                20,   # loader_job
                pretrained_model,
            )
        )
    elif model.framework == 'tensorflow':
        train_process = Process(
            target=deeplearning.train.train_imagenet.do_train_by_tensorflow,
            args=(
                model,
                output_dir_root,
                256,   # batchsize
                250,  # val_batchsize
                pretrained_model,
                train_image_num
            )
        )
    else:
        raise Exception('Unknown framework')
    train_process.start()
    model.pid = train_process.pid
    model.update_and_commit()
    logging.info('start imagenet training. PID: ', model.pid)
    return


def run_lstm_train(
    prepared_data_root,
    output_dir_root,
    dataset_id,
    model_id,
    epoch,
    pretrained_model,
    gpu_num,
    use_wakatigaki
):
    dataset = Dataset.query.get(dataset_id)
    model = Model.query.get(model_id)
    model.dataset = dataset
    model.epoch = epoch
    model.gpu = gpu_num
    model.enable_wakatigaki(use_wakatigaki)
    (input_data_path, pretrained_vocab, model) = deeplearning.prepare.prepare_for_lstm.do(
        model, prepared_data_root, pretrained_model, use_wakatigaki)
    train_process = Process(
        target=deeplearning.train.train_lstm.do_train,
        args=(
            model,
            output_dir_root,
            input_data_path,
            pretrained_vocab,
            use_wakatigaki,
            pretrained_model,
            None,    # resume
            gpu_num,
            128,     # runsize
            2e-3,    # learning_rate
            0.97,    # learning_rate_decay
            10,      # learning_rate_decay_after
            0.95,    # decay_rate
            0.0,     # dropout
            50,      # seq_length
            50,      # batchsize
            5        # grad_clip
        )
    )
    train_process.start()
    model.pid = train_process.pid
    model.update_and_commit()
    logging.info('start LSTM training. PID: ', model.pid)
