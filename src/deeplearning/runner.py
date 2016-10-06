# -*- encoding:utf-8 -*-
import os
import logging
from multiprocessing import Process, Event

from db_models.datasets import Dataset
from db_models.models import Model
import deeplearning.prepare.prepare_for_imagenet
import deeplearning.prepare.prepare_for_lstm
import deeplearning.train.train_imagenet
import deeplearning.train.train_lstm
from deeplearning.log_subscriber import train_logger
from time import sleep
import gevent
import re


logger = logging.getLogger(__name__)

INTERRUPTABLE_PROCESSES = {}


class Interruptable(object):
    def __init__(self):
        super(Interruptable, self).__init__()
        self.interrupt_event = Event()
        self.interruptable_event = Event()
        self.end_event = Event()
        self.completion = None

        def wait_for_end():
            while True:
                if self.end_event.is_set():
                    break
                gevent.sleep(1)
            if self.completion:
                self.completion()

        gevent.spawn(wait_for_end)

    def terminate(self):
        self.end_event.set()

    def set_interrupt(self):
        self.interrupt_event.set()

    def clear_interrupt(self):
        self.interrupt_event.set()

    def set_interruptable(self):
        self.interruptable_event.set()

    def clear_interruptable(self):
        self.interruptable_event.set()

    def is_interrupting(self):
        return self.interrupt_event.is_set()

    def is_interruptable(self):
        return self.interruptable_event.is_set()


def _create_trained_model_dir(path, root_output_dir, model_name):
    if path is None:
        path = os.path.join(root_output_dir, model_name)
    if not os.path.exists(path):
        os.mkdir(path)
    return path


# 学習の後片付け
def _cleanup_for_train_terminate(model_id):
    print('_cleanup_for_train_terminate')
    train_logger.terminate_train(model_id)
    del INTERRUPTABLE_PROCESSES[model_id]


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
        avoid_flipping,
        batchsize
):
    dataset = Dataset.query.get(dataset_id)
    model = Model.query.get(model_id)
    model.dataset = dataset
    model.epoch = epoch
    model.resize_mode = resize_mode
    model.channels = channels
    model.gpu = gpu_num
    model.batchsize = batchsize
    model.update_and_commit()
    model, train_image_num, val_image_num = deeplearning.prepare.prepare_for_imagenet.do(model, prepared_data_root)

    (model_dir, model_name) = os.path.split(model.network_path)
    model_name = re.sub(r"\.py$", "", model_name)
    trained_model_path = _create_trained_model_dir(model.trained_model_path,
                                                   output_dir_root, model_name)
    model.trained_model_path = trained_model_path
    train_log = os.path.join(trained_model_path, 'train.log')
    open(train_log, 'w').close()
    train_logger.file_subscribe(model_id, train_log)
    interruptable = Interruptable()
    if model.framework == 'chainer':
        train_process = Process(
            target=deeplearning.train.train_imagenet.do_train_by_chainer,
            args=(
                model,
                output_dir_root,
                250,  # val_batchsize
                20,  # loader_job
                pretrained_model,
                avoid_flipping,
                interruptable,
            )
        )
    elif model.framework == 'tensorflow':
        train_process = Process(
            target=deeplearning.train.train_imagenet.do_train_by_tensorflow,
            args=(
                model,
                output_dir_root,
                500,  # val_batchsize
                pretrained_model,
                train_image_num,
                val_image_num,
                avoid_flipping,
                False,  # resume
                interruptable,
            )
        )
    else:
        raise Exception('Unknown framework')
    train_process.start()
    model.pid = train_process.pid
    model.update_and_commit()
    logging.info('start imagenet training. PID: ', model.pid)

    def completion():
        _cleanup_for_train_terminate(model.id)

    interruptable.completion = completion

    INTERRUPTABLE_PROCESSES[model.id] = interruptable


# 再現に必要な情報はモデルと、稼働させるGPUだけのはず。
def resume_imagenet_train(output_dir_root, model, gpu_num):
    model.gpu = gpu_num
    model.update_and_commit()
    train_logger.file_subscribe(model.id, model.train_log_path)
    interruptable = Interruptable()
    if model.framework == 'chainer':
        train_process = Process(
            target=deeplearning.train.train_imagenet.resume_train_by_chainer,
            args=(
                model,
                output_dir_root,
                250,  # val_batchsize
                20,  # loader_job
                interruptable,
            )
        )
    elif model.framework == 'tensorflow':
        train_image_num, val_image_num = deeplearning.prepare.prepare_for_imagenet.get_image_num(
            model.dataset.dataset_path)
        train_process = Process(
            target=deeplearning.train.train_imagenet.do_train_by_tensorflow,
            args=(
                model,
                output_dir_root,
                500,  # val_batchsize
                None,
                train_image_num,
                val_image_num,
                False,  # not used
                True,  # resume
                interruptable,
            )
        )
    else:
        raise Exception('Unknown framework')
    train_process.start()
    model.pid = train_process.pid
    model.update_and_commit()
    logging.info('start imagenet training. PID: ', model.pid)

    def completion():
        _cleanup_for_train_terminate(model.id)

    interruptable.completion = completion

    INTERRUPTABLE_PROCESSES[model.id] = interruptable


def run_lstm_train(
        prepared_data_root,
        output_dir_root,
        dataset_id,
        model_id,
        epoch,
        pretrained_model,
        gpu_num,
        use_wakatigaki,
        batchsize=50
):
    dataset = Dataset.query.get(dataset_id)
    model = Model.query.get(model_id)
    model.dataset = dataset
    model.epoch = epoch
    model.gpu = gpu_num
    model.enable_wakatigaki(use_wakatigaki)
    model.batchsize = batchsize
    (input_data_path, pretrained_vocab, model) = deeplearning.prepare.prepare_for_lstm.do(
        model, prepared_data_root, pretrained_model, use_wakatigaki)
    interruptable = Interruptable()
    train_process = Process(
        target=deeplearning.train.train_lstm.do_train,
        args=(
            model,
            output_dir_root,
            input_data_path,
            pretrained_vocab,
            use_wakatigaki,
            pretrained_model,
            None,  # resume
            128,  # runsize
            2e-3,  # learning_rate
            0.97,  # learning_rate_decay
            10,  # learning_rate_decay_after
            0.95,  # decay_rate
            0.0,  # dropout
            50,  # seq_length
            batchsize,  # batchsize
            5,  # grad_clip
            interruptable,
        )
    )
    train_process.start()
    model.pid = train_process.pid
    model.update_and_commit()
    logging.info('start LSTM training. PID: ', model.pid)

    def completion():
        _cleanup_for_train_terminate(model.id)

    interruptable.completion = completion

    INTERRUPTABLE_PROCESSES[model.id] = interruptable


def resume_lstm_train(
        prepared_data_root,
        output_dir_root,
        model,
        gpu_num,
):
    model.gpu = gpu_num
    (input_data_path, pretrained_vocab, model) = deeplearning.prepare.prepare_for_lstm.do(
        model, prepared_data_root, None, model.use_wakatigaki)
    interruptable = Interruptable()
    train_process = Process(
        target=deeplearning.train.train_lstm.do_train,
        args=(
            model,
            output_dir_root,
            input_data_path,
            pretrained_vocab,
            model.use_wakatigaki,
            '',
            True,  # resume
            128,  # runsize
            2e-3,  # learning_rate
            0.97,  # learning_rate_decay
            10,  # learning_rate_decay_after
            0.95,  # decay_rate
            0.0,  # dropout
            50,  # seq_length
            model.batchsize,  # batchsize
            5,  # grad_clip
            interruptable,
        )
    )
    train_process.start()
    model.pid = train_process.pid
    model.update_and_commit()
    logging.info('start LSTM training. PID: ', model.pid)

    def completion():
        _cleanup_for_train_terminate(model.id)

    interruptable.completion = completion
    INTERRUPTABLE_PROCESSES[model.id] = interruptable
