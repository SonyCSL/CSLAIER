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

logger = logging.getLogger(__name__)

INTERRUPTABLE_PROCESSES = {}


class Interruptable(object):
    def __init__(self, interrupt_event, interruptable_event):
        super(Interruptable, self).__init__()
        self.interrupt_event = interrupt_event
        self.interruptable_event = interruptable_event

    def interrupt(self):
        self.interrupt_event.set()

    def interrupting(self):
        return self.interrupt_event.is_set()

    def interruptable(self):
        return self.interruptable_event.is_set()


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
    interrupt_event = Event()
    interruptable_event = Event()
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
                interrupt_event,
                interruptable_event
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
                interrupt_event,
                interruptable_event
            )
        )
    else:
        raise Exception('Unknown framework')
    train_process.start()
    model.pid = train_process.pid
    model.update_and_commit()
    logging.info('start imagenet training. PID: ', model.pid)
    INTERRUPTABLE_PROCESSES[model.pid] = Interruptable(interrupt_event, interruptable_event)
    return


# 再現に必要な情報はモデルと、稼働させるGPUだけのはず。
def resume_imagenet_train(output_dir_root, model, gpu_num):
    interrupt_event = Event()
    interruptable_event = Event()
    model.gpu = gpu_num
    model.update_and_commit()
    if model.framework == 'chainer':
        train_process = Process(
            target=deeplearning.train.train_imagenet.resume_train_by_chainer,
            args=(
                model,
                output_dir_root,
                250,  # val_batchsize
                20,  # loader_job
                interrupt_event,
                interruptable_event
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
                interrupt_event,
                interruptable_event
            )
        )
    else:
        raise Exception('Unknown framework')
    train_process.start()
    model.pid = train_process.pid
    model.update_and_commit()
    logging.info('start imagenet training. PID: ', model.pid)
    INTERRUPTABLE_PROCESSES[model.pid] = Interruptable(interrupt_event, interruptable_event)
    return


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
    interrupt_event = Event()
    interruptable_event = Event()
    dataset = Dataset.query.get(dataset_id)
    model = Model.query.get(model_id)
    model.dataset = dataset
    model.epoch = epoch
    model.gpu = gpu_num
    model.enable_wakatigaki(use_wakatigaki)
    model.batchsize = batchsize
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
            interrupt_event,
            interruptable_event
        )
    )
    train_process.start()
    model.pid = train_process.pid
    model.update_and_commit()
    logging.info('start LSTM training. PID: ', model.pid)
    INTERRUPTABLE_PROCESSES[model.pid] = Interruptable(interrupt_event, interruptable_event)


def resume_lstm_train(
        prepared_data_root,
        output_dir_root,
        model,
        gpu_num,
):
    interrupt_event = Event()
    interruptable_event = Event()
    model.gpu = gpu_num
    (input_data_path, pretrained_vocab, model) = deeplearning.prepare.prepare_for_lstm.do(
        model, prepared_data_root, None, model.use_wakatigaki)
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
            interrupt_event,
            interruptable_event
        )
    )
    train_process.start()
    model.pid = train_process.pid
    model.update_and_commit()
    logging.info('start LSTM training. PID: ', model.pid)
    INTERRUPTABLE_PROCESSES[model.pid] = Interruptable(interrupt_event, interruptable_event)
