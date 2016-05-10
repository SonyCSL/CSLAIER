# -*- encoding:utf-8 -*-
import os
from logging import getLogger
import nkf

import common.utils as ds_utils

logger = getLogger(__name__)

def do(
    model,
    prepared_data_root,
    pretrained_model,
    use_wakatigaki
):
    logger.info('Start making LSTM training data.')
    if model.prepared_file_path:
        # re-use existing directory
        for f in os.listdir(model.prepared_file_path):
            os.remove(os.path.join(model.prepared_file_path, f))
    else:
        model.prepared_file_path = os.path.join(prepared_data_root, ds_utils.get_timestamp())
        os.mkdir(model.prepared_file_path)
    if pretrained_model != "-1":
        trained_model_path = model.trained_model_path
        if trained_model_path:
            pretrained_vocab = os.path.join(trained_model_path, 'vocab2.bin')
            if not os.path.exists(pretrained_vocab):
                logger.error("Could not find vocab2.bin file. It is possible that previsou train have failed: {0}".format(pretrained_vocab))
                raise Exception("Could not find vocab2.bin file. It is possible that previsou train have failed: ", pretrained_vocab)
        else:
            pretrained_vocab = ''
    else:
        pretrained_vocab = ''
    input_data_path = make_train_text(model, use_wakatigaki)
    model.update_and_commit()
    logger.info('Finish making LSTM training data.')
    return input_data_path, pretrained_vocab, model

def make_train_text(model, use_wakatigaki):
    input_text = open(os.path.join(model.prepared_file_path, 'input.txt'), 'w')
    if use_wakatigaki:
        logger.info('Use wakatigaki option.')
        import MeCab
        none = None
        m = MeCab.Tagger("-Owakati")
        for f in ds_utils.find_all_files(model.dataset.dataset_path):
            raw_text = open(f, 'r').read()
            encoding = nkf.guess(raw_text)
            if encoding == 'BINARY':
                continue
            text = raw_text.decode(encoding, 'ignore')
            text = text.replace('\r', '')
            encoded_text = text.encode('UTF-8')
            lines = encoded_text.splitlines()
            for line in lines:
                result = m.parse(line)
                if isinstance(none, type(result)):
                    continue
                input_text.write(result)
                input_text.flush()
    else:
        for f in ds_utils.find_all_files(model.dataset.dataset_path):
            temp_text = open(f, 'r').read()
            encoding = nkf.guess(temp_text)
            decoded_text = temp_text.decode(encoding, 'ignore')
            decoded_text = decoded_text.replace('\r', '')
            encoded_text = decoded_text.encode('UTF-8')
            input_text.write(encoded_text)
            input_text.flush()
    input_text.close()
    return os.path.join(model.prepared_file_path, 'input.txt')