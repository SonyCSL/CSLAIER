# -*- encoding: utf-8 -*-
import datetime
import re
import os
import signal
import shutil
import random
import zipfile
from logging import getLogger
import cPickle as pickle

from werkzeug import secure_filename

from db_models.shared_models import db
import common.utils as ds_utils
import deeplearning.visualizer as visualizer
import deeplearning.predict.imagenet_inspect as inspection
import deeplearning.predict.text_predict as lstm_prediction

logger = getLogger(__name__)

class Model(db.Model):
    id                    = db.Column(db.Integer, primary_key = True)
    name                  = db.Column(db.Text, unique = True, nullable = False)
    type                  = db.Column(db.Text)
    framework             = db.Column(db.Text)
    epoch                 = db.Column(db.Integer)
    network_name          = db.Column(db.Text)
    network_path          = db.Column(db.Text)
    trained_model_path    = db.Column(db.Text)
    prepared_file_path    = db.Column(db.Text)
    is_trained            = db.Column(db.Integer)
    pid                   = db.Column(db.Integer)
    resize_mode           = db.Column(db.Text)
    channels              = db.Column(db.Integer)
    use_wakatigaki        = db.Column(db.Integer)
    dataset_id            = db.Column(db.Integer, db.ForeignKey('dataset.id'))
    updated_at            = db.Column(db.DateTime)
    created_at            = db.Column(db.DateTime)

    def __init__(self, name, type, network_path, network_name, framework):
        self.name         = name
        self.type         = type
        self.network_path = network_path
        self.network_name = network_name
        self.framework    = framework
        self.epoch        = 1
        self.is_trained   = 0
        self.pid          = None
        self.channels     = 3
        self.updated_at   = datetime.datetime.now()
        self.created_at   = datetime.datetime.now()

    def __repr__(self):
        return

    @property
    def line_graph(self):
        return self.__get_file_path(self.trained_model_path, 'line_graph.tsv')

    @property
    def train_log(self):
        return self.__get_file_path(self.trained_model_path, 'log.html')

    @property
    def mean_file(self):
        return self.__get_file_path(self.prepared_file_path, 'mean.npy')

    @property
    def labels_text(self):
        return self.__get_file_path(self.prepared_file_path, 'labels.txt')

    @property
    def vocab_file(self):
        return self.__get_file_path(self.trained_model_path, 'vocab2.bin')

    def __get_file_path(self, path, filename):
        if path is None:
            return None
        full_path = os.path.join(path, filename)
        return full_path if os.path.exists(full_path) else None

    def enable_wakatigaki(self, value):
        self.use_wakatigaki = 1 if value else 0

    def get_use_wakatigaki_in_bool(self):
        return True if self.use_wakatigaki == 1 else False

    @classmethod
    def create_new(cls,
        name,             # name of network
        type,             # type of network
        network_file_dir, # path which network code is stored
        network_name,     # network name ex) nin, googlenet, lstm
        model_template,   # name of network template
        code,             # network code
        framework         # chainer or tensorflow
    ):
        if not re.match(r".+\.py", name):
            name += '.py'
        if network_name is None or network_name is '':
            if model_template is not None or model_template is not '':
                network_name = re.sub(r"\.py$", "", model_template)
            else:
                network_name = None
        framework = framework.lower()
        if framework != 'chainer' and framework != 'tensorflow':
            raise ValueError('Invalid framework type. "chainer" or "tensorflow" is allowed.')
        network_file_path = os.path.join(network_file_dir, name)
        network_file = open(network_file_path, "w")
        network_file.write(code)
        return cls(name, type, network_file_path, network_name, framework)

    @classmethod
    def get_model_with_code(cls, id):
        model = cls.query.get(id)
        model.code = open(model.network_path).read()
        if model.channels == 1:
            model.channels = "Grayscale"
        elif model.channels == 3:
            model.channels = "RGB"
        if model.resize_mode is None or model.resize_mode == '':
            model.resize_mode = '---'
        return model

    @classmethod
    def get_train_progresses(cls):
        models = cls.query.all()
        progresses = []
        for m in models:
            progresses.append({'id': m.id, 'is_trained': m.is_trained})
        return progresses

    def delete(self):
        db.session.delete(self)
        if self.prepared_file_path and os.path.exists(self.prepared_file_path):
            try:
                shutil.rmtree(self.prepared_file_path)
            except Exception as e:
                logger.exception('Could not remove {0}. {1}'.format(self.prepared_file_path, e))
                raise e
        if self.trained_model_path and os.path.exists(self.trained_model_path):
            try:
                shutil.rmtree(self.trained_model_path)
            except Exception as e:
                logger.exception('Could not remove {0}. {1}'.format(self.trained_model_path, e))
                raise e
        if os.path.exists(self.network_path):
            try:
                os.remove(self.network_path)
            except Exception as e:
                logger.exception('Could not remove {0}. {1}'.format(self.network_path, e))
                raise e
        # remove .pyc file
        if os.path.exists(self.network_path + 'c'):
            try:
                os.remove(self.network_path + 'c')
            except Exception as e:
                logger.exception('Could not remove {0}c. {1}'.format(self.network_path, e))
                raise e
        db.session.commit()

    def get_pretrained_models(self):
        pretrained_models = ["New"]
        if self.trained_model_path:
            candidate = sorted(os.listdir(self.trained_model_path), reverse=True)
            if pretrained_models:
                pretrained_models = filter(lambda file:file.find('model')>-1, candidate)
                pretrained_models.append("New")
        return pretrained_models

    def update_and_commit(self):
        self.updated_at = datetime.datetime.now()
        db.session.add(self)
        db.session.commit()

    def get_layers(self, epoch):
        v = self.__get_visualizer(epoch)
        return v.get_layer_list()

    def get_visualized_layer(self, epoch, layer):
        if os.path.exists(os.path.join(self.trained_model_path, str(epoch), layer + '.png')):
            return {'filename': layer + '.png', 'epoch': epoch}
        if not os.path.exists(os.path.join(self.trained_model_path, str(epoch))):
            os.mkdir(os.path.join(self.trained_model_path, str(epoch)))
        v = self.__get_visualizer(epoch)
        layer = layer.replace('_', '/')
        filename = v.visualize(layer)
        if filename is None:
            raise Exception('Could not generate layer visualization.')
        return {'filename': filename, 'epoch': epoch}

    def inspect(self, epoch, uploaded, save_to):
        name, ext = os.path.splitext(uploaded.filename)
        ext = ext.lower()
        if ext not in ('.jpg', '.jpeg', '.gif', '.png'):
            raise Exception('File extension not allowed.')
        new_filename = os.path.join(save_to, ds_utils.get_timestamp() + '_' + secure_filename(uploaded.filename))
        uploaded.save(new_filename)
        results = inspection.inspect(
            new_filename,
            self.mean_file,
            self.get_trained_model(epoch),
            self.labels_text,
            self.network_path,
            self.resize_mode,
            self.channels,
            gpu=-1
        )
        return results, new_filename

    def lstm_predict(self, epoch, primetext, result_length):
        seed = int(random.random() + 10000)
        result = lstm_prediction.predict(
            self.get_trained_model(epoch),
            self.vocab_file,
            self.network_path,
            primetext,
            seed,
            128,  # unit
            0.0,  # dropout
            1,    # sample
            result_length,
            use_mecab = self.get_use_wakatigaki_in_bool()
        )
        return result.replace('<eos>', '\n')

    def get_trained_model(self, epoch):
        return self.__get_file_path(self.trained_model_path, "model{0:0>4}".format(epoch))

    def get_trained_files(self, epoch, root_out_dir):
        zipfile_path = os.path.join(root_out_dir, ds_utils.get_timestamp() + '_' + re.sub(r"\.py$", "", self.name) + '.zip')
        with zipfile.ZipFile(zipfile_path, 'w', zipfile.ZIP_DEFLATED) as f:
            f.write(self.network_path, os.path.basename(self.network_path))
            trained_model_path = self.get_trained_model(epoch)
            f.write(trained_model_path, os.path.basename(trained_model_path))
            if self.type == 'image':
                f.write(self.labels_text, os.path.basename(self.labels_text))
                f.write(self.mean_file, os.path.basename(self.mean_file))
            elif self.type=='text':
                f.write(self.vocab_file, os.path.basename(self.vocab_file))
        return zipfile_path

    def terminate_train(self):
        if self.pid is not None:
            try:
                os.kill(self.pid, signal.SIGTERM)
                logger.info('Process successfully terminated.')
            except OSError as e:
                logger.info("Process already terminated. ERROR NO: {0} - {1}".format(e.errno, e.strerror))
            self.is_trained = 0
            self.pid = None
            for f in os.listdir(self.trained_model_path):
                if f.startswith('previous_'):
                    try:
                        shutil.copyfile(os.path.join(self.trained_model_path, f), os.path.join(self.trained_model_path, f.replace('previous_', '')))
                        os.remove(os.path.join(self.trained_model_path, f))
                    except Exception as e:
                        logger.exception('Failed to restore backuped trained model. {0} {1}'.format(os.path.join(self.trained_model_path, f), e))
                        raise e
            self.update_and_commit()

    def __get_visualizer(self, epoch):
        if epoch > self.epoch:
            raise Exception('Selected epoch is bigger than trained epoch')
        trained_model_path = self.get_trained_model(epoch)
        if not os.path.exists(trained_model_path):
            raise Exception('Could not find the trained model')
        if self.type == 'image':
            return visualizer.LayerVisualizer(
                self.network_path,
                trained_model_path,
                os.path.join(self.trained_model_path, str(epoch))
            )
        elif self.type == 'text':
            vocab = pickle.load(open(os.path.join(self.trained_model_path, 'vocab2.bin')), 'rb')
            return visualizer.LayerVisualizer(
                self.network_path,
                trained_model_path,
                os.path.join(self.trained_model_path, str(epoch)),
                vocab_len = len(vocab),
                n_units=128,
                dropout=0.5
            )
        else:
            raise Exception('Unknown Model.type.')
