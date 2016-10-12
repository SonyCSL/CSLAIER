# -*- coding: utf-8 -*-
import json
import os
import re
import logging
from logging.handlers import RotatingFileHandler
from logging import getLogger
from time import sleep

from flask import Flask, url_for, render_template, request, redirect, \
    jsonify, send_from_directory, send_file, Response
import gevent
from gevent.wsgi import WSGIServer
from gevent.queue import Queue
from werkzeug.contrib.cache import SimpleCache
from sqlalchemy import desc
from sqlalchemy.orm import eagerload

from db_models.shared_models import db
from db_models.datasets import Dataset
from db_models.models import Model
import deeplearning.runner as runner
from deeplearning.log_subscriber import train_logger
import common.utils as ds_util
from common import strings
from gevent.wsgi import WSGIServer

__version__ = '0.6.1'

app = Flask(__name__)
app.config.from_envvar('DEEPSTATION_CONFIG')
deepstation_config_params = ('DATABASE_PATH', 'UPLOADED_RAW_FILE',
                             'UPLOADED_FILE', 'PREPARED_DATA', 'TRAINED_DATA',
                             'INSPECTION_TEMP', 'LOG_DIR')
# WebApp settings
app.config['DEEPSTATION_ROOT'] = os.getcwd()


def normalize_config_path():
    for param in deepstation_config_params:
        if not app.config[param].startswith('/'):
            app.config[param] = os.path.abspath(app.config['DEEPSTATION_ROOT']
                                                + os.sep + app.config[param])


normalize_config_path()

# Logging settings
formatter = logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s '
    '[in %(pathname)s:%(lineno)d]'
)
debug_log = os.path.join(app.config['LOG_DIR'], 'debug.log')
error_log = os.path.join(app.config['LOG_DIR'], 'error.log')

debug_file_handler = RotatingFileHandler(
    debug_log, maxBytes=100000000, backupCount=10
)
debug_file_handler.setLevel(logging.INFO)
debug_file_handler.setFormatter(formatter)

error_file_handler = RotatingFileHandler(
    error_log, maxBytes=100000000, backupCount=10
)
error_file_handler.setLevel(logging.ERROR)
error_file_handler.setFormatter(formatter)

loggers = (app.logger, getLogger('db_models.datasets'),
           getLogger('db_models.models'),
           getLogger('deeplearning.runner'),
           getLogger('deeplearning.prepare.prepare_for_imagenet'),
           getLogger('deeplearning.prepare.prepare_for_lstm'),
           getLogger('deeplearning.train.train_lstm'),
           getLogger('deeplearning.train.train_imagenet'))
for logger in loggers:
    logger.setLevel(logging.INFO)
    logger.addHandler(debug_file_handler)
    logger.addHandler(error_file_handler)

# Database settings
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = app.config['DEBUG']  # DEBUG用設定
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + app.config['DATABASE_PATH']
db.init_app(app)

cache = SimpleCache()


@app.route('/')
def index():
    datasets, dataset_count = Dataset.get_datasets_with_samples(3, 0)
    models = Model.query.options(eagerload('dataset')).order_by(desc(Model.updated_at))
    return render_template(
        'index.html',
        system_info=get_system_info(),
        datasets=datasets,
        dataset_count=dataset_count,
        models=models
    )


@app.route('/files/<int:dataset_id>/<path:image_path>')
def show_dataset_image(dataset_id, image_path):
    ds_path = cache.get('dataset_id_' + str(dataset_id))
    if ds_path is None:
        dataset = Dataset.query.get(dataset_id)
        ds_path = dataset.dataset_path
        cache.set('dataset_id_' + str(dataset_id), ds_path, timeout=24 * 60 * 60)
    return send_from_directory(ds_path, image_path)


@app.route('/layers/<int:id>/<int:epoch>/<string:filename>')
def show_visualized_layer(id, epoch, filename):
    model = Model.query.get(id)
    return send_from_directory(os.path.join(model.trained_model_path, str(epoch)), filename)


@app.route('/inspection/<string:filename>')
def show_inspection_uploaded_file(filename):
    return send_from_directory(app.config['INSPECTION_TEMP'], filename)


@app.route('/dataset/show/<int:id>/')
def show_dataset(id):
    page = request.args.get('page', type=int, default=1)
    ds = Dataset.query.get(id)
    dataset = ds.get_dataset_with_categories_and_samples(offset=(page - 1) * 20)
    return render_template('dataset/show_dataset.html',
                           dataset=dataset, current_page=page)


@app.route('/dataset/show/<int:id>/<path:category>')
def show_dataset_category(id, category):
    page = request.args.get('page', type=int, default=1)
    if category == '-':
        category = ''
    ds = Dataset.query.get(id)
    dataset = ds.get_dataset_with_category_detail(category, offset=(page - 1) * 100)
    return render_template('dataset/show_category_detail.html',
                           dataset=dataset, current_page=page)


@app.route('/dataset/remove/<int:id>')
def remove_dataset(id):
    dataset = Dataset.query.get(id)
    dataset.delete()
    return redirect(url_for('index'))


@app.route('/dataset/remove/<int:id>/category/', methods=['POST'])
def remove_category(id):
    category_path = request.form['category_path']
    if category_path != '/':
        Dataset.remove_category(id, category_path)
    return redirect(url_for('show_dataset', id=id))


@app.route('/dataset/<int:id>/create/category/', methods=['POST'])
def create_category(id):
    category_name = request.form['category_name']
    Dataset.create_category(id, category_name)
    return jsonify({'status': 'success'})


@app.route('/dataset/<int:id>/upload/<path:category_path>', methods=['POST'])
def upload_file_to_category(id, category_path):
    uploaded_file = request.files['fileInput']
    dataset = Dataset.query.get(id)
    dataset.save_uploaded_file_to_category(uploaded_file, category_path)
    return redirect(url_for('show_dataset_category',
                            id=id, category=category_path))


@app.route('/dataset/<int:id>/remove/file/<path:category_path>', methods=['POST'])
def remove_file_from_category(id, category_path):
    dataset = Dataset.query.get(id)
    dataset.remove_file_from_category(request.form['file_path'])
    return redirect(url_for('show_dataset_category',
                            id=id, category=category_path))


@app.route('/models/new', methods=['GET', 'POST'])
def create_new_model():
    if request.method == 'GET':
        model_templates = os.listdir(
            os.path.join(app.config['DEEPSTATION_ROOT'], 'src', 'model_templates')
        )
        if ds_util.get_tensorflow_version() == '---':
            for t in model_templates:
                if re.search(r'_tf\.py', t):
                    model_templates.remove(t)
        return render_template('model/new.html', templates=model_templates)
    # POST
    model_name = request.form['model_name'].strip()
    my_network = request.form['my_network']
    model_template = request.form['model_template']
    network_name = request.form['network_type'].strip()
    model_type = request.form['model_type']
    framework = request.form['framework']
    model = Model.create_new(
        model_name,
        model_type,
        os.path.join(app.config['DEEPSTATION_ROOT'], 'src', 'models'),
        network_name,
        model_template,
        my_network,
        framework
    )
    db.session.add(model)
    db.session.commit()
    return redirect(url_for('show_model', id=model.id))


@app.route('/models/show/<int:id>')
def show_model(id):
    model = Model.get_model_with_code(id)
    datasets = Dataset.query.filter_by(type=model.type)
    resumable = False
    trained_epoch = 0
    if model.trained_model_path:
        resume_path = os.path.join(model.trained_model_path, 'resume')
        resumable = os.path.exists(resume_path)
        if resumable:
            data = json.load(open(os.path.join(resume_path, 'resume.json')))
            trained_epoch = data.get('epoch', 0)
    return render_template('model/show.html',
                           model=model, datasets=datasets,
                           pretrained_models=model.get_pretrained_models(),
                           mecab_available=ds_util.is_module_available('Mecab'),
                           system_info=get_system_info(),
                           usable_epochs=model.get_usable_epochs(),
                           resumable=resumable, trained_epoch=trained_epoch)


@app.route('/models/inspect/', methods=['POST'])
def inspect_image():
    id = request.form['model_id']
    epoch = request.form['epoch']
    uploaded = request.files['fileInput']
    model = Model.query.get(id)
    try:
        results, image_path = model.inspect(
            int(epoch), uploaded, app.config['INSPECTION_TEMP'])
        image_path = image_path.replace(app.config['INSPECTION_TEMP'], '')
        if image_path.startswith('/'):
            image_path = image_path.replace('/', '')
        return render_template('model/inspect_result.html',
                               results=results, model=model,
                               epoch=epoch, image=image_path)
    except IOError:
        return render_template('model/inspect_result.html',
                               error=strings.EPOCH_FILE_UNDER_TRAINING_ERROR,
                               model=model, epoch=epoch)


@app.route('/admin/')
def admin_index():
    return render_template('admin/index.html')


@app.route('/admin/models/')
def admin_maintain_models():
    models = Model.query.all()
    return render_template('admin/models.html', models=models)


@app.route('/admin/models/remove/', methods=['POST'])
def admin_remove_model():
    id = int(request.form['model_id'])
    model = Model.query.get(id)
    model.delete()
    return redirect(url_for('admin_maintain_models'))


@app.route('/admin/datasets/')
def admin_maintain_datasets():
    datasets = Dataset.query.all()
    return render_template('admin/datasets.html', datasets=datasets)


@app.route('/admin/datasets/remove/<int:id>')
def admin_remove_dataset(id):
    dataset = Dataset.query.get(id)
    dataset.delete()
    return redirect(url_for('admin_maintain_datasets'))


@app.route('/admin/datasets/update/', methods=['POST'])
def admin_update_dataset_path():
    id = int(request.form['dataset_id'])
    new_path = request.form['new_path']
    dataset = Dataset.query.get(id)
    try:
        dataset.update_dataset_path(new_path)
    except Exception as e:
        return 'Could no update dataset_path: {}'.format(e)
    return redirect(url_for('admin_maintain_datasets'))


# =====================================================================
# API
# =====================================================================


@app.route('/api/dataset/get/<int:offset>/')
def api_get_dataset(offset):
    datasets, dataset_count = Dataset.get_datasets_with_samples(offset=offset, limit=3)
    ret_ds = []
    for dataset in datasets:
        ds = {
            'id': dataset.id,
            'name': dataset.name,
            'type': dataset.type,
        }
        if dataset.type == 'image':
            ds['category_num'] = dataset.category_num
            ds['file_num'] = dataset.file_num
            thumbs = []
            for t in dataset.thumbnails:
                thumbs.append(t)
            ds['thumbnails'] = thumbs
        elif dataset.type == 'text':
            ds['filesize'] = dataset.filesize
            texts = []
            for t in dataset.sample_text:
                texts.append(t)
            ds['sample_text'] = texts
        ret_ds.append(ds)
    return jsonify({'dataset_count': dataset_count, 'datasets': ret_ds})


@app.route('/api/dataset/upload', methods=['POST'])
def api_upload_dataset():
    uploaded_file = request.files['fileInput']
    dataset_name = request.form['dataset_name']
    dataset_type = request.form['dataset_type']
    dataset = Dataset(dataset_name, dataset_type)
    try:
        dataset.save_uploaded_data(uploaded_file,
                                   app.config['UPLOADED_RAW_FILE'],
                                   app.config['UPLOADED_FILE'])
    except ValueError as e:
        app.logger.exception('Error occurred, when uploading dataset. {0}'.format(e))
        return jsonify({
            'status': 'error',
            'message': e.message
        })
    except Exception as e:
        app.logger.exception('Error occurred, when uploading dataset. {0}'.format(e))
        return jsonify({
            'status': 'error',
            'message': e.message
        })
    db.session.add(dataset)
    db.session.commit()
    return jsonify({
        'status': 'success'
    })


@app.route('/api/dataset/set_path', methods=['POST'])
def api_dataset_register_by_path():
    given_path = request.form['dataset_path']
    path = None
    # check path existence
    if given_path.startswith('/'):
        # absolute path
        if os.path.exists(given_path) and os.path.isdir(given_path):
            path = os.path.normpath(given_path)
    else:
        # relative path
        abs_path = os.path.normpath(os.path.join(app.config['DEEPSTATION_ROOT'],
                                                 given_path))
        if os.path.exists(abs_path) and os.path.isdir(abs_path):
            path = abs_path
    if path is None:
        app.logger.error('Path does not exists: {0}'.format(given_path))
        return jsonify({
            'status': 'error',
            'message': 'There is no such directory',
            'path': given_path
        })
    # register to db
    name = request.form['dataset_name']
    type = request.form['dataset_type']
    dataset = Dataset(name, type, path)
    dataset.category_num = ds_util.count_categories(path)
    dataset.file_num = ds_util.count_files(path)
    db.session.add(dataset)
    db.session.commit()
    return jsonify({'status': 'success'})


@app.route('/api/dataset/<int:id>/get/text/full/<path:filepath>')
def api_dataset_get_full_text(id, filepath):
    dataset = Dataset.query.get(id)
    text = dataset.get_full_text(filepath)
    return jsonify({'status': 'success', 'text': text})


@app.route('/api/models/get/model_template/<string:model_name>')
def api_get_model_template(model_name):
    model_template = open(
        os.path.join(app.config['DEEPSTATION_ROOT'], 'src',
                     'model_templates', model_name)).read()
    return jsonify({'model_template': model_template})


@app.route('/api/models/remove', methods=['POST'])
def api_remove_model():
    id = request.form['model_id']
    model = Model.query.get(id)
    model.delete()
    return redirect(url_for('index'))


@app.route('/api/models/check_train_progress')
def api_check_train_progress():
    return jsonify({'progress': Model.get_train_progresses()})


@app.route('/api/models/start/train', methods=['POST'])
def api_start_train():
    dataset_id = request.form['dataset_id']
    model_id = request.form['model_id']
    epoch = request.form['epoch']
    pretrained_model = request.form['pretrained_model']
    gpu_num = request.form['gpu_num']
    type = request.form['model_type']
    batchsize = request.form['batchsize']
    if type == 'image':
        resize_mode = request.form['resize_mode']
        channels = request.form['channels']
        avoid_flipping = request.form['avoid_flipping']
        if int(avoid_flipping) == 1:
            avoid_flipping = True
        else:
            avoid_flipping = False
        runner.run_imagenet_train(
            app.config['PREPARED_DATA'],
            app.config['TRAINED_DATA'],
            dataset_id,
            model_id,
            int(epoch),
            pretrained_model,
            int(gpu_num),
            resize_mode,
            int(channels),
            avoid_flipping,
            int(batchsize)
        )
    elif type == 'text':
        use_wakati_temp = int(request.form['use_wakatigaki'])
        use_wakatigaki = True if use_wakati_temp == 1 else False
        runner.run_lstm_train(
            app.config['PREPARED_DATA'],
            app.config['TRAINED_DATA'],
            dataset_id,
            model_id,
            int(epoch),
            pretrained_model,
            int(gpu_num),
            use_wakatigaki,
            int(batchsize)
        )
    return jsonify({'status': 'OK'})


@app.route('/api/models/resume/train', methods=['POST'])
def api_resume_train():
    model_id = request.form['model_id']
    gpu_num = int(request.form['gpu_num'])
    model = Model.query.get(model_id)
    if model.type == 'image':
        runner.resume_imagenet_train(
            app.config['TRAINED_DATA'],
            model,
            gpu_num
        )
    elif model.type == 'text':
        runner.resume_lstm_train(
            app.config['PREPARED_DATA'],
            app.config['TRAINED_DATA'],
            model,
            gpu_num
        )
    return jsonify({'status': 'OK'})


@app.route('/api/models/<int:id>/get/train_data/log/')
def api_get_training_log(id):
    model = Model.query.get(id)
    # for backward compatibility.
    if model.train_log and os.path.exists(model.train_log):
        log = open(model.train_log).read()
        return jsonify({'status': 'ready', 'data': log, 'is_trained': model.is_trained})
    if model.train_log_path and os.path.exists(model.train_log_path):
        def data(row):
            obj = json.loads(row)
            data_type = obj['type']
            return obj[data_type] if data_type in obj else ""
        log = map(data, open(model.train_log_path))
        return jsonify({'status': 'ready', 'data': '<br>'.join(log), 'is_trained': model.is_trained})

    return jsonify({'status': 'log file not ready'})


# SSE "protocol" is described here: http://mzl.la/UPFyxY
class ServerSentEvent(object):
    def __init__(self, data):
        self.data = data
        self.event = None
        self.id = None
        self.desc_map = {
            self.data: "data",
            self.event: "event",
            self.id: "id"
        }

    def encode(self):
        if self.data == 'None':
            # これはタイムアウト避けのコメントアウト行なのでコンソールには出てこない。
            return ':\n\n'
        lines = ["{}: {}".format(v, k) for k, v in self.desc_map.iteritems() if k]
        return "{}\n\n".format("\n".join(lines))


@app.route('/api/models/<int:model_id>/get/train_data/log/subscribe')
def api_training_log_subscribe(model_id):
    def gen():
        queue = Queue()
        train_logger.subscribe(model_id, queue)
        try:
            while True:
                result = queue.get()
                ev = ServerSentEvent(str(result))
                yield ev.encode()
        # 通信が切断されるとこの例外が上がる。
        except GeneratorExit:
            train_logger.unsubscribe(model_id, queue)

    return Response(gen(), mimetype="text/event-stream")


@app.route('/api/models/<int:id>/get/train_data/graph/')
def api_get_training_graph(id):
    model = Model.query.get(id)
    if model.line_graph is None or not os.path.exists(model.line_graph):
        return jsonify({'status': 'graph not ready'})
    graph = open(model.line_graph).read()
    return jsonify({'status': 'ready', 'data': graph, 'is_trained': model.is_trained})


@app.route('/api/models/<int:id>/get/layer_names/<int:epoch>')
def api_get_layer_names(id, epoch):
    model = Model.query.get(id)
    return jsonify({'layers': model.get_layers(epoch)})


@app.route('/api/models/<int:id>/get/layer_viz/<int:epoch>/<string:layer_name>')
def api_get_layer_vizualization(id, epoch, layer_name):
    model = Model.query.get(id)
    result = model.get_visualized_layer(epoch, layer_name)
    result['status'] = 'success'
    return jsonify(result)


@app.route('/api/models/lstm/generate_text/', methods=['POST'])
def api_do_lstm_prediction():
    id = request.form['model_id']
    epoch = int(request.form['epoch'])
    result_length = int(request.form['result_length'])
    primetext = request.form['primetext']
    model = Model.query.get(id)

    try:
        return jsonify({'result': model.lstm_predict(epoch, primetext, result_length)})
    except IOError:
        # TODO: status code
        return jsonify({'error': strings.EPOCH_FILE_UNDER_TRAINING_ERROR})


@app.route('/api/models/download/files/', methods=['POST'])
def download_trained_files():
    id = request.form['model_id']
    epoch = int(request.form['epoch'])
    model = Model.query.get(id)
    zip_path = model.get_trained_files(epoch, os.path.join(app.config['DEEPSTATION_ROOT'], 'temp'))
    return send_file(
        zip_path,
        mimetype='application/octet-stream',
        as_attachment=True,
        attachment_filename=os.path.basename(zip_path)
    )


@app.route('/api/models/terminate/train/', methods=['POST'])
def api_terminate_trained():
    id = request.form['id']
    model = Model.query.get(id)
    interruptable = runner.INTERRUPTABLE_PROCESSES.get(model.id)
    if interruptable:
        interruptable.set_interrupt()
        while not interruptable.is_interruptable():
            # 待機中に中断ではなく完了した場合↓でreturnする。
            if not interruptable.is_interrupting():
                return jsonify({'status': 'success'})
            sleep(1)
        model.terminate_train()
        interruptable.terminate()
    return jsonify({'status': 'success'})


# =====================================================================
# misc.
# =====================================================================


def get_system_info():
    info = ds_util.get_system_info()
    info['deepstation_version'] = __version__
    return info


if __name__ == '__main__':
    app.debug = app.config['DEBUG']
    server = WSGIServer((app.config['HOST'], app.config['PORT']), app)
    server.serve_forever()