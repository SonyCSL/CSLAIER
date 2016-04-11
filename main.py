# -*- coding: utf-8 -*-

import os
import signal
import os.path
import platform
import sys
import traceback
import bottle
from bottle.ext import sqlite
import imp
import random
import re
import zipfile
import yaml
import shutil
import cv2
import numpy
import time
from multiprocessing import Process
import sqlite3
import subprocess
import chainer
from xml.etree.ElementTree import *
import six
import cPickle as pickle
from datetime import datetime
from json import dumps
from PIL import Image

import imagenet_inspect
import train
import visualizer
import scipy.misc


# initialization
DEEPSTATION_VERSION = "0.4.2"
DEEPSTATION_ROOT = (os.getcwd() + os.sep + __file__).replace('main.py', '')
f = open(DEEPSTATION_ROOT + os.sep + 'settings.yaml')
settings = yaml.load(f)
f.close()

app = bottle.Bottle()
plugin = sqlite.Plugin(dbfile=DEEPSTATION_ROOT + os.sep + 'deepstation.db' )
app.install(plugin)
UPLOADED_IMAGES_DIR = settings['uploaded_images'] if settings['uploaded_images'].startswith('/') else os.path.realpath(DEEPSTATION_ROOT + settings['uploaded_images'])
UPLOADED_RAW_FILES_DIR = settings['uploaded_raw_files'] if settings['uploaded_raw_files'].startswith('/') else os.path.realpath(DEEPSTATION_ROOT + settings['uploaded_raw_files'])
PREPARED_DATA_DIR      = settings['prepared_data'] if settings['prepared_data'].startswith('/') else os.path.realpath(DEEPSTATION_ROOT + settings['prepared_data'])
TRAINED_DATA_DIR       = settings['trained_data'] if settings['trained_data'].startswith('/') else os.path.realpath(DEEPSTATION_ROOT + settings['trained_data'])
TEMP_IMAGE_DIR         = settings['inspection_temp_image'] if settings['inspection_temp_image'].startswith('/') else os.path.realpath(DEEPSTATION_ROOT + settings['inspection_temp_image'])
INSPECTION_RAW_IMAGE   = settings['inspection_raw_image'] if settings['inspection_raw_image'].startswith('/') else os.path.realpath(DEEPSTATION_ROOT + settings['inspection_raw_image'])
NVIDIA_SMI_CMD         = settings['nvidia_smi']

# static files
@app.route('/statics/<filepath:path>')
def server_static(filepath):
    return bottle.static_file(filepath, DEEPSTATION_ROOT + os.sep + 'statics' + os.sep)
    
@app.route('/uploaded_images/<filepath:path>')
def uploaded_files(filepath):
    return bottle.static_file(filepath, UPLOADED_IMAGES_DIR )

@app.route('/inspection/images/<filepath:path>')
def images_for_inspection(filepath):
    return bottle.static_file(filepath, INSPECTION_RAW_IMAGE)

@app.route('/trained_models/download/<filepath:path>')
def download_trained_model(filepath):
    filename = filepath.split('/')[-1]
    return bottle.static_file(filepath, TRAINED_DATA_DIR, download=filename, mimetype="application/octet-stream")

@app.route('/layers/<id>/<epoch>/<filename>')
def show_layer_image(id, epoch, filename, db):
    model_row = db.execute('select trained_model_path from Model where id = ?', (id,))
    trained_model_path = model_row.fetchone()[0]
    return bottle.static_file(filename, trained_model_path + os.sep + epoch)

# main
@app.route('/')
def index(db):
    models = db.execute('select Model.id, Model.name, Model.epoch, Model.is_trained, Model.created_at, Model.network_name, Model.algorithm, Dataset.name from Model left join Dataset on Model.dataset_id = Dataset.id order by Model.id DESC')
    dataset_cur = db.execute('select id, name, dataset_path from Dataset')
    dataset_rows = dataset_cur.fetchall()
    datasets = []
    for d in dataset_rows:
        datasets.append({"id": d[0], "name": d[1], "dataset_path": d[2], "thumbnails": get_files_in_random_order(d[2], 4), "file_num": count_files(d[2]), "category_num": count_categories(d[2])})
    return bottle.template('index.html', models = models.fetchall(), datasets = datasets, system_info = get_system_info(), gpu_info = get_gpu_info(), chainer_version = get_chainer_version(), python_version = get_python_version(), deepstation_version = DEEPSTATION_VERSION)

@app.route('/inspection/upload', method='POST')
def do_upload_for_inspection(db):
    model_id = bottle.request.forms.get('model_id')
    epoch = int(bottle.request.forms.get('epoch'))
    upload = bottle.request.files.get('fileInput')
    name, ext = os.path.splitext(upload.filename)
    if ext not in ('.jpg', '.jpeg', '.gif', '.png'):
        return show_error_screen("File extension not allowed.")
    timestamp_str = get_timestamp()
    new_filename = INSPECTION_RAW_IMAGE + os.sep + timestamp_str + upload.filename
    try:
        upload.save(new_filename)
        row_model = db.execute('select prepared_file_path, trained_model_path, network_path, name ,resize_mode, channels from Model where id = ?', (model_id,))
    except:
        return show_error_screen(traceback.format_exc(sys.exc_info()[2]))
    model_info = row_model.fetchone()
    result = inspect(new_filename, model_info[1] + os.sep + 'model%04d'%epoch, model_info[0], model_info[2],model_info[4],model_info[5])
    return bottle.template('inspection_result.html',image=timestamp_str + upload.filename,results=result, name=model_info[3], epoch=epoch)

@app.route('/dataset/show/<id>')
def dataset_show(id, db):
    row = db.execute('select name, dataset_path from Dataset where id = ?', (id,))
    dataset_info = row.fetchone()
    name = dataset_info[0]
    dataset_root_path = dataset_info[1]
    if len(os.listdir(dataset_root_path)) == 1:
        dataset_root_path = dataset_root_path + os.sep + os.listdir(dataset_root_path)[0]
    dataset = []
    for path in find_all_directories(dataset_root_path):
        dataset.append({"path": path.replace(UPLOADED_IMAGES_DIR, ""), "file_num": count_files(path), "category": path.split(os.sep)[-1], "thumbnails": get_files_in_random_order(path, 4)})
    return bottle.template('dataset_show.html', dataset = dataset, name=name, dataset_id = id)

@app.route('/dataset/show/<id>/<filepath:path>')
def dataset_category_show(id, filepath, db):
    row = db.execute('select name from Dataset where id = ?', (id,))
    dataset_name = row.fetchone()[0]
    images = []
    for path in find_all_files(UPLOADED_IMAGES_DIR + os.sep + filepath):
        images.append(path.replace(UPLOADED_IMAGES_DIR + os.sep, ''))
    return bottle.template('dataset_category_detail.html', name = dataset_name, count = len(images), images = images, category = filepath.split(os.sep)[-1], dataset_id = id, dataset_path = filepath)

@app.route('/dataset/delete/file/<id>/<filepath:path>', method="POST")
def dataset_delete_an_image(id, filepath):
    file_name = bottle.request.forms.get('file_path')
    try:
        os.remove(UPLOADED_IMAGES_DIR + os.sep + file_name)
    except:
        return show_error_screen(traceback.format_exc(sys.exc_info()[2]))
    return bottle.redirect('/dataset/show/' + id + '/' + filepath)

@app.route('/dataset/delete/category/<id>', method="POST")
def dataset_delete_a_category(id):
    category_path = bottle.request.forms.get('category_path')
    try:
        shutil.rmtree(UPLOADED_IMAGES_DIR + os.sep + category_path)
    except:
        return show_error_screen(traceback.format_exc(sys.exc_info()[2]))
    return bottle.redirect('/dataset/show/' + id)

@app.route('/dataset/upload/<id>/<filepath:path>', method="POST")
def dataset_add_image_to_category(id, filepath):
    upload = bottle.request.files.get('fileInput')
    name, ext = os.path.splitext(upload.filename)
    if ext not in ('.jpg', '.jpeg', '.gif', '.png'):
        return show_error_screen('File extension not allowed.')
    new_filename = UPLOADED_IMAGES_DIR + os.sep + filepath + os.sep + get_timestamp() + '_' + upload.filename
    try:
        upload.save(new_filename)
    except:
        return show_error_screen(traceback.format_exc(sys.exc_info()[2]))
    return bottle.redirect('/dataset/show/' + id + '/' + filepath)

@app.route('/dataset/create/category/<id>', method="POST")
def dataset_create_category(id, db):
    category_name = bottle.request.forms.get('category_name')
    result = db.execute('select dataset_path from Dataset where id = ?', (id,))
    dataset_path = result.fetchone()[0]
    if len(os.listdir(dataset_path)) == 1:
        dataset_path = dataset_path + os.sep + os.listdir(dataset_path)[0]
    try:
        os.mkdir(dataset_path + os.sep + category_name)
    except:
        return show_error_screen(traceback.format_exc(sys.exc_info()[2]))
    bottle.response.content_type = 'application/json'
    return dumps({"status": "ok"})

@app.route('/dataset/remove/<id>')
def dataset_delete(id, db):
    row = db.execute('select dataset_path from Dataset where id = ?', (id,))
    dataset_path = row.fetchone()[0]
    try:
        db.execute('delete from Dataset where id = ?', (id,))
        shutil.rmtree(dataset_path)
    except:
        return show_error_screen(traceback.format_exc(sys.exc_info()[2]))
    return bottle.redirect('/')

@app.route('/models/show/<id>')
def show_model_detail(id, db):
    row_model = db.execute('select id, name, epoch, algorithm, is_trained, network_path, trained_model_path, graph_data_path, dataset_id, created_at, network_name, resize_mode, channels from Model where id = ?', (id,))
    model_info = row_model.fetchone()
    
    if model_info[12] == 3:
        color_mode = "RGB"
    else:
        color_mode = "Grayscale"
    
    ret = {
        "id": model_info[0],
        "name": model_info[1],
        "epoch": model_info[2],
        "algorithm": model_info[3],
        "is_trained": model_info[4],
        "network_path": model_info[5],
        "trained_model_path": model_info[6],
        "graph_data_path": model_info[7],
        "dataset_id": model_info[8],
        "created_at": model_info[9],
        "network_name": model_info[10],
        "resize_mode": model_info[11],
        "channels": color_mode
    }
    gpu_info = get_gpu_info()
    ret['gpu_num'] = 0 if 'gpus' not in gpu_info else len(gpu_info['gpus'])
    if ret['dataset_id'] is not None:
        row_dataset = db.execute('select name from Dataset where id = ?', (ret['dataset_id'],))
        dataset_info = row_dataset.fetchone()
        if dataset_info:
            ret['dataset_name'] = dataset_info[0]
        else:
            ret['dataset_name'] = '---'
    else:
        ret['dataset_name'] = '---'
    if model_info[6]:
        pretrained_models = sorted(os.listdir(model_info[6]), reverse=True)
        if pretrained_models:
            pretrained_models = filter(lambda file:file.find('model')>-1, pretrained_models)
            pretrained_models.append("New")
        else:
            pretrained_models=["New"]
    else:
        pretrained_models=["New"]
    if ret['resize_mode'] is None or ret['resize_mode'] == '':
        ret['resize_mode'] = '---'
    model_txt = open(ret['network_path']).read()
    row_all_datasets = db.execute('select id, name from Dataset')
    all_datasets_info = row_all_datasets.fetchall()
    return bottle.template('models_detail.html', model_info = ret, datasets = all_datasets_info, model_txt=model_txt,system_info = get_system_info(),gpu_info = get_gpu_info(), chainer_version = get_chainer_version(), python_version = get_python_version(),pretrained_models=pretrained_models, deepstation_version = DEEPSTATION_VERSION)

@app.route('/models/start/train', method="POST")
def kick_train_start(db):
    dataset_id = bottle.request.forms.get('dataset_id')
    model_id = bottle.request.forms.get('model_id')
    epoch = bottle.request.forms.get('epoch')
    pretrained_model = bottle.request.forms.get('pretrained_model')
    gpu_num = bottle.request.forms.get('gpu_num')
    resize_mode = bottle.request.forms.get('resize_mode')
    channels = int(bottle.request.forms.get('channels'))
    avoid_flipping = int(bottle.request.forms.get('avoid_flipping'))
    # image_size = int(bottle.request.forms.get('image_size'))
    image_size = 256
    row_ds = db.execute('select dataset_path, name from Dataset where id = ?', (dataset_id,))
    (ds_path, dataset_name) = row_ds.fetchone()
    prepared_file_path = PREPARED_DATA_DIR + os.sep + get_timestamp()
    bottle.response.content_type = 'application/json'
    try:
        os.mkdir(prepared_file_path)
        db.execute('update Model set prepared_file_path = ?, epoch = ?, is_trained = 1, dataset_id = ?, resize_mode = ?, channels = ? where id = ?', (prepared_file_path, epoch, dataset_id, resize_mode, channels,  model_id))
        db.commit()
        prepare_for_train(ds_path, prepared_file_path,image_size, resize_mode, channels)
        start_train(model_id, epoch, prepared_file_path, gpu_num,pretrained_model, db, avoid_flipping)
    except:
        return dumps({"status": "error", "traceback": traceback.format_exc(sys.exc_info()[2])})
    return dumps({"status": "OK", "dataset_name": dataset_name})
 
@app.route('/models/download/<id>/<epoch>')
def get_trained_model(id, epoch, db):
    row_model = db.execute('select trained_model_path from Model where id = ?', (id,))
    path = row_model.fetchone()[0]
    epoch = int(epoch)
    path = path.replace(TRAINED_DATA_DIR, '')
    return bottle.redirect('/trained_models/download' + path + '/model%04d'%epoch)

@app.route('/models/labels/download/<id>')
def get_label_text(id, db):
    row_model = db.execute('select prepared_file_path from Model where id = ?', (id,))
    path = row_model.fetchone()[0]
    return bottle.static_file('labels.txt', path, download='labels.txt', mimetype="text/plain")

@app.route('/models/mean/download/<id>')
def get_mean_file(id, db):
    row_model = db.execute('select prepared_file_path from Model where id = ?', (id,))
    path = row_model.fetchone()[0]
    return bottle.static_file('mean.npy', path, download='mean.npy', mimetype="application/octet-stream")

@app.route('/models/new')
def make_new_model():
    model_templates = os.listdir(DEEPSTATION_ROOT + os.sep + 'model_templates')
    return bottle.template('new_model.html', templates = model_templates)
    
@app.route('/models/create', method="POST")
def create_new_model(db):
    model_name = bottle.request.forms.get('model_name').strip()
    my_network = bottle.request.forms.get('my_network')
    model_template = bottle.request.forms.get('model_template')
    network_type = bottle.request.forms.get('network_type').strip()
    algorithm = None
    
    if not re.match(r".+\.py", model_name):
        model_name += '.py'
    if network_type is None or network_type is '':
        if model_template is not None or model_template is not '':
            network_type = re.sub(r"\.py$", "", model_template)
        else:
            network_type = None
    if algorithm is '':
        algorithm = None
    
    network_file_path = DEEPSTATION_ROOT + os.sep + 'models' + os.sep + model_name
    try:
        network_file = open(network_file_path, "w")
        network_file.write(my_network)
    except:
        return show_error_screen(traceback.format_exc(sys.exc_info()[2]))
    finally:
        network_file.close()
        
    t = (model_name, network_file_path, network_type, algorithm)
    try:
        row = db.execute("insert into Model(name, network_path, network_name, algorithm) values(?,?,?,?)", t)
    except:
        return show_error_screen(traceback.format_exc(sys.exc_info()[2]))
    return bottle.redirect('/models/show/' + str(row.lastrowid))

@app.route('/models/delete/<id>', method="POST")
def delete_model(id, db):
    row_model = db.execute('select network_path, trained_model_path, prepared_file_path from Model where id = ?', (id,));
    network_path, trained_model_path, prepared_file_path = row_model.fetchone()
    db.execute('delete from Model where id = ?', (id,))
    if prepared_file_path: shutil.rmtree(prepared_file_path)
    if trained_model_path: shutil.rmtree(trained_model_path)
    os.remove(network_path)
    if os.path.exists(network_path + 'c'): os.remove(network_path + 'c')
    return bottle.redirect('/')

@app.route('/cleanup')
def cleanup(db):
    rows = db.execute('select prepared_file_path from Model')
    paths = rows.fetchall()
    for p in paths:
        if p[0] is None: continue
        for f in os.listdir(p[0]):
            if f.split('.')[-1] in ['jpg', 'jpeg', 'gif', 'png']:
                os.remove(p[0] + os.sep + f)
    return bottle.redirect('/')

# API ----------------------------------------------------------

# hundle uploaded file
@app.route('/api/upload', method='POST')
def do_upload(db):
    bottle.response.content_type = 'application/json'
    dataset_name = bottle.request.forms.get('dataset_name')
    upload = bottle.request.files.get('fileInput')
    name, ext = os.path.splitext(upload.filename)
    if ext not in ('.zip'):
        return show_error_screen("File extension not allowed.")
    timestamp_str = get_timestamp()
    new_filename = re.sub(r'\.zip$', '_' + timestamp_str + '.zip', upload.filename)
    try:
        upload.save(UPLOADED_RAW_FILES_DIR + os.sep + new_filename, overwrite=True)
        zf = zipfile.ZipFile(UPLOADED_RAW_FILES_DIR + os.sep + new_filename, 'r')
        upload_image_dir_root = UPLOADED_IMAGES_DIR + os.sep + timestamp_str
        os.mkdir(upload_image_dir_root)
        db.execute('insert into Dataset(name, dataset_path, updated_at) values(?, ?, current_timestamp)', (dataset_name, upload_image_dir_root))
        for f in zf.namelist():
            temp_file_path = upload_image_dir_root + os.sep + f
            if ('__MACOSX' in f) or ('.DS_Store' in f):
                continue
            if not os.path.basename(f):
                if os.path.exists(temp_file_path):
                    continue
                os.mkdir(temp_file_path)
            else:
                if os.path.exists(temp_file_path):
                    uzf = file(temp_file_path, 'w+b')
                else:
                    uzf = file(temp_file_path, 'wb')
                uzf.write(zf.read(f))
                uzf.close()
    except:
        return dumps({'error': traceback.format_exc(sys.exc_info()[2])})
    finally:
        if 'zf' in locals():
            zf.close()
        if 'uzf' in locals():
            uzf.close()
    return dumps({'status': 'success'})

@app.route('/api/models/get_model_template/<model_name>')
def api_get_model_template(model_name):
    model_template = open(DEEPSTATION_ROOT + os.sep + 'model_templates' + os.sep + model_name).read()
    bottle.response.content_type = 'application/json'
    ret = {'model_template': model_template}
    return dumps(ret)
    
@app.route('/api/models/get_training_data/<id>')
def api_get_training_data(id, db):
    model_row = db.execute('select line_graph_data_path, is_trained from Model where id = ?', (id,))
    model = model_row.fetchone()
    bottle.response.content_type = 'application/json'
    if model[0] is None or not os.path.exists(model[0]):
        return dumps({'status': 'graph not ready', 'is_trained': model[1]})
    f = open(model[0], 'r')
    data = f.read()
    f.close()
    return dumps({'status': 'ready', 'data': data, 'is_trained': model[1]})

@app.route('/api/models/get_training_log/<id>')
def api_get_training_data(id, db):
    model_row = db.execute('select line_graph_data_path, is_trained from Model where id = ?', (id,))
    model = model_row.fetchone()
    bottle.response.content_type = 'application/json'
    if model[0] is None:
        return dumps({'status': 'graph not ready'})
    filename = model[0].replace('line_graph.tsv','log.html')
    if not os.path.exists(filename):
        return dumps({'status': 'graph not ready'})
    f = open(filename, 'r')
    data = f.read()
    f.close()
    return dumps({'status': 'ready', 'data': data, 'is_trained': model[1]})
    
@app.route('/api/models/check_train_progress')
def api_check_train_progress(db):
    model_row = db.execute('select id, is_trained from Model')
    models = model_row.fetchall()
    progress = []
    for m in models:
        progress.append({'id': m[0], 'is_trained': m[1]})
    bottle.response.content_type = 'application/json'
    return dumps({'progress': progress})
    
@app.route('/api/models/get_layer_viz/<model_id>/<epoch>/<layer_name>')
def api_get_layer_viz(model_id, epoch, layer_name, db):
    bottle.response.content_type = 'application/json'
    validation_result = validation_for_layer_viz(model_id, epoch, db)
    if validation_result['status'] == 'error':
        dumps(validation_result)
    else:
        network = validation_result['network']
        trained_model_path = validation_result['trained_model_path']
        trained_model = validation_result['trained_model']
    if os.path.exists(trained_model_path + os.sep + epoch + os.sep + layer_name + '.png'):
        return dumps({'status': 'success', 'filename': layer_name + '.png', 'epoch': epoch})
    if not os.path.exists(trained_model_path + os.sep + epoch):
        os.mkdir(trained_model_path + os.sep + epoch)
    v = visualizer.LayerVisualizer(network, trained_model, trained_model_path + os.sep + epoch)
    layer_name = layer_name.replace('_', '/')
    filename = v.visualize(layer_name)
    if filename is None:
        return dumps({'status': 'error', 'message': 'could not generate layer visualization'})
    else:
        return dumps({'status': 'success', 'filename': filename, 'epoch': epoch, })

@app.route('/api/models/get_layer_names/<model_id>/<epoch>')
def api_get_layer_names(model_id, epoch, db):
    bottle.response.content_type = 'application/json'
    validation_result = validation_for_layer_viz(model_id, epoch, db)
    if validation_result['status'] == 'error':
        dumps(validation_result)
    else:
        network = validation_result['network']
        trained_model_path = validation_result['trained_model_path']
        trained_model = validation_result['trained_model']
    
    v = visualizer.LayerVisualizer(network, trained_model, trained_model_path)
    return dumps(v.get_layer_list())

def validation_for_layer_viz(model_id, epoch, db):
    model_row = db.execute('select epoch, network_path, trained_model_path from Model where id = ?', (model_id,))
    (epoch_max, network, trained_model_path) = model_row.fetchone()
    if int(epoch, 10) > epoch_max:
        return {'status': 'error', 'message': 'selected epoch is bigger than trained epoch.'}
    epoch_str = "{0:0>4}".format(int(epoch, 10))
    trained_model = None
    for f in find_all_files(trained_model_path):
        if f.find(epoch_str) > -1:
            trained_model = f
            break
    if trained_model is None:
        return {'status': 'error', 'message': 'could not find the trained_model'}
    return {'status': 'OK', 'network': network, 'trained_model_path': trained_model_path, 'trained_model': trained_model}

@app.route('/api/models/kill_train', method='POST')
def api_kill_train(db):
    bottle.response.content_type = 'application/json'
    model_id = bottle.request.forms.get('id')
    c = db.execute('select pid from Model where id = ?', (model_id,))
    pid = c.fetchone()[0]
    if pid is not None:
        os.kill(pid, signal.SIGTERM)
        db.execute('update Model set is_trained = 0, pid = null where id = ?', (model_id,))
    return dumps({'status': 'success', 'message': 'successfully terminated'})

#------- private methods ---------
def find_all_files(directory):
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.startswith('__MACOSX') or f.startswith('.DS_Store'):
                continue
            yield os.path.join(root, f)

def find_all_directories(directory):
    for root, dirs, files in os.walk(directory):
        if len(dirs) == 0:
            yield root

def make_train_data(target_dir, prepared_data_dir, image_insize, resize_mode, channels):
    train = open(prepared_data_dir + os.sep + 'train.txt', 'w')
    test = open(prepared_data_dir + os.sep + 'test.txt', 'w')
    labelsTxt = open(prepared_data_dir + os.sep + 'labels.txt', 'w')
    classNo = 0
    count = 0
    for path, dirs, files in os.walk(target_dir):
        if not dirs:
            start = path.rfind(os.sep) + 1
            labelsTxt.write(path[start:].split(os.sep)[0] + "\n")
            startCount = count
            length = len(files)
            for f in files:
                if(f.split('.')[-1] not in ["jpg", "jpeg", "gif", "png"]):
                    continue
                if (os.path.getsize(os.path.join(path, f))) <= 0:
                    continue
                imagepath = prepared_data_dir + os.sep + "image%07d" %count + ".jpg"
                resize_image(os.path.join(path, f), imagepath, image_insize,resize_mode, channels)
                if count - startCount < length * 0.75:
                    train.write(imagepath + " %d\n" % classNo)
                else:
                    test.write(imagepath + " %d\n" % classNo)
                count += 1
            classNo += 1
    train.close()
    test.close()
    labelsTxt.close()
    return

def resize_image(source, dest,image_insize,resize_mode,channels):
    output_side_length = image_insize
    
    if channels == 1:
        mode = "L"
    else:
        mode = "RGB"
    
    image = Image.open(source)
    image = image.convert(mode)
    image = numpy.array(image)
    
    height = image.shape[0]
    width = image.shape[1]
    
    if resize_mode not in ['crop', 'squash', 'fill', 'half_crop']:
        raise ValueError('resize_mode "%s" not supported' % resize_mode)
    
	### Resize
    interp = 'bilinear'
    
    width_ratio = float(width) / output_side_length
    height_ratio = float(height) / output_side_length
    if resize_mode == 'squash' or width_ratio == height_ratio:
        image = scipy.misc.imresize(image, (output_side_length, output_side_length), interp=interp)
    elif resize_mode == 'crop':
        # resize to smallest of ratios (relatively larger image), keeping aspect ratio
        if width_ratio > height_ratio:
            resize_height = output_side_length
            resize_width = int(round(width / height_ratio))
        else:
            resize_width = output_side_length
            resize_height = int(round(height/ width_ratio))
        image = scipy.misc.imresize(image, (resize_height, resize_width), interp=interp)
        
        # chop off ends of dimension that is still too long
        if width_ratio > height_ratio:
            start = int(round((resize_width-output_side_length)/2.0))
            image = image[:,start:start+output_side_length]
        else:
            start = int(round((resize_height-output_side_length)/2.0))
            image = image[start:start+output_side_length,:]
    else:
        if resize_mode == 'fill':
            # resize to biggest of ratios (relatively smaller image), keeping aspect ratio
            if width_ratio > height_ratio:
                resize_width = output_side_length
                resize_height = int(round(height / width_ratio))
                if (output_side_length - resize_height) % 2 == 1:
                    resize_height += 1
            else:
                resize_height = output_side_length
                resize_width = int(round(width/ height_ratio))
                if (output_side_length - resize_width) % 2 == 1:
                    resize_width += 1
            image = scipy.misc.imresize(image, (resize_height, resize_width), interp=interp)
        elif resize_mode == 'half_crop':
            # resize to average ratio keeping aspect ratio
            new_ratio = (width_ratio + height_ratio) / 2.0
            resize_width = int(round(width / new_ratio))
            resize_height = int(round(height / new_ratio))
            if width_ratio > height_ratio and (output_side_length - resize_height) % 2 == 1:
                resize_height += 1
            elif width_ratio < height_ratio and (output_side_length - resize_width) % 2 == 1:
                resize_width += 1
            image = scipy.misc.imresize(image, (resize_height, resize_width), interp=interp)
            # chop off ends of dimension that is still too long
            if width_ratio > height_ratio:
                start = int(round((resize_width-output_side_length)/2.0))
                image = image[:,start:start+output_side_length]
            else:
                start = int(round((resize_height-output_side_length)/2.0))
                image = image[start:start+output_side_length,:]
        else:
            raise Exception('unrecognized resize_mode "%s"' % resize_mode)
            
        # fill ends of dimension that is too short with random noise
        if width_ratio > height_ratio:
            padding = int((output_side_length - resize_height)/2)
            noise_size = (padding, output_side_length)
            if channels > 1:
                noise_size += (channels,)
            noise = numpy.random.randint(0, 255, noise_size).astype('uint8')
            image = numpy.concatenate((noise, image, noise), axis=0)
        else:
            padding = int((output_side_length - resize_width)/2)
            noise_size = (output_side_length, padding)
            if channels > 1:
                noise_size += (channels,)
            noise = numpy.random.randint(0, 255, noise_size).astype('uint8')
            image = numpy.concatenate((noise, image, noise), axis=1)
    cv2.imwrite(dest, image)
    return

def compute_mean(prepared_data_dir):
    sum_image = None
    count = 0
    for line in open(prepared_data_dir + os.sep + 'train.txt'):
        filepath = line.strip().split()[0]
        image = numpy.asarray(Image.open(filepath))
        if image.ndim == 3:
            image = image.transpose(2, 0, 1)
        else:
            zeros = numpy.zeros((256, 256))
            image = numpy.array([image, zeros, zeros])
        if sum_image is None:
            sum_image = numpy.ndarray(image.shape, dtype=numpy.float32)
            sum_image[:] = image
        else:
            sum_image += image
        count += 1
    mean = sum_image / count
    pickle.dump(mean, open(prepared_data_dir + os.sep + 'mean.npy', 'wb'), -1)
    return

def prepare_for_train(target_dir, prepared_data_dir,image_insize, resize_mode, channels):
    make_train_data(target_dir, prepared_data_dir, image_insize,resize_mode, channels)
    compute_mean(prepared_data_dir)

def start_train(model_id, epoch, prepared_data_dir, gpu, pretrained_model, db, avoid_flipping):
    if not is_prepared_to_train(prepared_data_dir):
        raise Exception('preparation is not done')
    train_process = Process(
        target=train.do_train,
        args = (
            DEEPSTATION_ROOT + os.sep + 'deepstation.db',
            prepared_data_dir + os.sep + 'train.txt',
            prepared_data_dir + os.sep + 'test.txt',
            prepared_data_dir + os.sep + 'mean.npy',
            TRAINED_DATA_DIR,
            'models',
            model_id,
            32,
            250,
            int(epoch, 10),
            int(gpu, 10),
            20,
            pretrained_model,
            avoid_flipping
        )
    )
    train_process.start()
    db.execute('update Model set pid = ? where id = ?', (train_process.pid, model_id))
    db.commit()
    return
    
def is_prepared_to_train(prepared_data_dir):
    if not os.path.isfile(prepared_data_dir + os.sep + 'mean.npy'):
        return False
    if not os.path.isfile(prepared_data_dir + os.sep + 'train.txt'):
        return False
    if not os.path.isfile(prepared_data_dir + os.sep + 'test.txt'):
        return False
    return True

def inspect(image_file_path, target_model, prepared_data_dir, network, resize_mode="fill", channels=3):
    # for backward compatibility
    if resize_mode not in ['fill', 'squash', 'crop', 'half_crop']:
        resize_mode = 'fill'
    if not isinstance(channels, int):
        channels = int(channels)
    gpu_info = get_gpu_info()
    gpu = -1 if 'error' in gpu_info else 0
    ret = imagenet_inspect.inspect(image_file_path, prepared_data_dir + os.sep + 'mean.npy', target_model, prepared_data_dir + os.sep + 'labels.txt', network, resize_mode, channels, gpu)
    return ret

def count_files(path):
    ch = os.listdir(path)
    counter = 0
    for c in ch:
        if os.path.isdir(path + os.sep + c):
            counter += count_files(path + os.sep + c)
        else:
            counter += 1
    return counter
    
# path配下の画像をランダムでnum枚取り出す。
# path配下がディレクトリしか無い場合は配下のディレクトリから
def get_files_in_random_order(path, num):
    children_files = os.listdir(path)
    children_files_num = len(children_files)
    if children_files_num is 0:
        return []
    elif children_files_num is 1:
        if os.path.isdir(path + os.sep + children_files[0]):
            path = path + os.sep + children_files[0]
            temp_file_num = len(os.listdir(path))
            if temp_file_num < num:
                num = temp_file_num
        else:
            num = 1
    elif children_files_num < num:
        num = children_files_num
    files = []
    candidates = random.sample(map(lambda n: path + os.sep + n, os.listdir(path)), num)
    for f in candidates:
        if os.path.isdir(f):
            files.extend(get_files_in_random_order(f, 1))
        else:
            files.append(f.replace(UPLOADED_IMAGES_DIR, ''))
    return files;
    
def get_timestamp():
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
def count_categories(path):
    ch = os.listdir(path)
    count = 0
    if len(ch) is 1:
        if os.path.isdir(path + os.sep + ch[0]):
            count += count_categories(path + os.sep + ch[0])
    else:
        for c in ch:
            if os.path.isdir(path + os.sep + c):
                count += 1
    return count
    
def get_gpu_info():
    ret = {}
    current_platform = platform.system()
    try:
        if current_platform == 'Windows':
            xml = subprocess.check_output([NVIDIA_SMI_CMD, '-q', '-x'], shell=True)
        else:
            xml = subprocess.check_output([NVIDIA_SMI_CMD, '-q', '-x'])
    except:
        return {'error': 'command_not_available'}
		
    elem = fromstring(xml)
    ret['driver_version'] = elem.find('driver_version').text
    gpus = elem.findall('gpu')
    ret_gpus = []
    for g in gpus:
        info = {
            'product_name': g.find('product_name').text,
            'uuid': g.find('uuid').text,
            'fan': g.find('fan_speed').text,
            'minor_number': g.find('minor_number').text
        }
        temperature = g.find('temperature')
        info['temperature'] = temperature.find('gpu_temp').text
        power = g.find('power_readings')
        info['power_draw'] = power.find('power_draw').text
        info['power_limit'] = power.find('power_limit').text
        memory = g.find('fb_memory_usage')
        info['memory_total'] = memory.find('total').text
        info['memory_used'] = memory.find('used').text
        utilization = g.find('utilization')
        info['gpu_util'] = utilization.find('gpu_util').text
        ret_gpus.append(info)
    if current_platform == 'Linux':
        ret_gpus.sort(cmp=lambda x,y: cmp(int(x['minor_number']), int(y['minor_number'])))
    ret['gpus'] = ret_gpus
    return ret
	
def get_system_info():
    df = subprocess.check_output(['df', '-h'])
    disks=df[:-1].split('\n')
    info=[]
    info.append( disks[0].split() )
    for i,disk in enumerate(disks):
        row = disk.split()
        if row[0].find('/') > -1:
           info.append(row)
    return info


def get_chainer_version():
    return chainer.__version__
    
def get_python_version():
    v = sys.version_info
    return str(v[0]) + '.' + str(v[1]) + '.' + str(v[2])
    
def show_error_screen(error):
    return bottle.template('errors.html', detail=error)
    
app.run(server=settings['server_engine'], host=settings['host'], port=settings['port'], debug=settings['debug'])
