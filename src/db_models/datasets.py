# -*- encoding: utf-8 -*-
import os
import datetime
import zipfile
import re
import shutil
from math import ceil
from logging import getLogger

import nkf
from sqlalchemy import desc
from werkzeug import secure_filename

from db_models.shared_models import db
from db_models.models import Model
import common.utils as ds_util

logger = getLogger(__name__)

class Dataset(db.Model):
    id           = db.Column(db.Integer, primary_key = True)
    name         = db.Column(db.Text, unique = True, nullable = False)
    dataset_path = db.Column(db.Text, unique = True)
    type         = db.Column(db.Text)
    updated_at   = db.Column(db.DateTime)
    created_at   = db.Column(db.DateTime)
    models       = db.relationship('Model', backref='dataset', lazy='dynamic')

    def __init__(self, name, type, dataset_path=None):
        self.name         = name
        self.dataset_path = dataset_path
        self.type         = type
        self.updated_at   = datetime.datetime.now()
        self.created_at   = datetime.datetime.now()

    def __repr__(self):
        return

    @classmethod
    def get_datasets_with_samples(cls):
        datasets = cls.query.order_by(desc(Dataset.updated_at))
        ret = []
        for dataset in datasets:
            if not os.path.exists(dataset.dataset_path): continue
            dataset.file_num = ds_util.count_files(dataset.dataset_path)
            dataset.category_num = ds_util.count_categories(dataset.dataset_path)
            if dataset.type == 'image':
                dataset.thumbnails = []
                thumbnails = ds_util.get_files_in_random_order(dataset.dataset_path, 4)
                for t in thumbnails:
                    dataset.thumbnails.append('/files/' + str(dataset.id) + t.replace(dataset.dataset_path, ''))
            elif dataset.type == 'text':
                dataset.sample_text = ds_util.get_texts_in_random_order(dataset.dataset_path, 1, 180)
                dataset.filesize = ds_util.calculate_human_readable_filesize(ds_util.get_file_size_all(dataset.dataset_path))
            ret.append(dataset)
        return ret

    @classmethod
    def get_dataset_with_categories_and_samples(cls, id, limit=20, offset=0):
        dataset = cls.query.get(id)
        dataset_root = dataset.dataset_path
        if len(os.listdir(dataset_root)) == 1:
            dataset_root = os.path.join(dataset_root, os.listdir(dataset_root)[0])
        dataset.category_num = ds_util.count_categories(dataset.dataset_path)
        dataset.pages = int(ceil(float(dataset.category_num) / limit))
        dataset.categories = []
        for index, p in enumerate(ds_util.find_all_directories(dataset_root)):
            if index < offset or offset + limit -1 < index: continue
            if dataset.type == 'image':
                thumbs = ds_util.get_files_in_random_order(p, 4)
                thumbs = map(lambda t:'/files/' + str(dataset.id) + t.replace(dataset.dataset_path, ''), thumbs)
                dataset.categories.append({
                    'dataset_type': dataset.type,
                    'path': p.replace(dataset.dataset_path, ''),
                    'file_num': ds_util.count_files(p),
                    'category': os.path.basename(p),
                    'thumbnails': thumbs
                })
            elif dataset.type == 'text':
                dataset.categories.append({
                    'dataset_type': dataset.type,
                    'path': p.replace(dataset.dataset_path, ''),
                    'file_num': ds_util.count_files(p),
                    'category': os.path.basename(p),
                    'sample_text': ds_util.get_texts_in_random_order(p, 1, 180)
                })
        return dataset

    @classmethod
    def get_dataset_with_category_detail(cls, id, category):
        dataset = cls.query.get(id)
        category_root = os.path.join(dataset.dataset_path, category)
        dataset.category = os.path.basename(category_root)
        files = []
        for p in ds_util.find_all_files(category_root):
            if dataset.type == 'image':
                files.append('/files/' + str(dataset.id) + p.replace(dataset.dataset_path, ''))
            elif dataset.type == 'text':
                files.append({
                    'sample_text': ds_util.get_text_sample(p, 180),
                    'text_path': p.replace(dataset.dataset_path, '')
                })
        dataset.files = files
        dataset.count = ds_util.count_files(category_root)
        dataset.category_root = category_root.replace(dataset.dataset_path, '')
        return dataset

    def delete(self):
        db.session.delete(self)
        try:
            shutil.rmtree(self.dataset_path)
        except Exception as e:
            logger.exception('Could not delete {0}. {1}'.format(self.dataset_path, e))
            raise
        db.session.commit()

    @classmethod
    def remove_category(cls, id, category_path):
        dataset = cls.query.get(id)
        abs_path = os.path.normpath(dataset.dataset_path + category_path)
        try:
            shutil.rmtree(abs_path)
        except Exception as e:
            logger.exception('Could not delete {0}. {1}'.format(dataset.dataset_path, e))
            raise

    @classmethod
    def create_category(cls, id, name):
        ds = cls.query.get(id)
        if len(os.listdir(ds.dataset_path)) == 1:
            only_one_child = os.listdir(ds.dataset_path)[0]
            candidate = os.path.join(ds.dataset_path, only_one_child)
            path_name_sample = ds_util.get_files_in_random_order(candidate, 1)[0]
            if os.path.split(path_name_sample)[0] != candidate:
                ds.dataset_path = candidate
        os.mkdir(os.path.join(ds.dataset_path, name))

    def save_uploaded_data(self, uploaded_file, save_raw_file_to, save_to):
        filename = uploaded_file.filename
        name, ext = os.path.splitext(filename)
        if ext not in ('.zip'):
            raise ValueError('Invalid file type. Only zip file is allowed: ' + filename)
        timestamp_str = ds_util.get_timestamp()
        new_filename = secure_filename(re.sub(r'\.zip$', '_' + timestamp_str + '.zip', filename))
        uploaded_file.save(os.path.join(save_raw_file_to, new_filename))
        # extract zip file
        extract_to = os.path.join(save_to, timestamp_str)
        self.dataset_path = extract_to
        try:
            os.mkdir(extract_to)
        except Exception as e:
            logger.exception('Could not create directory to extract zip file: {0} {1}'.format(extract_to, e))
            raise
        try:
            zf = zipfile.ZipFile(os.path.join(save_raw_file_to, new_filename), 'r')
            for f in zf.namelist():
                if ('__MACOSX' in f) or ('.DS_Store' in f): continue
                temp_path = os.path.join(extract_to, f)
                if not os.path.basename(f):
                    if not os.path.exists(temp_path):
                        os.mkdir(temp_path)
                else:
                    temp, ext = os.path.splitext(temp_path)
                    ext = ext.lower()
                    if self.type == 'image':
                        if ext not in ('.jpg', 'jpeg', '.png', '.gif'):
                            continue
                    elif self.type == 'text':
                        if ext not in ('.txt',):
                            continue
                    if os.path.exists(temp_path):
                        uzf = file(temp_path, 'w+b')
                    else:
                        uzf = file(temp_path, 'wb')
                    uzf.write(zf.read(f))
                    uzf.close()
        except Exception as e:
            logger.exception('Could not extract zip file: {0}'.format(e))
            raise
        finally:
            if 'zf' in locals():
                zf.close()
            if 'uzf' in locals():
                uzf.close()

    def save_uploaded_file_to_category(self, uploaded_file, category):
        filename = uploaded_file.filename
        name, ext = os.path.splitext(filename)
        ext = ext.lower()
        if self.type == 'image':
            if ext not in ('.jpg', '.jpeg', '.png', '.gif'):
                raise ValueError('Invalid file type.')
        elif self.type == 'text':
            if ext not in ('.txt',):
                raise ValueError('Invalid file type.')
        new_filename = os.path.join( self.dataset_path , category , ds_util.get_timestamp() + '_' + secure_filename(filename))
        if self.type == 'image':
            uploaded_file.save(new_filename)
        elif self.type == 'text':
            text = uploaded_file.stream.read()
            if nkf.guess(text) == 'binary':
                raise ValueError('Invalid file type. File must be a text file.')
            f = open(new_filename, 'w')
            f.write(text)
            f.close()

    def remove_file_from_category(self, target_file):
        print target_file
        if self.type == 'image':
            target_file = target_file.replace('/files/' + str(self.id) + '/', '')
        file_path = os.path.normpath(self.dataset_path + os.sep + target_file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    def get_full_text(self, target_file):
        file_path = os.path.join(self.dataset_path, target_file)
        text = ds_util.get_text_sample(file_path)
        text = text.replace("\r", '')
        text = text.replace("\n", '<br>')
        return text
