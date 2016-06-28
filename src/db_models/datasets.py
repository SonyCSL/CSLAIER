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
import common.utils as ds_util

logger = getLogger(__name__)


class Dataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.Text, unique=True, nullable=False)
    dataset_path = db.Column(db.Text, unique=True)
    type = db.Column(db.Text)
    category_num = db.Column(db.Integer)
    file_num = db.Column(db.Integer)
    updated_at = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime)
    models = db.relationship('Model', backref='dataset', lazy='dynamic')

    def __init__(self, name, type, dataset_path=None):
        self.name = name
        self.dataset_path = dataset_path
        self.type = type
        self.updated_at = datetime.datetime.now()
        self.created_at = datetime.datetime.now()

    def __repr__(self):
        return

    @classmethod
    def get_datasets_with_samples(cls, limit=0, offset=0):
        if limit == 0:
            datasets = cls.query.order_by(desc(Dataset.updated_at))
        else:
            datasets = cls.query.order_by(desc(Dataset.updated_at)).limit(limit).offset(offset)
        ret = []
        dirty = False
        for dataset in datasets:
            if not os.path.exists(dataset.dataset_path):
                continue
            if dataset.file_num is None:
                dataset.file_num = ds_util.count_files(dataset.dataset_path)
                dirty = True
            if dataset.category_num is None:
                dataset.category_num = ds_util.count_categories(dataset.dataset_path)
                dirty = True
            if dirty:
                dataset.update_and_commit()
            if dataset.type == 'image':
                dataset.thumbnails = []
                thumbnails = ds_util.get_images_in_random_order(dataset.dataset_path, 4)
                if len(thumbnails) == 0:
                    continue
                for t in thumbnails:
                    dataset.thumbnails.append('/files/' + str(dataset.id)
                                              + t.replace(dataset.dataset_path, ''))
            elif dataset.type == 'text':
                dataset.sample_text = ds_util.get_texts_in_random_order(dataset.dataset_path,
                                                                        1, 180)
                dataset.filesize = ds_util.calculate_human_readable_filesize(
                    ds_util.get_file_size_all(dataset.dataset_path))
            ret.append(dataset)
        return ret, cls.query.count()

    def get_dataset_with_categories_and_samples(self, limit=20, offset=0):
        dataset_root = self.dataset_path
        if len(os.listdir(dataset_root)) == 1:
            dataset_root = os.path.join(dataset_root, os.listdir(dataset_root)[0])
        self.pages = int(ceil(float(self.category_num) / limit))
        self.categories = []
        for index, p in enumerate(ds_util.find_all_directories(dataset_root)):
            if index < offset or offset + limit - 1 < index:
                continue
            if self.type == 'image':
                thumbs = ds_util.get_images_in_random_order(p, 4)
                thumbs = map(lambda t: '/files/' + str(self.id) + t.replace(self.dataset_path, ''),
                             thumbs)
                self.categories.append({
                    'dataset_type': self.type,
                    'path': p.replace(self.dataset_path, ''),
                    'file_num': ds_util.count_files(p),
                    'category': os.path.basename(p),
                    'thumbnails': thumbs
                })
            elif self.type == 'text':
                self.categories.append({
                    'dataset_type': self.type,
                    'path': p.replace(self.dataset_path, ''),
                    'file_num': ds_util.count_files(p),
                    'category': os.path.basename(p),
                    'sample_text': ds_util.get_texts_in_random_order(p, 1, 180)
                })
        return self

    def get_dataset_with_category_detail(self, category, offset=0, limit=100):
        category_root = os.path.join(self.dataset_path, category)
        self.category = os.path.basename(category_root)
        files = []
        for i, p in enumerate(ds_util.find_all_files(category_root)):
            if i < offset or offset + limit - 1 < i:
                continue
            if self.type == 'image':
                files.append('/files/' + str(self.id) + p.replace(self.dataset_path, ''))
            elif self.type == 'text':
                files.append({
                    'sample_text': ds_util.get_text_sample(p, 180),
                    'text_path': p.replace(self.dataset_path, '')
                })
        self.files = files
        self.count = ds_util.count_files(category_root)
        self.pages = int(ceil(float(self.count) / limit))
        self.category_root = category_root.replace(self.dataset_path, '')
        self.original_category = category
        return self

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
        deleted_file_num = ds_util.count_files(abs_path)
        try:
            shutil.rmtree(abs_path)
        except Exception as e:
            logger.exception('Could not delete {0}. {1}'.format(dataset.dataset_path, e))
            raise
        dataset.category_num -= 1
        dataset.file_num -= deleted_file_num
        dataset.update_and_commit()

    @classmethod
    def create_category(cls, id, name):
        ds = cls.query.get(id)
        target = ds.dataset_path
        if len(os.listdir(target)) == 1:
            only_one_child = os.listdir(target)[0]
            candidate = os.path.join(target, only_one_child)
            path_name_sample = ds_util.get_files_in_random_order(candidate, 1)[0]
            if os.path.split(path_name_sample)[0] != candidate:
                target = candidate
        try:
            os.mkdir(os.path.join(target, name))
            ds.category_num += 1
        except Exception as e:
            logger.exception('Could not create directory: {0} {1}'
                             .format(os.path.join(target, name), e))
            raise
        ds.update_and_commit()

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
            logger.exception('Could not create directory to extract zip file: {0} {1}'
                             .format(extract_to, e))
            raise
        file_num = 0
        category_num = 0
        try:
            zf = zipfile.ZipFile(os.path.join(save_raw_file_to, new_filename), 'r')
            for f in zf.namelist():
                if ('__MACOSX' in f) or ('.DS_Store' in f):
                    continue
                temp_path = os.path.join(extract_to, f)
                if not os.path.basename(f):
                    if not os.path.exists(temp_path):
                        os.mkdir(temp_path)
                        category_num += 1
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
                    file_num += 1
        except Exception as e:
            logger.exception('Could not extract zip file: {0}'.format(e))
            raise
        finally:
            if 'zf' in locals():
                zf.close()
            if 'uzf' in locals():
                uzf.close()
        self.category_num = category_num
        self.file_num = file_num
        self.update_and_commit()

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
        new_filename = os.path.join(self.dataset_path, category,
                                    ds_util.get_timestamp() + '_' + secure_filename(filename))
        if self.type == 'image':
            uploaded_file.save(new_filename)
        elif self.type == 'text':
            text = uploaded_file.stream.read()
            if nkf.guess(text) == 'binary':
                raise ValueError('Invalid file type. File must be a text file.')
            f = open(new_filename, 'w')
            f.write(text)
            f.close()
        self.file_num += 1
        self.update_and_commit()

    def remove_file_from_category(self, target_file):
        if self.type == 'image':
            target_file = target_file.replace('/files/' + str(self.id) + '/', '')
        file_path = os.path.normpath(self.dataset_path + os.sep + target_file)
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                logger.exception("Could not remove file: {0} {1}".format(file_path, e))
                raise
            self.file_num -= 1
            self.update_and_commit()

    def get_full_text(self, target_file):
        file_path = os.path.join(self.dataset_path, target_file)
        text = ds_util.get_text_sample(file_path)
        text = text.replace("\r", '')
        text = text.replace("\n", '<br>')
        return text

    def update_and_commit(self):
        self.updated_at = datetime.datetime.now()
        db.session.add(self)
        db.session.commit()
