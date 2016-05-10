# -*- coding: utf-8 -*-
import sys
import os
import random
import datetime
import pkg_resources
import subprocess
from xml.etree import ElementTree

import nkf

def get_python_version():
    v = sys.version_info
    return str(v[0]) + '.' + str(v[1]) + '.' + str(v[2])

def get_chainer_version():
    from chainer import __version__ as version
    return version

def get_disk_info():
    try:
        df = subprocess.check_output(['df', '-h'])
    except:
        return None
    disks = df[:-1].split('\n')
    titles = disks[0].split()
    filesystem_index = None
    mounted_on_index = None
    for i, title in enumerate(titles):
        if title.startswith('Filesystem'):
            filesystem_index = i
        elif title.startswith('Mounted'):
            mounted_on_index = i
    disk_info = []
    for disk in disks:
        row = disk.split()
        if row[filesystem_index].startswith('/'):
            st = os.statvfs(row[mounted_on_index])
            disk_info.append({
                'mount': row[mounted_on_index],
                'size': calculate_human_readable_filesize(st.f_frsize * st.f_blocks),
                'used': calculate_human_readable_filesize(st.f_frsize * (st.f_blocks-st.f_bfree)),
                'avail': calculate_human_readable_filesize(st.f_frsize * st.f_favail)
            })
    return disk_info

def get_gpu_info(nvidia_smi_cmd='nvidia-smi'):
    try:
        xml = subprocess.check_output([nvidia_smi_cmd, '-q', '-x'])
    except:
        return None
    ret = {}
    elem = ElementTree.fromstring(xml)
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
    ret_gpus.sort(cmp=lambda x,y: cmp(int(x['minor_number']), int(y['minor_number'])))
    ret['gpus'] = ret_gpus
    return ret

def get_system_info():
    return {
        'python_version': get_python_version(),
        'chainer_version': get_chainer_version(),
        'disk_info': get_disk_info(),
        'gpu_info': get_gpu_info()
    }

def is_module_available(module_name):
    for dist in pkg_resources.working_set:
        if dist.project_name.lower().find(module_name.lower()) > -1:
            return True
    return False

def get_timestamp():
    return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

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

def get_file_size_all(path):
    size = 0
    for f in find_all_files(path):
        size += os.path.getsize(f)
    return size

def calculate_human_readable_filesize(byte):
    if byte / 1024 < 1:
        return str(byte) + 'bytes'
    elif byte / (1024 ** 2) < 1:
        return str(byte / 1024) + 'k bytes'
    elif byte / (1024 ** 3) < 1:
        return str(byte / ( 1024 ** 2)) + 'M bytes'
    else:
        return str(byte / (1024 ** 3)) + 'G Bytes'

def count_files(path):
    ch = os.listdir(path)
    counter = 0
    for c in ch:
        if os.path.isdir(path + os.sep + c):
            counter += count_files(path + os.sep + c)
        else:
            counter += 1
    return counter

def get_files_in_random_order(path, num):
    """
    path配下の画像をランダムでnum枚取り出す。
    path配下がディレクトリしか無い場合はさらに配下のディレクトリから
    """
    children_files = []
    for cf in os.listdir(path):
        if os.path.isdir(path + os.sep + cf):
            if len(os.listdir(path + os.sep + cf)) != 0:
                children_files.append(cf)
        else:
            children_files.append(cf)
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
            files.append(f)
    return files;

def get_texts_in_random_order(path, num, character_num=-1):
    files = get_files_in_random_order(path, num)
    ret = []
    for f in files:
        if os.path.exists(f):
            ret.append(get_text_sample(f, character_num))
    return ret

def get_text_sample(path, character_num=-1):
    raw_text = open(path).read()
    encoding = nkf.guess(raw_text)
    text = raw_text.decode(encoding)
    if character_num > -1:
        return text[0:character_num]
    else:
        return text
