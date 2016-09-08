import shutil
import os


def remove_resume_file(base_path):
    try:
        shutil.rmtree(os.path.join(base_path, 'resume'))
    except OSError:
        pass
