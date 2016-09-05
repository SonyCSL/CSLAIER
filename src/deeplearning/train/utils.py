import shutil
import os


def remove_resume_file(base_path):
    print('remove_resume_file')
    try:
        shutil.rmtree(os.path.join(base_path, 'resume'))
    except OSError:
        pass
