from werkzeug.contrib.profiler import ProfilerMiddleware
from main import app
import os
from gevent.wsgi import WSGIServer


app.config.from_envvar('CSLAIER_CONFIG')
cslaier_config_params = ('DATABASE_PATH', 'UPLOADED_RAW_FILE', 'UPLOADED_FILE', 'PREPARED_DATA',
                             'TRAINED_DATA', 'INSPECTION_TEMP', 'LOG_DIR')
# WebApp settings
app.config['CSLAIER_ROOT'] = os.getcwd()


def normalize_config_path():
    for param in cslaier_config_params:
        if not app.config[param].startswith('/'):
            app.config[param] = os.path.abspath(app.config['CSLAIER_ROOT'] + os.sep + app.config[param])

normalize_config_path()

app.config['PROFILE'] = True
app.wsgi_app = ProfilerMiddleware(app.wsgi_app)
app.debug = app.config['DEBUG']
server = WSGIServer((app.config['HOST'], app.config['PORT']), app)
server.serve_forever()
