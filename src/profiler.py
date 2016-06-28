from werkzeug.contrib.profiler import ProfilerMiddleware
from main import app
import os

app.config.from_envvar('DEEPSTATION_CONFIG')
deepstation_config_params = ('DATABASE_PATH', 'UPLOADED_RAW_FILE', 'UPLOADED_FILE', 'PREPARED_DATA',
                             'TRAINED_DATA', 'INSPECTION_TEMP', 'LOG_DIR')
# WebApp settings
app.config['DEEPSTATION_ROOT'] = os.getcwd()


def normalize_config_path():
    for param in deepstation_config_params:
        if not app.config[param].startswith('/'):
            app.config[param] = os.path.abspath(app.config['DEEPSTATION_ROOT'] + os.sep + app.config[param])

normalize_config_path()

app.config['PROFILE'] = True
app.wsgi_app = ProfilerMiddleware(app.wsgi_app)
app.run(
    host=app.config['HOST'],
    port=app.config['PORT'],
    debug=app.config['DEBUG'],
    use_evalex=False,
    threaded=True
)
