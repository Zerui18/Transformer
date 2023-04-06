from .experiment import ExperimentConfig
from .manager import ExperimentManager
import flask

app = flask.Flask(__name__)
exp_manager = ExperimentManager()

### Current Experiment ###

@app.route('/current', methods=['GET'])
def get_current_experiment():
    return flask.jsonify({
        'success': True,
        'current_experiment': exp_manager.current_experiment.get_dict_representation()
    })

@app.route('/current/stop', methods=['GET'])
def stop_current_experiment():
    try:
        exp_manager.stop_current_experiment()
        return flask.jsonify({'success': True})
    except Exception as e:
        return flask.jsonify({'success': False, 'error': str(e)})

### Queued Experiments ###

@app.route('/queue', methods=['GET'])
def get_queued_experiments():
    return flask.jsonify({
        'success': True,
        'queued_experiments': [exp.get_dict_representation() for exp in exp_manager.queued_experiments]
    })

@app.route('/queue/create_and_append', methods=['POST'])
def create_and_append_experiment():
    ''' Creates an experiment from the given configs and appends it to the queue.
    
        Form data:
            `name (str)`: The name of the experiment.
            `model (str)`: The model config.
            `dls (str)`: The data loaders config.
            `trainer (str)`: The trainer config.
            `resume_from_directory (str, optional)`: The directory to resume from.
            `resume_from_checkpoint (str, optional)`: The checkpoint to resume from.
    '''
    name = flask.request.form['name']
    # load configs
    configs = {}
    for key in ['model', 'dls', 'trainer']:
        if key in flask.request.files:
            # uploaded yaml file
            configs[key] = flask.request.files[key].read()
        elif key in flask.request.form:
            # raw yaml
            configs[key] = flask.request.form[key]
        else:
            return flask.jsonify({'success': False, 'error': f'No {key} config provided.'})
    # create experiment
    try:
        resume_dir = flask.request.form.get('resume_from_directory')
        resume_ckpt = flask.request.form.get('resume_from_checkpoint')
        config = ExperimentConfig(configs['dls'], configs['model'], configs['trainer'], resume_dir, resume_ckpt)
        exp_manager.create_and_append_experiment(name, config)
        return flask.jsonify({'success': True})
    except Exception as e:
        return flask.jsonify({'success': False, 'error': str(e)})

@app.route('queue/move', methods=['GET'])
def move_in_queue():
    try:
        src = int(flask.request.args.get('src'))
        dst = int(flask.request.args.get('dst'))
        exp_manager.move_in_queue(src, dst)
        return flask.jsonify({'success': True})
    except Exception as e:
        return flask.jsonify({'success': False, 'error': str(e)})

### Stopped Experiments ###

@app.route('/stopped', methods=['GET'])
def get_stopped_experiments():
    return flask.jsonify({
        'success': True,
        'stopped_experiments': [exp.get_dict_representation() for exp in exp_manager.stopped_experiments]
    })

@app.route('stopped/enqueue', methods=['GET'])
def enqueue_stopped():
    try:
        index = int(flask.request.args.get('index'))
        exp_manager.enqueue_stopped(index)
        return flask.jsonify({'success': True})
    except Exception as e:
        return flask.jsonify({'success': False, 'error': str(e)})

@app.route('/stopped/enqueue_all', methods=['GET'])
def enqueue_all_stopped():
    try:
        exp_manager.enqueue_all_stopped()
        return flask.jsonify({'success': True})
    except Exception as e:
        return flask.jsonify({'success': False, 'error': str(e)})

@app.route('/stopped/remove', methods=['GET'])
def remove_stopped():
    try:
        index = int(flask.request.args.get('index'))
        exp_manager.remove_stopped(index)
        return flask.jsonify({'success': True})
    except Exception as e:
        return flask.jsonify({'success': False, 'error': str(e)})

@app.route('/stopped/remove_all', methods=['GET'])
def remove_all_stopped():
    try:
        exp_manager.remove_all_stopped()
        return flask.jsonify({'success': True})
    except Exception as e:
        return flask.jsonify({'success': False, 'error': str(e)})

### Failed Experiments ###

@app.route('/failed', methods=['GET'])
def get_failed_experiments():
    return flask.jsonify({
        'success': True,
        'failed_experiments': [exp.get_dict_representation() for exp in exp_manager.failed_experiments]
    })

@app.route('/failed/remove', methods=['GET'])
def remove_failed():
    try:
        index = int(flask.request.args.get('index'))
        exp_manager.remove_failed(index)
        return flask.jsonify({'success': True})
    except Exception as e:
        return flask.jsonify({'success': False, 'error': str(e)})

@app.route('/failed/remove_all', methods=['GET'])
def remove_all_failed():
    try:
        exp_manager.remove_all_failed()
        return flask.jsonify({'success': True})
    except Exception as e:
        return flask.jsonify({'success': False, 'error': str(e)})