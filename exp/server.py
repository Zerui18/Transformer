from .experiment import ExperimentConfig
from .manager import ExperimentManager
import flask

app = flask.Flask(__name__)
exp_manager = ExperimentManager()

@app.route('/queue', methods=['GET'])
def get_queued_experiments():
    return flask.jsonify({
        'success': True,
        'queued_experiments': [exp.get_dict_representation() for exp in exp_manager.queued_experiments]
    })

@app.route('/queue/create_and_append', methods=['GET'])
def create_and_append_experiment():
    name = flask.request.args.get('name')
    model_config_file = flask.request.args.get('model_config_file')
    dl_config_file = flask.request.args.get('dl_config_file')
    trainer_config_file = flask.request.args.get('trainer_config_file')
    try:
        config = ExperimentConfig(name, model_config_file, dl_config_file, trainer_config_file)
        exp_manager.create_and_append_experiment(name, config)
        return flask.jsonify({'success': True})
    except Exception as e:
        return flask.jsonify({'success': False, 'error': str(e)})

@app.route('queue/move', methods=['GET'])
def move_experiment_in_queue():
    try:
        src = int(flask.request.args.get('src'))
        dst = int(flask.request.args.get('dst'))
        exp_manager.move_experiment(src, dst)
        return flask.jsonify({'success': True})
    except Exception as e:
        return flask.jsonify({'success': False, 'error': str(e)})
    
@app.route('/queue/remove', methods=['GET'])
def remove_experiment_from_queue():
    try:
        index = int(flask.request.args.get('index'))
        exp_manager.remove_experiment(index)
        return flask.jsonify({'success': True})
    except Exception as e:
        return flask.jsonify({'success': False, 'error': str(e)})
    
@app.route('/queue/stop_all', methods=['GET'])
def stop_all_queued_experiments():
    try:
        exp_manager.stop_all()
        return flask.jsonify({'success': True})
    except Exception as e:
        return flask.jsonify({'success': False, 'error': str(e)})

@app.route('/queue/start_all', methods=['GET'])
def start_all_queued_experiments():
    try:
        exp_manager.start_all()
        return flask.jsonify({'success': True})
    except Exception as e:
        return flask.jsonify({'success': False, 'error': str(e)})

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