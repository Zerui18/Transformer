from __future__ import annotations

import argparse
import math
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

import flask
import yaml

from .experiment import Experiment, ExperimentConfig, ExperimentState
from .machine import get_machine_state
from .manager import ExperimentManager
from .tb_events import read_scalars

# REST API + web UI for the ExperimentManager.
#
# Two ways to run it:
#   1. Standalone server (enqueue experiments through the UI / API):
#        uv run python -m exp.server --master-dir experiments/my-sweep
#   2. Embedded in a sweep script (replaces the trailing `input()`):
#        from exp.server import serve
#        serve(exp_manager)   # blocks; pass block=False to run in a background thread
#
# All endpoints return JSON of the shape {'success': bool, ...} with HTTP 200;
# mutation endpoints accept an optional 'name' alongside 'index' and refuse to act
# if the experiment at that index no longer matches (the queue changed underneath
# the UI between a render and a click).

def _latest_run(points: list[tuple[int, float, float]]) -> list[tuple[int, float, float]]:
	''' Keeps only the latest run of a scalar series: a step-counter reset means the experiment was restarted from scratch, so earlier points would zigzag the chart.

	Args:
		1. points: list[tuple[int, float, float]]  (step, wall_time, value) datapoints sorted by wall_time
	Returns: latest: list[tuple[int, float, float]]  the datapoints since the last step reset (resumed runs continue their step counter and are kept whole)
	'''
	out: list[tuple[int, float, float]] = []
	last_step = None
	for point in points:
		if last_step is not None and point[0] < last_step:
			out = []
		out.append(point)
		last_step = point[0]
	return out

def _downsample(points: list[tuple[int, float, float]], max_points: int) -> list[tuple[int, float, float]]:
	''' Downsamples a scalar series by striding, always keeping the first and last points.

	Args:
		1. points: list[tuple[int, float, float]]  (step, wall_time, value) datapoints
		2. max_points: int  [> 1] maximum number of points to keep
	Returns: sampled: list[tuple[int, float, float]]  the downsampled series
	'''
	if len(points) <= max_points:
		return points
	stride = (len(points) - 1) / (max_points - 1)
	sampled = [points[round(i * stride)] for i in range(max_points - 1)]
	sampled.append(points[-1])
	return sampled

def create_app(exp_manager: ExperimentManager, configs_directory: str = 'configs') -> flask.Flask:
	''' Creates the Flask app exposing the REST API and web UI for the given manager.

	Args:
		1. exp_manager: ExperimentManager  the manager to control (lives in this process)
		2. configs_directory: str  ['configs'] directory of config-set templates offered by the "new experiment" form
	Returns: app: flask.Flask  the configured app
	'''
	app = flask.Flask(__name__, static_folder=str(Path(__file__).parent / 'webui'), static_url_path='')
	app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024 # configs are small; cap uploads

	@app.errorhandler(Exception)
	def handle_exception(e: Exception) -> tuple[flask.Response, int]:
		''' Ensures every error surfaces as JSON the client can display, never an HTML 500 page. '''
		from werkzeug.exceptions import HTTPException
		if isinstance(e, HTTPException):
			return flask.jsonify({'success': False, 'error': f'{e.code} {e.name}: {e.description}'}), e.code
		return flask.jsonify({'success': False, 'error': f'{type(e).__name__}: {e}'}), 500

	def _ok(**payload) -> flask.Response:
		return flask.jsonify({'success': True, **payload})

	def _fail(error: str) -> flask.Response:
		return flask.jsonify({'success': False, 'error': error})

	def _body() -> dict[str, Any]:
		''' Returns the request payload from JSON or form data.

		Returns: body: dict  the merged request payload
		'''
		body = flask.request.get_json(silent=True)
		if body is None:
			body = flask.request.form.to_dict()
		return body

	def _buckets() -> dict[str, list[Experiment]]:
		return {
			'queued': exp_manager.queued_experiments,
			'stopped': exp_manager.stopped_experiments,
			'failed': exp_manager.failed_experiments,
			'completed': exp_manager.completed_experiments,
		}

	def _tracked_dir_names() -> set[str]:
		return {Path(exp.directory).name for exp in exp_manager.all_experiments()}

	def _untracked_disk_folders() -> list[dict[str, Any]]:
		''' Lists experiment folders on disk that are not tracked by the manager (eg, from before a restart).

		Returns: folders: list[dict]  [{name, mtime}] sorted by name
		'''
		master = Path(exp_manager.master_directory)
		if not master.is_dir():
			return []
		tracked = _tracked_dir_names()
		folders = []
		for entry in sorted(master.iterdir()):
			try:
				if entry.is_dir() and entry.name not in tracked and not entry.name.startswith('.'):
					folders.append({'name': entry.name, 'mtime': entry.stat().st_mtime})
			except OSError:
				continue # the folder vanished mid-scan (concurrent delete)
		return folders

	def _resolve_experiment(bucket: str, key: str) -> tuple[Experiment | None, Path | None, str | None]:
		''' Resolves a bucket + key to an experiment (tracked) and/or its directory.

		Args:
			1. bucket: str  one of current | queued | stopped | failed | completed | disk
			2. key: str  the index within the bucket, or the folder name for the disk bucket
		Returns: (experiment, directory, error): tuple  experiment is None for disk folders; directory is always set on success

		Honours an optional ?name= query arg: if given and the resolved experiment's name
		differs, resolution fails (the bucket shifted since the client rendered).
		'''
		expected_name = flask.request.args.get('name')
		if bucket == 'current':
			exp = exp_manager.current_experiment
			if exp is None:
				return None, None, 'No experiment is currently running.'
			if expected_name is not None and exp.name != expected_name:
				return None, None, 'The current experiment changed — refresh and retry.'
			return exp, Path(exp.directory), None
		if bucket == 'disk':
			master = Path(exp_manager.master_directory)
			# the key must be an actual directory entry — this also rules out path traversal
			if not master.is_dir() or key not in os.listdir(master):
				return None, None, f'No folder named {key!r} in the master directory.'
			if key in _tracked_dir_names():
				return None, None, f'{key!r} is tracked by the manager — use its queue actions instead.'
			return None, master / key, None
		buckets = _buckets()
		if bucket not in buckets:
			return None, None, f'Unknown bucket {bucket!r}.'
		try:
			index = int(key)
			exp = buckets[bucket][index]
		except (ValueError, IndexError):
			return None, None, f'No experiment at index {key!r} in {bucket}.'
		if expected_name is not None and exp.name != expected_name:
			return None, None, f'The {bucket} list changed — refresh and retry.'
		return exp, Path(exp.directory), None

	def _guarded(action: Callable[[int], None], bucket_name: str) -> flask.Response:
		''' Runs an index-based manager action with the optional stale-index name guard.

		Args:
			1. action: Callable[[int], None]  the manager method to invoke with the index
			2. bucket_name: str  which bucket the index refers to (for the name guard)
		Returns: response: flask.Response  the JSON API response
		'''
		body = _body()
		try:
			index = int(body['index'])
		except (KeyError, ValueError):
			return _fail("Missing or invalid 'index'.")
		expected_name = body.get('name')
		with exp_manager._lock:
			bucket = _buckets()[bucket_name]
			if not 0 <= index < len(bucket):
				return _fail(f'No experiment at index {index} in {bucket_name} — refresh and retry.')
			if expected_name is not None and bucket[index].name != expected_name:
				return _fail(f'The {bucket_name} queue changed — refresh and retry.')
			try:
				action(index)
			except Exception as e:
				return _fail(str(e))
		return _ok()

	### Web UI ###

	@app.get('/')
	def index() -> flask.Response:
		return app.send_static_file('index.html')

	### Status ###

	@app.get('/api/status')
	def get_status() -> flask.Response:
		with exp_manager._lock:
			current = exp_manager.current_experiment
			payload = {
				'server_time': time.time(),
				'master_directory': str(exp_manager.master_directory),
				'current': current.get_dict_representation() if current is not None else None,
				**{name: [exp.get_dict_representation() for exp in bucket] for name, bucket in _buckets().items()},
			}
		payload['disk'] = _untracked_disk_folders()
		payload['machine'] = get_machine_state(exp_manager.master_directory)
		return _ok(**payload)

	### Current Experiment ###

	@app.post('/api/current/stop')
	def stop_current() -> flask.Response:
		# the optional name guards against stopping a different experiment than the one
		# the user confirmed (the current slot may have advanced since their last poll)
		expected_name = _body().get('name')
		try:
			exp_manager.stop_current_experiment(expected_name=expected_name)
			return _ok()
		except Exception as e:
			return _fail(str(e))

	### Queued Experiments ###

	@app.post('/api/queue/create')
	def create_experiment() -> flask.Response:
		body = _body()
		# uploaded yaml files (curl multipart convenience) take precedence over inline strings
		raw = {}
		for key in ['model', 'dls', 'trainer']:
			if key in flask.request.files:
				raw[key] = flask.request.files[key].read()
			elif key in body:
				raw[key] = body[key]
			else:
				return _fail(f'No {key} config provided.')
		name = str(body.get('name', '')).strip()
		if not name:
			return _fail('No experiment name provided.')
		if any(c in name for c in '/\\\x00') or name in ('.', '..'):
			return _fail('Experiment names must be valid folder names.')
		resume_dir = str(body.get('resume_from_directory') or '').strip() or None
		resume_ckpt = str(body.get('resume_from_checkpoint') or '').strip() or None
		# validate resume inputs: the checkpoint must be a bare filename living under
		# <resume_dir>/checkpoints — never an arbitrary path (torch.load is not safe on
		# untrusted files)
		if resume_ckpt and not resume_dir:
			return _fail('resume_from_checkpoint requires resume_from_directory.')
		if resume_dir is not None:
			if not Path(resume_dir).is_dir():
				return _fail(f'Resume directory {resume_dir!r} does not exist.')
			if resume_ckpt is not None:
				if any(c in resume_ckpt for c in '/\\\x00') or resume_ckpt in ('.', '..'):
					return _fail('resume_from_checkpoint must be a bare checkpoint filename.')
				ckpt_path = Path(resume_dir) / 'checkpoints' / resume_ckpt
				if not ckpt_path.is_file():
					return _fail(f'Checkpoint {resume_ckpt!r} not found under {resume_dir}/checkpoints/.')
				resume_ckpt = str(ckpt_path)
		try:
			config = ExperimentConfig.from_yaml_strings(
				raw['model'], raw['dls'], raw['trainer'], resume_dir, resume_ckpt)
			exp_manager.create_and_append_experiment(name, config)
			return _ok()
		except Exception as e:
			return _fail(str(e))

	@app.post('/api/queue/move')
	def move_in_queue() -> flask.Response:
		body = _body()
		try:
			src, dst = int(body['src']), int(body['dst'])
		except (KeyError, ValueError):
			return _fail("Missing or invalid 'src'/'dst'.")
		expected_name = body.get('name')
		with exp_manager._lock:
			queue = exp_manager.queued_experiments
			if expected_name is not None and not (0 <= src < len(queue) and queue[src].name == expected_name):
				return _fail('The queue changed — refresh and retry.')
			try:
				exp_manager.move_in_queue(src, dst)
			except Exception as e:
				return _fail(str(e))
		return _ok()

	@app.post('/api/queue/stop')
	def stop_queued() -> flask.Response:
		return _guarded(exp_manager.stop_queued, 'queued')

	@app.post('/api/queue/stop_all')
	def stop_all_queued() -> flask.Response:
		exp_manager.stop_all_queued()
		return _ok()

	### Stopped Experiments ###

	@app.post('/api/stopped/enqueue')
	def enqueue_stopped() -> flask.Response:
		return _guarded(exp_manager.enqueue_stopped, 'stopped')

	@app.post('/api/stopped/enqueue_all')
	def enqueue_all_stopped() -> flask.Response:
		exp_manager.enqueue_all_stopped()
		return _ok()

	@app.post('/api/stopped/remove')
	def remove_stopped() -> flask.Response:
		return _guarded(exp_manager.remove_stopped, 'stopped')

	@app.post('/api/stopped/remove_all')
	def remove_all_stopped() -> flask.Response:
		exp_manager.remove_all_stopped()
		return _ok()

	### Failed Experiments ###

	@app.post('/api/failed/enqueue')
	def enqueue_failed() -> flask.Response:
		return _guarded(exp_manager.enqueue_failed, 'failed')

	@app.post('/api/failed/enqueue_all')
	def enqueue_all_failed() -> flask.Response:
		exp_manager.enqueue_all_failed()
		return _ok()

	@app.post('/api/failed/remove')
	def remove_failed() -> flask.Response:
		return _guarded(exp_manager.remove_failed, 'failed')

	@app.post('/api/failed/remove_all')
	def remove_all_failed() -> flask.Response:
		exp_manager.remove_all_failed()
		return _ok()

	### Completed Experiments ###

	@app.post('/api/completed/clear')
	def clear_completed() -> flask.Response:
		return _guarded(exp_manager.clear_completed, 'completed')

	@app.post('/api/completed/clear_all')
	def clear_all_completed() -> flask.Response:
		exp_manager.clear_all_completed()
		return _ok()

	### Experiment Details (any bucket, incl. current & untracked disk folders) ###

	@app.get('/api/experiments/<bucket>/<key>/yaml')
	def get_experiment_yaml(bucket: str, key: str) -> flask.Response:
		exp, directory, error = _resolve_experiment(bucket, key)
		if error is not None:
			return _fail(error)
		try:
			if exp is not None:
				configs = {
					'model': yaml.safe_dump(exp.config.model_config, sort_keys=False),
					'dls': yaml.safe_dump(exp.config.dls_config, sort_keys=False),
					'trainer': yaml.safe_dump(exp.config.trainer_config, sort_keys=False),
				}
				name = exp.name
			else:
				configs = {}
				for cfg in ['model', 'dls', 'trainer']:
					path = directory / f'{cfg}.yaml'
					configs[cfg] = path.read_text() if path.is_file() else ''
				name = directory.name
			return _ok(name=name, **configs)
		except Exception as e:
			return _fail(str(e))

	@app.get('/api/experiments/<bucket>/<key>/metrics')
	def get_experiment_metrics(bucket: str, key: str) -> flask.Response:
		exp, directory, error = _resolve_experiment(bucket, key)
		if error is not None:
			return _fail(error)
		max_points = flask.request.args.get('max_points', default=800, type=int)
		try:
			scalars = read_scalars(directory)
		except OSError as e:
			return _fail(str(e))
		metrics = {}
		for tag, points in scalars.items():
			# NaN/inf are not valid JSON — drop them rather than corrupting the response
			points = [p for p in points if math.isfinite(p[2])]
			points = _latest_run(points)
			if not points:
				continue
			points = _downsample(points, max(2, max_points))
			metrics[tag] = {
				'steps': [p[0] for p in points],
				'wall_times': [p[1] for p in points],
				'values': [p[2] for p in points],
			}
		return _ok(metrics=metrics)

	@app.get('/api/experiments/<bucket>/<key>/checkpoints')
	def get_experiment_checkpoints(bucket: str, key: str) -> flask.Response:
		exp, directory, error = _resolve_experiment(bucket, key)
		if error is not None:
			return _fail(error)
		ckpt_dir = directory / 'checkpoints'
		checkpoints = []
		if ckpt_dir.is_dir():
			for path in sorted(ckpt_dir.iterdir()):
				if path.is_file():
					stat = path.stat()
					checkpoints.append({'name': path.name, 'size': stat.st_size, 'mtime': stat.st_mtime})
		return _ok(checkpoints=checkpoints)

	### Untracked Disk Folders ###

	@app.post('/api/disk/resume')
	def resume_disk_folder() -> flask.Response:
		body = _body()
		name = str(body.get('name', ''))
		checkpoint = body.get('checkpoint') or None
		_, directory, error = _resolve_experiment('disk', name)
		if error is not None:
			return _fail(error)
		if checkpoint is not None:
			# must be a bare filename that exists under the folder's checkpoints dir
			if any(c in checkpoint for c in '/\\\x00') or checkpoint in ('.', '..') \
					or not (directory / 'checkpoints' / checkpoint).is_file():
				return _fail(f'Checkpoint {checkpoint!r} not found under {name}/checkpoints/.')
		# default to the trainer's rolling last.ckpt when none was chosen explicitly
		if checkpoint is None and (directory / 'checkpoints' / 'last.ckpt').is_file():
			checkpoint = 'last.ckpt'
		try:
			config = ExperimentConfig.resuming_from_directory(str(directory), checkpoint)
			exp_manager.create_and_append_experiment(name, config)
			return _ok()
		except Exception as e:
			return _fail(str(e))

	@app.post('/api/disk/delete')
	def delete_disk_folder() -> flask.Response:
		body = _body()
		name = str(body.get('name', ''))
		_, directory, error = _resolve_experiment('disk', name)
		if error is not None:
			return _fail(error)
		try:
			import shutil
			shutil.rmtree(directory)
			return _ok()
		except Exception as e:
			return _fail(str(e))

	### Config Templates ###

	@app.get('/api/templates')
	def list_templates() -> flask.Response:
		root = Path(configs_directory)
		if not root.is_dir():
			return _ok(templates=[])
		required = {'model.yaml', 'dls.yaml', 'trainer.yaml'}
		templates = sorted(entry.name for entry in root.iterdir()
							if entry.is_dir() and required <= {f.name for f in entry.iterdir()})
		return _ok(templates=templates)

	@app.get('/api/templates/<name>')
	def get_template(name: str) -> flask.Response:
		root = Path(configs_directory)
		if not root.is_dir() or name not in os.listdir(root):
			return _fail(f'No config template named {name!r}.')
		configs = {}
		for cfg in ['model', 'dls', 'trainer']:
			path = root / name / f'{cfg}.yaml'
			configs[cfg] = path.read_text() if path.is_file() else ''
		return _ok(name=name, **configs)

	return app

def serve(exp_manager: ExperimentManager, host: str = '127.0.0.1', port: int = 5001,
			configs_directory: str = 'configs', block: bool = True) -> threading.Thread | None:
	''' Starts the web UI server for the given manager.

	Args:
		1. exp_manager: ExperimentManager  the manager to expose (must live in this process)
		2. host: str  ['127.0.0.1'] bind address; use '0.0.0.0' to reach the UI from another machine
		3. port: int  [5001] bind port
		4. configs_directory: str  ['configs'] directory of config-set templates for the create form
		5. block: bool  [True] serve on this thread; if False, serve on a daemon thread and return it
	Returns: thread: threading.Thread | None  the server thread when block=False, else never returns

	Typical sweep-script usage (replaces the trailing `input()`):
		exp_manager = ExperimentManager('experiments/my-sweep')
		...enqueue experiments...
		serve(exp_manager)
	'''
	app = create_app(exp_manager, configs_directory)
	run = lambda: app.run(host=host, port=port, threaded=True, use_reloader=False)
	print(f'zlab experiment manager UI: http://{host}:{port}/')
	if block:
		try:
			run()
		except KeyboardInterrupt:
			pass
		finally:
			# terminate any running child so Ctrl-C doesn't hang in multiprocessing's atexit join
			print('shutting down: stopping scheduler and any running experiment...')
			exp_manager.shutdown()
		return None
	thread = threading.Thread(target=run, daemon=True, name='exp-webui-server')
	thread.start()
	return thread

def main():
	''' Standalone entrypoint: starts an empty manager and serves the UI (experiments are enqueued through the UI/API). '''
	parser = argparse.ArgumentParser(description='zlab experiment manager web UI')
	parser.add_argument('--master-dir', required=True, help='Directory where experiment folders are created')
	parser.add_argument('--host', default='127.0.0.1', help='Bind address (0.0.0.0 for remote access)')
	parser.add_argument('--port', type=int, default=5001, help='Bind port')
	parser.add_argument('--configs-dir', default='configs', help='Directory of config-set templates')
	args = parser.parse_args()
	exp_manager = ExperimentManager(args.master_dir)
	serve(exp_manager, host=args.host, port=args.port, configs_directory=args.configs_dir, block=True)

if __name__ == '__main__':
	main()
