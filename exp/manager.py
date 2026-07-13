from __future__ import annotations

import os
import threading
import time
from multiprocessing import Array, Process, Value

from apscheduler.schedulers.background import BackgroundScheduler

from .experiment import Experiment, ExperimentConfig, ExperimentState

def run_experiment_child_process(name: str, directory: str, state: Value, err_buffer: Array, progress: Array, experiment_config: ExperimentConfig):
    ''' Entry point for the child process of an experiment.

    Args:
        1. name: str  the experiment name
        2. directory: str  the experiment folder path
        3. state: Value  [int] shared ExperimentState
        4. err_buffer: Array  [bytes] shared error buffer
        5. progress: Array  [int32, (4,)] shared [epoch, batch, batches_per_epoch, global_step]
        6. experiment_config: ExperimentConfig  the experiment config
    '''
    # configure torch
    import torch
    torch.set_float32_matmul_precision('high')
    import pytorch_lightning
    pytorch_lightning.seed_everything(42)
    # run experiment
    experiment = Experiment(name, directory, state, err_buffer, experiment_config, progress)
    experiment.run()

def run_experiment(experiment: Experiment) -> Process:
    ''' Creates, starts, and returns a Process for the given experiment.

    Args:
        1. experiment: Experiment  the experiment to launch
    Returns: process: Process  the started child process
    '''
    process = Process(target=run_experiment_child_process,
                      args=(experiment.name, str(experiment.directory), experiment._state,
                            experiment._err_buffer, experiment._progress, experiment.config))
    process.start()
    return process

class ExperimentManager:
    ''' Manages the deployment of experiments.

    Properties:
        1. master_directory: str  the directory where all experiment folders are created
        2. single_process: bool  whether experiments run inline in this process (debugging) instead of child processes
        3. completed_experiments: list[Experiment]  experiments that completed successfully
        4. queued_experiments: list[Experiment]  experiments waiting to run
        5. stopped_experiments: list[Experiment]  experiments stopped by the user
        6. failed_experiments: list[Experiment]  experiments that crashed
        7. current_experiment: Experiment | None  the experiment currently running

    Note: This class should only be used in the main process. On platforms where
    multiprocessing uses the spawn start method (macOS/Windows), any script that
    enqueues experiments MUST wrap its top-level code in `if __name__ == '__main__':`
    — otherwise every child process re-executes the script at bootstrap and crashes.

    The manager automatically runs the next experiment in `queued_experiments` when the
    current experiment completes, stops, or crashes. Each experiment runs in a new child
    process, which is closed when the experiment finishes. All public methods are
    thread-safe (the REST API mutates the queues from Flask worker threads while a
    background scheduler advances the queue); the lock is never held while waiting on
    a child process, so the API stays responsive during stops.
    '''

    def __init__(self, master_directory: str, single_process: bool = False):
        self.single_process = single_process
        self.master_directory = str(master_directory)
        self.completed_experiments: list[Experiment] = []
        self.queued_experiments: list[Experiment] = []
        self.stopped_experiments: list[Experiment] = []
        self.failed_experiments: list[Experiment] = []
        self._current_experiment: Experiment | None = None
        self.current_experiment_process: Process | None = None
        # guards every queue/current mutation across Flask threads + the scheduler thread
        self._lock = threading.RLock()
        # when the current experiment reached a terminal state but its process is still alive
        self._terminal_since: float | None = None
        if self.single_process:
            # configure torch
            import torch
            torch.set_float32_matmul_precision('high')
        else:
            # start the current experiment checker
            self._current_exp_checker = BackgroundScheduler()
            self._current_exp_checker.add_job(self._check_current_experiment, 'interval', seconds=1)
            self._current_exp_checker.start()

    ### Current Experiment ###

    @property
    def current_experiment(self) -> Experiment | None:
        return self._current_experiment

    @current_experiment.setter
    def current_experiment(self, experiment: Experiment | None):
        with self._lock:
            self._current_experiment = experiment
            if experiment is None:
                self._run_next_experiment_in_queue()

    def stop_current_experiment(self, expected_name: str | None = None):
        ''' Stops the currently running experiment and moves it to the stopped queue.

        Args:
            1. expected_name: str | None  [None] if given, refuse to stop unless the current experiment still has this name (guards a UI acting on stale state)

        Blocks until the child process exits (join, then SIGTERM, then SIGKILL — each
        with a 5s grace period) but never holds the manager lock while waiting, so
        other API calls stay responsive.
        '''
        with self._lock:
            experiment = self._current_experiment
            if experiment is None:
                raise RuntimeError('No experiment is currently running.')
            if expected_name is not None and experiment.name != expected_name:
                raise RuntimeError('The current experiment changed — refresh and retry.')
            # compare-and-set under the shared value's own lock: never overwrite a
            # terminal state the child has already reached (a completed run must not
            # be filed as stopped)
            with experiment._state.get_lock():
                if experiment._state.value not in (ExperimentState.QUEUING, ExperimentState.RUNNING):
                    raise RuntimeError('The current experiment has already finished — it is being filed automatically.')
                experiment._state.value = ExperimentState.STOPPED
            process = self.current_experiment_process
        # wait outside the lock
        self._wait_for_exit(process)
        with self._lock:
            # another thread (a concurrent stop, or the checker's STOPPED branch) may
            # have finalized it while we were waiting
            if self._current_experiment is experiment:
                self._finalize_current(self.stopped_experiments)

    def _wait_for_exit(self, process: Process | None):
        ''' Waits for a child process to exit, escalating join -> SIGTERM -> SIGKILL with 5s grace periods.

        Note: must NOT be called while holding the manager lock.
        '''
        if process is None:
            return
        if process.is_alive():
            print('waiting for current experiment to stop...')
            process.join(timeout=5.0)
        if process.is_alive():
            # pytorch_lightning traps SIGTERM and only stops at the next loop boundary,
            # so give it another grace period before the untrappable SIGKILL
            print('current experiment did not stop in time, terminating...')
            process.terminate()
            process.join(timeout=5.0)
        if process.is_alive():
            print('current experiment ignored SIGTERM, killing...')
            process.kill()
            process.join(timeout=5.0)

    def _finalize_current(self, bucket: list[Experiment]):
        ''' Files the current experiment into the given bucket, closes its (exited) process, and advances the queue.

        Note: must be called while holding the manager lock.
        '''
        experiment = self._current_experiment
        experiment.finished_at = time.time()
        process = self.current_experiment_process
        # close() raises if the process is somehow still alive — in that case just
        # drop the handle; the OS reaps it when it exits
        if process is not None and not process.is_alive():
            process.close()
        self.current_experiment_process = None
        self._terminal_since = None
        bucket.append(experiment)
        self.current_experiment = None # property setter advances the queue

    def _run_next_experiment_in_queue(self):
        if len(self.queued_experiments) > 0:
            self._set_current_experiment(0)

    def _set_current_experiment(self, index: int):
        ''' Sets the current experiment to the experiment at the given index in the queue and runs it. '''
        assert (self.current_experiment is None) and (self.current_experiment_process is None), 'Cannot set current experiment while another experiment is running.'
        self.current_experiment = self.queued_experiments.pop(index)
        self.current_experiment.launched_at = time.time()
        self.current_experiment.finished_at = None
        self._terminal_since = None
        if self.single_process:
            try:
                self.current_experiment.run()
            except Exception as e:
                # run() has already recorded FAILED + err_buffer; keep the sweep going
                print(f'Experiment {self.current_experiment.name} failed: {e}')
            # since we're not using self._current_exp_checker
            # manually check if the experiment completed, stopped, or failed
            # and move it to the appropriate queue
            self.current_experiment.finished_at = time.time()
            if self.current_experiment.state == ExperimentState.COMPLETED:
                self.completed_experiments.append(self.current_experiment)
            elif self.current_experiment.state == ExperimentState.STOPPED:
                self.stopped_experiments.append(self.current_experiment)
            else:
                # FAILED, or died before reaching a terminal state
                self.failed_experiments.append(self.current_experiment)
            # remove the current experiment
            self.current_experiment = None
        else:
            try:
                self.current_experiment_process = run_experiment(self.current_experiment)
            except Exception as e:
                # Process.start() itself failed (fork resources, spawn pickling, ...)
                experiment = self._current_experiment
                experiment.state = ExperimentState.FAILED
                experiment.err_buffer = f'Failed to launch child process: {type(e).__name__}: {e}'
                experiment.finished_at = time.time()
                self.failed_experiments.append(experiment)
                self._current_experiment = None
                self._run_next_experiment_in_queue()

    def _check_current_experiment(self):
        ''' Checks whether the current experiment reached a terminal state and files it once its process has exited. '''
        with self._lock:
            experiment = self.current_experiment
            process = self.current_experiment_process
            if experiment is None or process is None:
                return
            state = experiment.state
            process_dead = not process.is_alive()
            if state == ExperimentState.FAILED and process_dead:
                print(f'Experiment {experiment.name} failed.')
                print(experiment)
                self._finalize_current(self.failed_experiments)
            elif state == ExperimentState.COMPLETED and process_dead:
                print(f'Experiment {experiment.name} completed.')
                self._finalize_current(self.completed_experiments)
            elif state == ExperimentState.STOPPED and process_dead:
                # a stop request whose caller never finished finalizing (eg, the server
                # thread died mid-stop) — finish filing it so the queue advances
                print(f'Experiment {experiment.name} stopped.')
                self._finalize_current(self.stopped_experiments)
            elif state in (ExperimentState.RUNNING, ExperimentState.QUEUING) and process_dead:
                # the child process died without reporting (eg, OOM-killed / segfault)
                print(f'Experiment {experiment.name} died without reporting a state.')
                experiment.state = ExperimentState.FAILED
                if not experiment.err_buffer:
                    experiment.err_buffer = 'Child process died unexpectedly (killed or crashed without a Python exception).'
                self._finalize_current(self.failed_experiments)
            elif state in (ExperimentState.FAILED, ExperimentState.COMPLETED, ExperimentState.STOPPED):
                # terminal state but the process is still tearing down — give it a
                # minute (checkpoint flushing, dataloader workers) before SIGKILL
                if self._terminal_since is None:
                    self._terminal_since = time.time()
                elif time.time() - self._terminal_since > 60.0:
                    print(f'Experiment {experiment.name} finished but its process did not exit, killing...')
                    process.kill()

    def shutdown(self):
        ''' Stops the scheduler and terminates any running child process (call on Ctrl-C / server exit). '''
        if not self.single_process:
            self._current_exp_checker.shutdown(wait=False)
        with self._lock:
            process = self.current_experiment_process
        if process is not None:
            try:
                if process.is_alive():
                    process.terminate()
                self._wait_for_exit(process)
            except ValueError:
                pass # process handle already closed by a concurrent finalize

    ### Queued Experiments ###

    def all_experiments(self) -> list[Experiment]:
        ''' Returns every experiment tracked by the manager (current + all queues).

        Returns: experiments: list[Experiment]  all tracked experiments
        '''
        with self._lock:
            current = [self.current_experiment] if self.current_experiment is not None else []
            return current + self.queued_experiments + self.stopped_experiments + self.failed_experiments + self.completed_experiments

    def create_and_append_experiment(self, name: str, config: ExperimentConfig):
        ''' Creates an Experiment from the given config and appends it to the queue.

        Args:
            1. name: str  the experiment name, must be unique among tracked experiments
            2. config: ExperimentConfig  the experiment config
        '''
        with self._lock:
            if any(exp.name == name for exp in self.all_experiments()):
                raise ValueError(f'An experiment named {name!r} already exists in the manager.')
            state = Value('i', ExperimentState.QUEUING)
            err_buffer = Array('c', 4096)
            progress = Array('i', 4)
            if config.resume_from_directory is not None:
                directory = str(config.resume_from_directory)
            else:
                directory = os.path.join(self.master_directory, name)
                if os.path.isdir(directory) and len(os.listdir(directory)) > 0:
                    raise ValueError(f'Experiment folder {directory!r} already exists and is not empty. '
                                     f'Pick a new name, or resume/delete the existing folder.')
                os.makedirs(directory, exist_ok=True)
            experiment = Experiment(name, directory, state, err_buffer, config, progress)
            self.enqueue(experiment)

    def enqueue(self, experiment: Experiment):
        ''' Adds the given experiment to the queued experiments.

        This is the only way to add experiments to the queue.
        '''
        with self._lock:
            self.queued_experiments.append(experiment)
            if self.current_experiment is None:
                self._run_next_experiment_in_queue()

    def move_in_queue(self, src_index: int, dst_index: int):
        ''' Moves the experiment at the source index to the destination index in the queued experiments. '''
        with self._lock:
            n = len(self.queued_experiments)
            if not (0 <= src_index < n) or not (0 <= dst_index < n):
                raise IndexError(f'Invalid move {src_index} -> {dst_index} for a queue of {n}.')
            self.queued_experiments.insert(dst_index, self.queued_experiments.pop(src_index))

    def stop_queued(self, index: int):
        ''' Stops the experiment at the given index in the queued experiments. '''
        with self._lock:
            experiment = self.queued_experiments.pop(index)
            experiment.state = ExperimentState.STOPPED
            self.stopped_experiments.append(experiment)

    def stop_all_queued(self):
        ''' Stops all experiments in the queued experiments. '''
        with self._lock:
            for experiment in self.queued_experiments:
                experiment.state = ExperimentState.STOPPED
                self.stopped_experiments.append(experiment)
            self.queued_experiments = []

    ### Stopped Experiments ###

    def enqueue_stopped(self, index: int):
        ''' Requeues the experiment at the given index in the stopped queue. '''
        with self._lock:
            experiment = self.stopped_experiments.pop(index)
            experiment.state = ExperimentState.QUEUING
            self.enqueue(experiment)

    def enqueue_all_stopped(self):
        ''' Requeues all experiments in the stopped queue. '''
        with self._lock:
            stopped, self.stopped_experiments = self.stopped_experiments, []
            for experiment in stopped:
                experiment.state = ExperimentState.QUEUING
                self.enqueue(experiment)

    def remove_stopped(self, index: int):
        ''' Removes the experiment at the given index in the stopped queue and deletes its folder. '''
        with self._lock:
            self.stopped_experiments.pop(index).remove_exp_folder()

    def remove_all_stopped(self):
        ''' Removes all experiments in the stopped queue and deletes their folders. '''
        with self._lock:
            for experiment in self.stopped_experiments:
                experiment.remove_exp_folder()
            self.stopped_experiments = []

    ### Failed Experiments ###

    def enqueue_failed(self, index: int):
        ''' Requeues (retries) the experiment at the given index in the failed queue. '''
        with self._lock:
            experiment = self.failed_experiments.pop(index)
            experiment.state = ExperimentState.QUEUING
            self.enqueue(experiment)

    def enqueue_all_failed(self):
        ''' Requeues (retries) all experiments in the failed queue. '''
        with self._lock:
            failed, self.failed_experiments = self.failed_experiments, []
            for experiment in failed:
                experiment.state = ExperimentState.QUEUING
                self.enqueue(experiment)

    def remove_failed(self, index: int):
        ''' Removes the experiment at the given index in the failed queue and deletes its folder. '''
        with self._lock:
            self.failed_experiments.pop(index).remove_exp_folder()

    def remove_all_failed(self):
        ''' Removes all experiments in the failed queue and deletes their folders. '''
        with self._lock:
            for experiment in self.failed_experiments:
                experiment.remove_exp_folder()
            self.failed_experiments = []

    ### Completed Experiments ###

    def clear_completed(self, index: int):
        ''' Removes the experiment at the given index from the completed list (keeps its folder on disk). '''
        with self._lock:
            self.completed_experiments.pop(index)

    def clear_all_completed(self):
        ''' Removes all experiments from the completed list (keeps their folders on disk). '''
        with self._lock:
            self.completed_experiments = []
