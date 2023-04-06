from multiprocessing import Process, Value, Array
from apscheduler.schedulers.background import BackgroundScheduler
from .experiment import Experiment, ExperimentState, ExperimentConfig

def run_experiment_child_process(name: str, state: Value, err_buffer: Value, experiment_config: ExperimentConfig):
    ''' Entry point for the child process of an experiment. '''
    # configure torch
    import torch
    torch.set_float32_matmul_precision('high')
    # run experiment
    experiment = Experiment(name, state, err_buffer, experiment_config)
    experiment.run()

def run_experiment(experiment: Experiment) -> Process:
    ''' Creates, starts, and returns a Process for the given experiment. '''
    process = Process(target=run_experiment_child_process, args=(experiment.name, experiment._state, experiment._err_buffer, experiment.config))
    # process.daemon = True
    process.start()
    return process

class ExperimentManager:
    ''' Mangages the deployment of experiments.
    
        Note: This class should only be used in the main process.

        Manages 4 queues:
            - `completed_experiments`: Experiments that have completed.
            - `queued_experiments`: Experiments that are queued to run.
            - `stopped_experiments`: Experiments that have been stopped.
            - `failed_experiments`: Experiments that have failed.
        
        And 1 current experiment:
            - `current_experiment`: The experiment that is currently running.
        
        The manager will automatically run the next experiment in `queued_experiments` when the current experiment completes, stops, or crashes.
        Each experiment is run in a new child process, which is closed when the experiment completes, stops, or crashes.
    '''

    def __init__(self):
        # start the current experiment checker
        self._current_exp_checker = BackgroundScheduler()
        self._current_exp_checker.add_job(self._check_current_experiment, 'interval', seconds=1)
        self._current_exp_checker.start()
    
    ### Completed Experiments ###
    completed_experiments: list[Experiment] = []

    ### Queued Experiments ###
    queued_experiments: list[Experiment] = []
    stopped_experiments: list[Experiment] = []
    failed_experiments: list[Experiment] = []

    ### Current Experiment ###
    _current_experiment: Experiment = None
    current_experiment_process: Process = None

    @property
    def current_experiment(self) -> Experiment:
        return self._current_experiment
    
    @current_experiment.setter
    def current_experiment(self, experiment: Experiment or None):
        self._current_experiment = experiment
        if experiment is None:
            self._run_next_experiment_in_queue()
    
    def stop_current_experiment(self):
        self.current_experiment.state = ExperimentState.STOPPED
        removed_current_experiment = self._remove_current_experiment()
        self.stopped_experiments.append(removed_current_experiment)
    
    def _run_next_experiment_in_queue(self):
        if len(self.queued_experiments) > 0:
            self._set_current_experiment(0)

    def _set_current_experiment(self, index: int):
        ''' Sets the current experiment to the experiment at the given index in the queue and runs it. '''
        assert (self.current_experiment is None) and (self.current_experiment_process is None), 'Cannot set current experiment while another experiment is running.'
        self.current_experiment = self.queued_experiments.pop(index)
        self.current_experiment_process = run_experiment(self.current_experiment)
    
    def _remove_current_experiment(self):
        ''' Removes and returns the current experiment and closes its process. '''
        assert (self.current_experiment is not None) and (self.current_experiment_process is not None), 'Cannot remove current experiment while no experiment is running.'
        current_experiment = self.current_experiment
        # wait for the process to finish (5s timeout) before removing it
        if self.current_experiment_process.is_alive():
            print('waiting for current experiment to stop...')
            self.current_experiment_process.join(timeout = 5.0)
        # if the process is still running, terminate it
        if self.current_experiment_process.is_alive():
            print('current experiment did not stop in time, terminating...')
            self.current_experiment_process.close()
        self.current_experiment_process = None
        self.current_experiment = None
        return current_experiment
    
    def _check_current_experiment(self):
        ''' Check if the current experiment has failed or completed and handles it accordingly. '''
        if self.current_experiment is not None:
            if self.current_experiment.state == ExperimentState.FAILED:
                print(f'Experiment {self.current_experiment.name} failed.')
                self._remove_current_experiment()
            elif self.current_experiment.state == ExperimentState.COMPLETED:
                print(f'Experiment {self.current_experiment.name} completed.')
                self._remove_current_experiment()

    ### Queued Experiments ###
    def create_and_append_experiment(self, name: str, config: ExperimentConfig):
        state = Value('i', ExperimentState.QUEUING)
        err_buffer = Array('c', 1024)
        experiment = Experiment(name, state, err_buffer, config)
        self.enqueue(experiment)

    def enqueue(self, experiment: Experiment):
        ''' Adds the given experiment to the queued experiments.
        
            This is the only way to add experiments to the queue.
        '''
        self.queued_experiments.append(experiment)
        if self.current_experiment is None:
            self._run_next_experiment_in_queue()

    def move_in_queue(self, src_index: int, dst_index: int):
        ''' Moves the experiment at the source index to the destination index in the queued experiments. '''
        self.queued_experiments.insert(dst_index, self.queued_experiments.pop(src_index))

    def enqueue_stopped(self, index: int):
        ''' Restarts the experiment at the given index in the stopped queued. '''
        experiment = self.stopped_experiments.pop(index)
        experiment.state = ExperimentState.QUEUING
        self.enqueue(experiment)
    
    def enqueue_all_stopped(self):
        ''' Restarts all experiments in the stopped queue. '''
        for experiment in self.stopped_experiments:
            experiment.state = ExperimentState.QUEUING
            self.enqueue(experiment)
        self.stopped_experiments = []
    
    def stop_queued(self, index: int):
        ''' Stops the experiment at the given index in the queued experiments. '''
        experiment = self.queued_experiments.pop(index)
        experiment.state = ExperimentState.STOPPED
        self.stopped_experiments.append(experiment)

    def stop_all_queued(self):
        ''' Stops all experiments in the queued experiments. '''
        for experiment in self.queued_experiments:
            experiment.state = ExperimentState.STOPPED
            self.stopped_experiments.append(experiment)
        self.queued_experiments = []
    
    def remove_stopped(self, index: int):
        ''' Removes the experiment at the given index in the stopped queue. '''
        self.stopped_experiments.pop(index).remove_exp_folder()
    
    def remove_all_stopped(self):
        ''' Removes all experiments in the stopped queue. '''
        for experiment in self.stopped_experiments:
            experiment.remove_exp_folder()
        self.stopped_experiments = []
    
    def remove_failed(self, index: int):
        ''' Removes the experiment at the given index in the failed queue. '''
        self.failed_experiments.pop(index).remove_exp_folder()
    
    def remove_all_failed(self):
        ''' Removes all experiments in the failed queue. '''
        for experiment in self.failed_experiments:
            experiment.remove_exp_folder()
        self.failed_experiments = []