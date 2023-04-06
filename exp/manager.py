from multiprocessing import Process, Value
from .experiment import Experiment, ExperimentState, ExperimentConfig

def run_experiment_child_process(name: str, state: Value, experiment_config: ExperimentConfig):
    ''' Entry point for the child process of an experiment. '''
    # configure torch
    import torch
    torch.set_float32_matmul_precision('high')
    # run experiment
    experiment = Experiment(name, state, experiment_config)
    experiment.init_resources()
    experiment.run()

def run_experiment(experiment: Experiment) -> Process:
    ''' Creates, starts, and returns a Process for the given experiment. '''
    process = Process(target=run_experiment_child_process, args=(experiment.name, experiment._state, experiment.config))
    # process.daemon = True
    process.start()
    return process

class ExperimentManager:

    _current_experiment: Experiment = None
    current_experiment_process: Process = None
    queued_experiments: list[Experiment] = []
    
    def __init__(self):
        pass

    ### Current Experiment ###
    @property
    def current_experiment(self) -> Experiment:
        return self._current_experiment
    
    @current_experiment.setter
    def current_experiment(self, experiment: Experiment or None):
        self._current_experiment = experiment
        if experiment is None:
            self._run_next_experiment_in_queue()
    
    def stop_current_experiment(self):
        removed_current_experiment = self._remove_current_experiment()
        self.append_experiment(removed_current_experiment) # requeue the experiment
    
    def _run_next_experiment_in_queue(self):
        for i in range(len(self.queued_experiments)):
            if self.queued_experiments[i].state == ExperimentState.QUEUING:
                self._set_current_experiment(i)
                break

    def _set_current_experiment(self, index: int):
        ''' Sets the current experiment to the experiment at the given index and runs it. '''
        assert (self.current_experiment is None) and (self.current_experiment_process is None), 'Cannot set current experiment while another experiment is running.'
        self.current_experiment = self.queued_experiments.pop(index)
        self.current_experiment_process = run_experiment(self.current_experiment)
    
    def _remove_current_experiment(self):
        ''' Removes and returns the current experiment and closes its process. '''
        assert (self.current_experiment is not None) and (self.current_experiment_process is not None), 'Cannot remove current experiment while no experiment is running.'
        current_experiment = self.current_experiment
        # request the experiment to stop
        current_experiment.stop()
        print('waiting for current experiment to stop...')
        # wait for the process to finish (5s timeout) before removing it
        if self.current_experiment_process.is_alive():
            self.current_experiment_process.join(timeout = 5.0)
        # if the process is still running, terminate it
        if self.current_experiment_process.is_alive():
            print('current experiment did not stop in time, terminating...')
            self.current_experiment_process.close()
        self.current_experiment_process = None
        self.current_experiment = None
        return current_experiment

    ### Queued Experiments ###
    def create_and_append_experiment(self, name: str, config: ExperimentConfig):
        state = Value('i', ExperimentState.QUEUING)
        experiment = Experiment(name, state, config)
        self.append_experiment(experiment)

    def append_experiment(self, experiment: Experiment):
        self.queued_experiments.append(experiment)
        if self.current_experiment is None:
            self._run_next_experiment_in_queue()

    def move_experiment(self, src_index: int, dst_index: int):
        self.queued_experiments.insert(dst_index, self.queued_experiments.pop(src_index))
    
    def remove_experiment(self, index: int):
        self.queued_experiments.pop(index)
    
    def start_all(self):
        for experiment in self.queued_experiments:
            experiment.start()
        if self.current_experiment is None:
            self._run_next_experiment_in_queue()
    
    def stop_all(self):
        for experiment in self.queued_experiments:
            experiment.stop()