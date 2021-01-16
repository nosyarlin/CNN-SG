from clearml import Task
from clearml.automation import UniformParameterRange, UniformIntegerParameterRange, DiscreteParameterRange
from clearml.automation import HyperParameterOptimizer
from clearml.automation.optuna import OptimizerOptuna
import optuna
import argparse
from datetime import date


def job_complete_callback(
    job_id,                 # type: str
    objective_value,        # type: float
    objective_iteration,    # type: int
    job_parameters,         # type: dict
    top_performance_job_id  # type: str
):
    print('Job completed!', job_id, objective_value,
          objective_iteration, job_parameters)
    if job_id == top_performance_job_id:
        print('WOOT WOOT we broke the record! Objective reached {}'.format(
            objective_value))


task = Task.init(project_name='Nosyarlin', task_name='Hyper-Parameter Optimization',
                 task_type=Task.TaskTypes.optimizer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process Command-line Arguments')
    parser.add_argument('--task_id', default=None, action='store',
                        help='Clearml Task ID that you want to optimise')

    args = parser.parse_args(
        # ['--task_id', '9fe8c12745d54f34bb4df0304d18bd7f']
    )

    # Get the template task experiment that we want to optimize if not already manually input
    if not args.task_id:
        args.task_id = Task.get_task(
            project_name='Nosyarlin', task_name='Train_' + date.today().strftime('%Y-%m-%d')).id

    optimizer = HyperParameterOptimizer(
        base_task_id=args.task_id,

        # setting the hyper-parameters to optimize
        hyper_parameters=[
            DiscreteParameterRange('Args/epochs', values=[15]),
            DiscreteParameterRange('Args/run_test', values=[False]),
            DiscreteParameterRange(
                'Args/archi', values=['inception', 'resnet50', 'mobilenet']),
            UniformIntegerParameterRange(
                'Args/batch_size', min_value=32, max_value=128, step_size=16),
            DiscreteParameterRange('Args/weight_decay',
                                   values=[0, 1e-4, 1e-5, 1e-6]),
            UniformParameterRange(
                'Args/lr', min_value=0.0005, max_value=0.005, step_size=0.0005),
            DiscreteParameterRange('Args/betadist_alpha',
                                   values=[0.8, 0.85, 0.9]),
            DiscreteParameterRange('Args/betadist_beta',
                                   values=[0.85, 0.9, 0.99]),
            DiscreteParameterRange('Args/eps', values=[1e-7, 1e-8, 1e-9]),
            UniformIntegerParameterRange(
                'Args/step_size', min_value=5, max_value=55, step_size=10),
            UniformParameterRange(
                'Args/gamma', min_value=0.05, max_value=0.2, step_size=0.05)
        ],
        # setting the objective metric we want to maximize/minimize
        objective_metric_title='Training',
        objective_metric_series='accuracy',
        objective_metric_sign='max',

        # setting optimizer
        optimizer_class=OptimizerOptuna,
        optuna_pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3, n_warmup_steps=0, interval_steps=1),

        # Configuring optimization parameters
        execution_queue='default',
        max_number_of_concurrent_tasks=2,
        # optimization_time_limit=60.,
        # compute_time_limit=120,
        total_max_jobs=20,
        min_iteration_per_job=15000,
        max_iteration_per_job=150000,
    )

    # setting the time gap between two consecutive reports
    optimizer.set_report_period(60)
    optimizer.start(job_complete_callback=job_complete_callback)
    optimizer.wait()  # wait until process is done
    top_exp = optimizer.get_top_experiments(top_k=5)
    print([t.id for t in top_exp])
    optimizer.stop()  # make sure background optimization stopped

    print("Optimisation completed")
