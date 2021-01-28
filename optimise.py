from clearml import Task
from clearml.automation import UniformParameterRange, UniformIntegerParameterRange, DiscreteParameterRange, ParameterSet
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


task = Task.init(project_name='Nosyarlin', task_name='Hyper-Parameter Optimization ' + date.today().strftime('%Y-%m-%d'),
                 task_type=Task.TaskTypes.optimizer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process Command-line Arguments')
    parser.add_argument('--task_id', default=None, action='store',
                        help='Clearml Task ID that you want to optimise')

    args = parser.parse_args(
        # ['--task_id', '160f736150454ce1b4290afe1b221fd9']
    )

    # Get the template task experiment that we want to optimize if not already manually input
    if not args.task_id:
        args.task_id = Task.get_task(
            project_name='Nosyarlin', task_name='Train_' + date.today().strftime('%Y-%m-%d')).id

    optimizer = HyperParameterOptimizer(
        base_task_id=args.task_id,

        # setting the hyper-parameters to optimize
        hyper_parameters=[
            DiscreteParameterRange(
                'Args/epochs', values=[10]),
            DiscreteParameterRange(
                'Args/skip_test', values=[True]),
            DiscreteParameterRange(
                'Args/archi', values=['inception', 'resnet50']),
            DiscreteParameterRange(
                'Args/weight_decay', values=[1e-5, 1e-6, 1e-7]),
            DiscreteParameterRange(
                'Args/lr', values=[0.0005, 0.001]),
            DiscreteParameterRange(
                'Args/dropout', values=[0.1, 0.2, 0.3]),
            # ParameterSet(
            #     parameter_combinations=[
            #         {'Args/betadist_alpha':0.9, 'Args/betadist_beta':0.99},
            #         {'Args/betadist_alpha':0.8, 'Args/betadist_beta':0.9}]),
            DiscreteParameterRange(
                'Args/eps', values=[1e-8]),
            DiscreteParameterRange(
                'Args/gamma', values=[0.1]),
            DiscreteParameterRange(
                'Args/path_to_save_results', values=['E:/JoejynDocuments/CNN_Animal_ID/Nosyarlin/SBWR_BTNR_CCNR/Results/Test/'])
        ],
        # setting the objective metric we want to maximize/minimize
        objective_metric_title='Training and Validation',
        objective_metric_series='Val accuracy',
        objective_metric_sign='min',

        # setting optimizer
        optuna_pruner=optuna.pruners.HyperbandPruner(),
        optimizer_class=OptimizerOptuna,

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
