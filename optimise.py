from clearml import Task
from clearml.automation import UniformParameterRange, UniformIntegerParameterRange, DiscreteParameterRange
from clearml.automation import HyperParameterOptimizer
from clearml.automation.optuna import OptimizerOptuna
import argparse

task = Task.init(project_name='Nosyarlin', task_name='Hyper-Parameter Optimization', 
    task_type= Task.TaskTypes.optimizer)

parser = argparse.ArgumentParser(description='Process Command-line Arguments')
parser.add_argument('--task_id', default= None, action= 'store', help= 'Clearml Task ID that you want to optimise')

args = parser.parse_args([
        '--task_id', '9fe8c12745d54f34bb4df0304d18bd7f'])

optimizer = HyperParameterOptimizer(
    base_task_id= args.task_id,  
    
    # setting the hyper-parameters to optimize
    hyper_parameters=[
        UniformIntegerParameterRange('Args/batch_size', min_value=32, max_value=128, step_size=16),
        DiscreteParameterRange('Args/weight_decay', values = [0.025, 0.05]),
        UniformParameterRange('Args/lr', min_value=0.0005, max_value=0.005, step_size=0.0005),
    ],
    # setting the objective metric we want to maximize/minimize
    objective_metric_title='OverallVal',
    objective_metric_series='accuracy',
    objective_metric_sign='max',  

    # setting optimizer 
    optimizer_class=OptimizerOptuna,
    
    # Configuring optimization parameters
    execution_queue='default',  
    max_number_of_concurrent_tasks=2,  
    optimization_time_limit=30., 
    compute_time_limit=120, 
    total_max_jobs=10,  
    min_iteration_per_job=15000,  
    max_iteration_per_job=150000,  
)

optimizer.set_report_period(1) # setting the time gap between two consecutive reports
optimizer.start()
optimizer.wait() # wait until process is done
optimizer.stop() # make sure background optimization stopped