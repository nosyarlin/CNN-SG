from clearml import Task
from clearml.automation import UniformParameterRange, UniformIntegerParameterRange
from clearml.automation import HyperParameterOptimizer
from clearml.automation.optuna import OptimizerOptuna

task = Task.init(project_name='Nosyarlin', task_name='Hyper-Parameter Optimization', 
    task_type= Task.TaskTypes.optimizer)

optimizer = HyperParameterOptimizer(
    base_task_id= "8e7bc6643cd7485e9c67b6944584c769",  
    
    # setting the hyper-parameters to optimize
    hyper_parameters=[
        UniformIntegerParameterRange('batch_size', min_value=32, max_value=128, step_size=16),
        UniformParameterRange('weight_decay', min_value=0, max_value=0.01, step_size=0.005),
        UniformParameterRange('lr', min_value=0.0005, max_value=0.005, step_size=0.0005),
    ],
    # setting the objective metric we want to maximize/minimize
    objective_metric_title='accuracy',
    objective_metric_series='total',
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