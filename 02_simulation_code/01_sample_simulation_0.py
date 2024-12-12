# This script runs a sample simulation.
import os
import numpy as np
import simulation_running_functions as fun

# %%
path_to_output = None ## REPLACE WITH DESIRED OUTPUT FOLDER
os.chdir(path_to_output)

### Specify nderlying distribution parameters
mu=0 #mean
sigma=1 #standard deviation 

## Specify gent information parameters
infotype='group_hist_upward' ## Must be in ['landscape', 'group_hist', 'own_hist','group_hist_upward']
infosummary='mean' ## Must be in ['mean', 'median']

## Specify simulation parameters
num_tstep=1000
num_agents=10
num_replicates=10
num_cores=5 ## Be careful not to request too many cores if using multiprocessing

verbose=True #print statements



## threshold parameters
num_thresholds=50
threshold_type='uniform' #must be in 'constant, 'uniform'
threshold_min=-3
threshold_max=6

## loop parameters
looped_over_param='corr' ## must be in 'corr','alpha' 
num_corr=1 #used to specify granularity of the vector

#evaluation parameters
reward_assessment='cumulative' ## Must be in ['cumulative', 'last']

potential_threshold_values=fun.create_potential_threshold_values(threshold_min,threshold_max,num_thresholds)

corr_values=np.linspace(start=0, stop=0, num=num_corr) 
alpha_values=[-32, -8, -4, -2, -1.5, -1, 0, 1, 1.5, 2, 4, 8, 32]

if looped_over_param=='corr':
    
    alpha=0
    len_looped_over=num_corr
    index_values=np.around(corr_values,3)
    
elif looped_over_param=='alpha':

    corr=0
    len_looped_over=len(alpha_values)
    index_values=np.around(alpha_values,3)


# run simulation
output_raw, exploit_frac_raw= fun.run_parallel_loop(len_looped_over,num_cores,looped_over_param,corr_values,alpha_values,potential_threshold_values,threshold_type,num_agents,num_replicates,num_tstep,mu,sigma,infotype,reward_assessment)

fun.save_output(output_raw,exploit_frac_raw,index_values,potential_threshold_values,infotype,infosummary, num_thresholds,num_tstep,num_agents,num_replicates,reward_assessment,threshold_type,threshold_min,threshold_max,looped_over_param)

