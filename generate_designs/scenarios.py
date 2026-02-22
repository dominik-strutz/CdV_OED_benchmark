import numpy as np
from itertools import product
 
scenarios = []

def list2dict_combinations(list_of_lists):
    return [
        dict(
            study_area        = study_area,
            model_prior       = model_prior,
            velocity_model    = velocity_model,
            vel_sigma         = vel_sigma,
            noise_correlation = noise_correlation,
            drop_mean         = drop_mean,
            drop_gradient     = drop_gradient,
            optimisation      = optimisation,
            EIG_method        = EIG_method,
            EIG_N             = EIG_N,
            N_rec_max         = N_rec_max,
        ) for (study_area, model_prior, velocity_model, vel_sigma, noise_correlation,
               drop_mean, drop_gradient, optimisation, EIG_method, EIG_N, N_rec_max) in product(*list_of_lists)
    ]

####################################################################################################
# Compare N_EIG needed #############################################################################
####################################################################################################

comp_EIG_param_list = [
    ['full'],
    ['displacement'],
    ['gradient'],
    [0.05],
    [100.0],
    [0.0],
    [0.0],
    ['genetic', 'iterative'],
    ['NMC', 'DN'],
    [int(1e2), int(2e2), int(5e2), int(1e3), int(2e3),int(5e3)],
    [10],
]

scenarios += list2dict_combinations(comp_EIG_param_list)

comp_EIG_param_list = [
    ['shoulder'],
    ['uniform'],
    ['gradient'],
    [0.05],
    [100.0],
    [0.0],
    [0.0],
    ['genetic', 'iterative'],
    ['NMC', 'DN'],
    [int(1e2), int(2e2), int(5e2), int(1e3), int(2e3), int(5e3)],
    [10],
]
scenarios += list2dict_combinations(comp_EIG_param_list)

comp_EIG_param_list_att = [
    ['full',],
    ['displacement'],
    ['gradient'],
    [ 0.05],
    [100.0],
    [  0.35],
    [-30.0],
    ['genetic', 'iterative'],
    ['NMC', 'DN'],
    [int(1e2), int(2e2), int(5e2), int(1e3), int(2e3),int(5e3)],
    [10],
]
scenarios += list2dict_combinations(comp_EIG_param_list_att)

comp_EIG_param_list_att = [
    ['shoulder'],
    ['uniform'],
    ['gradient'],
    [ 0.05],
    [100.0],
    [  0.35],
    [-30.0],
    ['genetic', 'iterative'],
    ['NMC', 'DN'],
    [int(1e2), int(2e2), int(5e2), int(1e3), int(2e3),int(5e3)],
    [10],
]
scenarios += list2dict_combinations(comp_EIG_param_list_att)


# ####################################################################################################
# # Optimisation algorithm ##########################################################################
# ####################################################################################################

model_priors_param_list = [
    ['full', 'shoulder'],
    ['uniform', 'displacement'],
    ['homogeneous', 'gradient', 'heterogeneous'],
    [0.01, 0.05],
    [100.0],
    [0.0],
    [0.0],
    ['iterative'],
    ['NMC', 'DN'],
    [int(1e3)],
    [30],
]

scenarios += list2dict_combinations(model_priors_param_list)

model_priors_att_param_list = [
    ['full', 'shoulder'],
    ['uniform', 'displacement'],
    ['homogeneous', 'gradient', 'heterogeneous'],
    [0.01, 0.05],
    [100.0],
    [  0.35],
    [-30.0],
    ['iterative'],
    ['NMC', 'DN'],
    [int(1e3)],
    [30],
]
scenarios += list2dict_combinations(model_priors_att_param_list)

# ####################################################################################################
# # Compare Model Priors #############################################################################
# ####################################################################################################

model_priors_param_list = [
    ['full'],
    ['uniform', 'displacement'],
    ['gradient'],
    [0.05],
    [100.0],
    [0.0],
    [0.0],
    ['genetic', 'iterative'],
    ['NMC', 'DN'],
    [int(1e3)],
    [12],
]

scenarios += list2dict_combinations(model_priors_param_list)

model_priors_att_param_list = [
    ['full'],
    ['uniform', 'displacement'],
    ['gradient'],
    [ 0.05],
    [100.0],
    [  0.35],
    [-30.0],
    ['genetic', 'iterative'],
    ['NMC'],
    [int(1e3)],
    [12],
]
scenarios += list2dict_combinations(model_priors_att_param_list)

# ####################################################################################################
# # Compare Velocity Models ##########################################################################
# ####################################################################################################

velocity_models_param_list = [
    ['full', 'shoulder'],
    ['uniform', 'displacement'],
    ['homogeneous', 'gradient', 'heterogeneous'],
    [0.01, 0.05],
    [100.0],
    [0.0],
    [0.0],
    ['genetic', 'iterative'],
    ['NMC', 'DN'],
    [int(1e3)],
    [12],
]

scenarios += list2dict_combinations(velocity_models_param_list)

velocity_models_att_param_list = [
    ['full', 'shoulder'],
    ['uniform', 'displacement'],
    ['homogeneous', 'gradient', 'heterogeneous'],
    [0.01, 0.05],
    [100.0],
    [  0.35],
    [-30.0],
    ['genetic', 'iterative'],
    ['NMC'],
    [int(1e3)],
    [12],
]
 
scenarios += list2dict_combinations(velocity_models_att_param_list)   

# ####################################################################################################
# # Compare Vel Sigma ################################################################################
# ####################################################################################################

vel_sigma_param_list = [
    ['full', 'shoulder'],
    ['uniform', 'displacement'],
    ['homogeneous', 'gradient', 'heterogeneous'],
    [0.005, 0.01, 0.05, 0.1],
    [100.0],
    [0.0],
    [0.0],
    ['genetic', 'iterative'],
    ['NMC', 'DN'],
    [int(1e3)],
    [12],
]

scenarios += list2dict_combinations(vel_sigma_param_list)

vel_sigma_att_param_list = [
    ['full', 'shoulder'],
    ['uniform', 'displacement'],
    ['homogeneous', 'gradient', 'heterogeneous'],
    [0.005, 0.01, 0.05, 0.1],
    [100.0],
    [  0.35],
    [-30.0],
    ['genetic', 'iterative'],
    ['NMC'],
    [int(1e3)],
    [12],
]

scenarios += list2dict_combinations(vel_sigma_att_param_list)

# ####################################################################################################
# # Compare Noise Correlation #######################################################################
# ####################################################################################################

noise_correlation_param_list = [
    ['full', 'shoulder'],
    ['uniform', 'displacement'],
    ['gradient'],
    [0.05],
    [ 0.0, 50.0, 100.0, 200.0, 400.0],
    [0.0],
    [0.0],
    ['genetic', 'iterative'],
    ['NMC', 'DN'],
    [int(1e3)],
    [12],
]

scenarios += list2dict_combinations(noise_correlation_param_list)

noise_correlation_att_param_list = [
    ['full', 'shoulder'],
    ['uniform', 'displacement'],
    ['gradient'],
    [ 0.05],
    [ 0.0, 50.0, 100.0, 200.0, 400.0],
    [ 0.35],
    [-30.0],
    ['genetic', 'iterative'],
    ['NMC'],
    [int(1e3)],
    [12],
]
scenarios += list2dict_combinations(noise_correlation_att_param_list)

# ####################################################################################################
# # Compare Heuristic Attenuation ###################################################################
# ####################################################################################################

heuristic_attenuation_param_list = [
    ['full', 'shoulder'],
    ['uniform', 'displacement'],
    ['gradient'],
    [ 0.05],
    [ 100.0], 
    [  0.2,  0.35,    0.5],
    [-15.0, -30.0, -100.0],
    ['genetic', 'iterative'],
    ['NMC'],
    [int(1e3)],
    [12],
]
scenarios += list2dict_combinations(heuristic_attenuation_param_list)

# ####################################################################################################
# # Compare EIG Methods ##############################################################################
# ####################################################################################################

EIG_methods_param_list = [
    ['full', 'shoulder'],
    ['uniform', 'displacement'],
    ['heterogeneous'],
    [0.01],
    [100.0],
    [0.0],
    [0.0],
    ['genetic', 'iterative'],
    ['DN', 'NMC'], 
    [int(1e3)],
    [12],
]

scenarios += list2dict_combinations(EIG_methods_param_list)

EIG_methods_param_list = [
    ['full', 'shoulder'],
    ['uniform', 'displacement'],
    ['homogeneous', 'gradient',],
    [0.05],
    [100.0],
    [0.0],
    [0.0],
    ['genetic', 'iterative'],
    ['DN', 'NMC'], 
    [int(1e3)],
    [12],
]

scenarios += list2dict_combinations(EIG_methods_param_list)


# ####################################################################################################
# # Reference Designs ################################################################################
# ####################################################################################################

reference_designs_param_list = [
    ['full', 'shoulder'],
    ['uniform', 'displacement'],
    ['homogeneous', 'gradient', ],
    [0.05],
    [100.0],
    [0.0],
    [0.0],
    ['random', 'sobol'],
    ['NMC'], 
    [int(1e3)],
    [12],
]

scenarios += list2dict_combinations(reference_designs_param_list)

reference_designs_param_list = [
    ['full', 'shoulder'],
    ['uniform', 'displacement'],
    ['heterogeneous'],
    [0.01],
    [100.0],
    [0.0],
    [0.0],
    ['random', 'sobol'],
    ['NMC'], 
    [int(1e3)],
    [12],
]


scenarios += list2dict_combinations(reference_designs_param_list)

reference_designs_att_param_list = [
    ['full', 'shoulder'],
    ['uniform', 'displacement'],
    ['heterogeneous'],
    [0.01],
    [100.0],
    [0.35],
    [-30.0],
    ['random', 'sobol'],
    ['NMC'],
    [int(1e3)],
    [12],
]
scenarios += list2dict_combinations(reference_designs_att_param_list)

reference_designs_att_param_list = [
    ['full', 'shoulder'],
    ['uniform', 'displacement'],
    ['homogeneous', 'gradient'],
    [0.05],
    [100.0],
    [0.35],
    [-30.0],
    ['random', 'sobol'],
    ['NMC'],
    [int(1e3)],
    [12],
]
scenarios += list2dict_combinations(reference_designs_att_param_list)

####################################################################################################
# Remove Duplicates and Unwanted Scenarios ########################################################
####################################################################################################

if __name__ == '__main__':
    print(len(scenarios))

scenarios_tmp = []
for s in scenarios:
    if not (s['study_area'] == 'shoulder' and s['model_prior'] == 'displacement'):
        scenarios_tmp.append(s)
scenarios = scenarios_tmp

if __name__ == '__main__':
    print(len(scenarios))

# # only keep sceanrios with homogeneous velocity model
# scenarios_tmp = []
# for s in scenarios:
#     if s['velocity_model'] == 'homogeneous':
#         scenarios_tmp.append(s)
# scenarios = scenarios_tmp

# remove duplicates but keep order
# print(len(scenarios))
scenarios_tmp = []
for s in scenarios:
    if s not in scenarios_tmp:
        scenarios_tmp.append(s)
scenarios = scenarios_tmp

# convert all ints to str and all floats to str with precision 4
scenarios_tmp = []
for s in scenarios:
    s_tmp = s.copy()
    for k, v in s.items():
        if isinstance(v, int):
            s_tmp[k] = str(v)
        elif isinstance(v, float):
            s_tmp[k] = f'{v:.4f}'
    scenarios_tmp.append(s_tmp)
scenarios = scenarios_tmp

if __name__ == '__main__':
    print(len(scenarios))

    # for s in scenarios:
    #     print(s['study_area'], s['model_prior'], s['velocity_model'], s['vel_sigma'], s['noise_correlation'], s['drop_mean'], s['drop_gradient'], s['optimisation'], s['EIG_method'], s['EIG_N'], s['N_rec_max'])