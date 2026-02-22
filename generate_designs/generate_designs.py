import os
import time
import sys
import argparse
from typing import Dict, Any, List, Optional

import torch
import numpy as np
import zuko
import pandas as pd
import pygad
from tqdm.auto import tqdm

from geobed.utils.sample_distribution import SampleDistribution
from geobed import BED_base_explicit, BED_base_nuisance

# Add current directory to path
sys.path.append(os.getcwd())

from helpers.geographic_setup import (
    design_space_full, design_space_shoulder,
    topo_data,
)    

from helpers.helper_functions import (
    concave_hull2D_prior_dist_constructor,
)

from helpers.likelihood import (
    logistic_picking_likelihood_tt,
    DataLikelihoodAttenuation, DataLikelihood,
    # DataLikelihoodPLArrival, DataLikelihoodArrival,
)

from helpers.forward import TTLookup
import ast

def parse_arguments() -> Dict[str, Any]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run scenario with given parameters.')
    parser.add_argument('--study_area', type=str, required=False, help='Study area')
    parser.add_argument('--model_prior', type=str, required=False, help='Model prior')
    parser.add_argument('--velocity_model', type=str, required=False, help='Velocity model')
    parser.add_argument('--vel_sigma', type=float, required=False, help='Velocity model sigma')
    parser.add_argument('--noise_correlation', type=float, required=False, help='Noise correlation')
    parser.add_argument('--drop_mean', type=float, required=False, help='Drop mean')
    parser.add_argument('--drop_gradient', type=float, required=False, help='Drop gradient')
    parser.add_argument('--optimisation', type=str, required=False, help='Optimisation method')
    parser.add_argument('--EIG_method', type=str, required=False, help='EIG method')
    parser.add_argument('--EIG_N', type=int, required=False, help='Number of samples for EIG')
    parser.add_argument('--N_rec_max', type=int, required=False, help='Number of receivers')


    args = parser.parse_args(sys.argv[1:])
    return vars(args)



def print_scenario(scen: Dict[str, Any]) -> None:
    """Print scenario parameters."""
    print("="*60)
    print("="*60)
    print("Scenario")
    print("="*60)
    print(f"Running scenario: {scen['study_area']}")
    print(f"Model prior: {scen['model_prior']}")
    print(f"Velocity model: {scen['velocity_model']}")
    print(f"Velocity model sigma: {scen['vel_sigma']}")
    print(f"Noise correlation: {scen['noise_correlation']}")
    print(f"Drop mean: {scen['drop_mean']}")
    print(f"Drop gradient: {scen['drop_gradient']}")
    print(f"Optimisation: {scen['optimisation']}")
    print(f"EIG method: {scen['EIG_method']}")
    print(f"EIG_N: {scen['EIG_N']}")
    print(f"Number of receivers: {scen['N_rec_max']} ")
    print("="*60)
    print()
    print()


def setup_design_data(base_path: str) -> pd.DataFrame:
    """Setup design data CSV file."""
    csv_filename = f'{base_path}/design_data.csv'
    if not os.path.exists(csv_filename):
        design_data = pd.DataFrame(
            data = None,
            columns = [
                'study_area', 'model_prior', 'velocity_model',
                'vel_sigma', 'noise_correlation',
                'drop_mean', 'drop_gradient',
                'optimisation',
                'EIG_method',
                'N_rec',
                'design',
                'EIG',
                'EIG_N',
                'EIG_ref',
                'runtime',
                'EIG_candidates',
            ],
        )
        design_data.to_csv(csv_filename, index=False)
    
    # Load existing design data
    return pd.read_csv(csv_filename)


def is_scenario_in_design_data(design_data: pd.DataFrame, scen: Dict[str, Any], N_rec: int) -> bool:
    
    design_data = pd.DataFrame(design_data, columns=[
        'study_area', 'model_prior', 'velocity_model',
        'vel_sigma', 'noise_correlation',
        'drop_mean', 'drop_gradient',
        'optimisation', 'EIG_method', 'EIG_N', 'N_rec',
    ])

    
    
    """Check if a scenario is already in the design data."""
    return ((design_data['study_area'] == scen['study_area']) &
            (design_data['model_prior'] == scen['model_prior']) &
            (design_data['velocity_model'] == scen['velocity_model']) &
            (design_data['vel_sigma'] == scen['vel_sigma']) &
            (design_data['noise_correlation'] == scen['noise_correlation']) &
            (design_data['drop_mean'] == scen['drop_mean']) &
            (design_data['drop_gradient'] == scen['drop_gradient']) &
            (design_data['optimisation'] == scen['optimisation']) &
            (design_data['EIG_method'] == scen['EIG_method']) &
            (design_data['EIG_N'] == scen['EIG_N']) &
            (design_data['N_rec'] == N_rec)).any()


def setup_environment() -> None:
    """Setup environment variables for better performance."""
    # Need to set this environment variable to avoid file locking issues with h5py on the cluster
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
    
    # Setting thread count to 1 to avoid issues with multiprocessing on the cluster
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def get_design_space(study_area: str) -> torch.Tensor:
    """Get design space based on study area."""
    if study_area == 'full':
        return design_space_full
    elif study_area == 'shoulder':
        return design_space_shoulder
    else:
        raise ValueError("Study area not implemented")


def get_model_prior_samples(study_area: str, model_prior: str) -> torch.Tensor:
    """Get model prior samples based on study area and model prior."""
    if model_prior == 'uniform':
        if study_area == 'full':
            return torch.load('data/priors/prior_samples_full_uniform.pt')
        elif study_area == 'shoulder':
            return torch.load('data/priors/prior_samples_shoulder_uniform.pt')
            
    elif model_prior == 'displacement':
        if study_area == 'shoulder':
            raise ValueError("Displacement prior not implemented for shoulder area")
        elif study_area == 'full':
            return torch.load('data/priors/prior_samples_full_disp.pt')
    else:
        raise ValueError("Model prior not implemented. Choose 'uniform' or 'displacement'")


def get_forward_function(velocity_model: str, model_prior: str, study_area: str, model_prior_samples: torch.Tensor) -> TTLookup:
    """Get forward function based on velocity model."""
    if velocity_model in ['homogeneous', 'gradient', 'heterogeneous']:
        mp_string = {
            'uniform': 'uniform',
            'displacement': 'disp',
        }[model_prior]
        
        return TTLookup(
            model_prior_samples, design_space_full,
            torch.load(f"data/data_lookup/{velocity_model}_{study_area}_{mp_string}.pt"),
        )
    else:
        raise ValueError("Velocity model not implemented. Choose 'homogeneous', 'gradient' or 'heterogeneous'")


def setup_data_likelihood(forward_function: TTLookup, scen: Dict[str, Any]) -> tuple:
    """Setup data likelihood and nuisance distribution."""

    if (scen['drop_mean'] != 0) and (scen['drop_gradient'] != 0):
        picking_likelihood = logistic_picking_likelihood_tt(
            b = scen['drop_gradient'],
            c = scen['drop_mean'],
        )
    
        data_likelihood = DataLikelihoodAttenuation(
            forward_function=forward_function,
            picking_likelihood=picking_likelihood,
            dependence_distance=scen['noise_correlation'],
            vel_sigma=scen['vel_sigma'],
            tt_obs_std=0.01, 
        )
        
        nuisance_dist = zuko.distributions.BoxUniform(
                lower=torch.tensor([0.0,]),
                upper=torch.tensor([1.0,]),
            )        
    else:
        data_likelihood = DataLikelihood(
            forward_function=forward_function,
            dependence_distance=scen['noise_correlation'],
            vel_sigma=scen['vel_sigma'],
            tt_obs_std=0.01, 
        )
        nuisance_dist = None
        
    return data_likelihood, nuisance_dist


def setup_bed_class(data_likelihood, model_prior_samples: torch.Tensor, nuisance_dist: Optional[zuko.distributions.BoxUniform]) -> Any:
    """Setup BED class based on whether we have nuisance parameters."""
    model_prior_sample_dist = SampleDistribution(model_prior_samples)
    
    if nuisance_dist is not None:
        return BED_base_nuisance(
            data_likelihood_func=data_likelihood,
            m_prior_dist=model_prior_sample_dist,
            nuisance_dist=nuisance_dist,
        )
    else:
        return BED_base_explicit(
            data_likelihood_func=data_likelihood,
            m_prior_dist=model_prior_sample_dist,
        )


def setup_method_kwargs(scen: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Setup keyword arguments for different EIG methods."""
    has_nuisance = (scen['drop_mean'] != 0) and (scen['drop_gradient'] != 0)
    
    NMC_kwargs = dict(
        N = scen['EIG_N'],
        M_prime = 50 if has_nuisance else None,
        reuse_M = True,
        memory_efficient = False # True if scen['EIG_N'] > 1000 else False,
    )

    DN_kwargs = dict(
        N = scen['EIG_N'],
        M_prime = 50 if has_nuisance else None,
    )

    return {
        'NMC': NMC_kwargs,
        'DN': DN_kwargs,
    }


def calculate_reference_eig(BED_class, optimal_design: torch.Tensor, has_nuisance: bool, base_path: str, method: str, i_rec: int) -> torch.Tensor:
    """Calculate reference EIG using more samples."""
    return BED_class.calculate_EIG(
        optimal_design.unsqueeze(0),
        eig_method='NMC',
        eig_method_kwargs=dict(
            N=5000,
            M_prime=50 if has_nuisance else None,
            reuse_M=True,),
        random_seed=1,
        progress_bar=False,
        filename=f'{base_path}/EIG_reference_{method}_{i_rec}.pt',
    )[0]


def save_design_data(design_data: List[Dict[str, Any]], csv_filename: str) -> None:
    """Save design data to CSV file."""
    pd.DataFrame(design_data).to_csv(csv_filename, index=False)


def print_iteration_header(scen: Dict[str, Any], i_rec: int) -> None:
    """Print header for current iteration."""
    print()
    print("+"*100)
    print(f"Running scenario (Receiver: {i_rec}/{scen['N_rec_max']}) | {time.ctime()}")
    print("+"*100)
    for i, (key, value) in enumerate(scen.items()):
        print(f"{key} = {value}  |  ", end="\n" if (i + 1) % 3 == 0 else " ")
    print()
    print("+"*100)
    print()


def run_iterative_optimization(scen: Dict[str, Any], BED_class, design_space: torch.Tensor, method_kwargs: Dict[str, Dict[str, Any]], 
                              design_data: List[Dict[str, Any]], csv_filename: str, base_path: str, num_workers: int) -> None:
    """Run iterative optimization."""
    design_list = design_space.clone()
    optimal_design = None
    
    print("Running iterative optimization...")

    for i_rec in tqdm(range(2, scen['N_rec_max']+1), disable=True):
        
        # dont do this, we need to construct the design in case we run higher N_rec afterwards
        if is_scenario_in_design_data(pd.DataFrame(design_data), scen, i_rec):
            # print(f"Current scenario {scen} with {i_rec} receivers already in design data")
            continue
        # else:
            # print('Calculating design for scenario -----------------------------------------------------')
        
        print_iteration_header(scen, i_rec)
        
        start_time = time.time()
        
        if i_rec <= 3:
            # Use genetic algorithm for smaller number of receivers
            N_pop = max(min(int(i_rec * 2), 100), 10)
            N_generations = 1000
            N_design_space = design_space.shape[0]

            def fitness_func(ga_instance, solution, solution_idx):
                design = design_space[list(solution)].unsqueeze(0)
                eig = BED_class.calculate_EIG(
                    design,
                    eig_method=scen['EIG_method'],
                    eig_method_kwargs=method_kwargs[str(scen['EIG_method'])],
                    num_workers=num_workers,
                    parallel_library='mpire',
                    random_seed=0,
                    progress_bar=False,
                )[0]
                return eig.item()

            torch.manual_seed(0)
            design_space_dist = concave_hull2D_prior_dist_constructor(
                design_space[..., :3], topo_data,
                base_dist='sobol',
                buffer=100, depth=0, ratio=0.05,
            )
            random_design_list = torch.stack([
                design_space_dist.sample(i_rec) for _ in range(N_pop)
            ]).float()
            initial_population = torch.cdist(
                random_design_list, design_space[..., :3]).argmin(dim=-1).tolist()

            ga = pygad.GA(
                num_generations=N_generations,
                num_parents_mating=N_pop // 2,
                fitness_func=fitness_func,
                sol_per_pop=N_pop,
                num_genes=i_rec,
                keep_elitism=2,
                initial_population=initial_population,
                gene_type=int,
                gene_space={'low': 0, 'high': N_design_space - 1},
                allow_duplicate_genes=False,
                random_mutation_min_val=0,
                random_mutation_max_val=N_design_space - 1,
                mutation_type="adaptive",
                mutation_probability=(0.5, 0.1),
                stop_criteria=f'saturate_{N_generations // 10}',
                random_seed=0,
                save_solutions=True,
            )

            ga.run()

            eig_optimal = np.array(ga.best_solutions_fitness).max()
            eig = np.array(ga.best_solutions_fitness)
            optimal_design_idx = ga.best_solution(ga.last_generation_fitness)[0]
            optimal_design = design_space[optimal_design_idx]
        else:
            # Use greedy approach for larger number of receivers
            # print("Using greedy approach for larger number of receivers")
            
            if optimal_design is None:
                #load optimal design from design_data
                # Find the previous optimal design from the design_data list of dicts
                prev_design_entry = next(
                    (entry for entry in design_data
                     if entry['study_area'] == scen['study_area']
                     and entry['model_prior'] == scen['model_prior']
                     and entry['velocity_model'] == scen['velocity_model']
                     and entry['vel_sigma'] == scen['vel_sigma']
                     and entry['noise_correlation'] == scen['noise_correlation']
                     and entry['drop_mean'] == scen['drop_mean']
                     and entry['drop_gradient'] == scen['drop_gradient']
                     and entry['optimisation'] == scen['optimisation']
                     and entry['EIG_method'] == scen['EIG_method']
                     and entry['EIG_N'] == scen['EIG_N']
                     and entry['N_rec'] == i_rec-1),
                    None
                )
                if prev_design_entry is None:
                    raise ValueError(f"No previous design found for N_rec={i_rec-1}")
                # Convert design to tensor if it's a string (from CSV)
                if isinstance(prev_design_entry['design'], str):
                    optimal_design = torch.tensor(np.array(ast.literal_eval(prev_design_entry['design'])))
                else:
                    optimal_design = torch.tensor(prev_design_entry['design'])
            
            # print()
            # print(scen)
            # print(optimal_design)
            # print()
            # print()
                                    
            existing_design = optimal_design.clone().detach()[None]
            existing_design = existing_design.repeat(design_space.shape[0], 1, 1)
            
            design_list = torch.cat([design_space.unsqueeze(1), existing_design], dim=1)

            # print(f"Design list shape: {design_list.shape}")

            eig, info = BED_class.calculate_EIG(
                design_list,
                eig_method=scen['EIG_method'],
                eig_method_kwargs=method_kwargs[str(scen['EIG_method'])],
                random_seed=0,
                num_workers=num_workers,
                # num_workers=16,
                parallel_library='mpire',
                progress_bar=False,
                filename=f'{base_path}/EIG_{scen["EIG_method"]}_{i_rec}.pt',
            )
            
            optimal_design = design_list[eig.argmax()]
            eig = eig.detach()
            
        end_time = time.time()
        total_time = end_time - start_time
        
        has_nuisance = (scen['drop_mean'] != 0) and (scen['drop_gradient'] != 0)
        eig_reference = calculate_reference_eig(
            BED_class, optimal_design, has_nuisance, base_path, scen["EIG_method"], i_rec
        )
        
        design_data.append({
            'study_area': scen['study_area'],
            'model_prior': scen['model_prior'],
            'velocity_model': scen['velocity_model'],
            'vel_sigma': scen['vel_sigma'],
            'noise_correlation': scen['noise_correlation'],
            'drop_mean': scen['drop_mean'],
            'drop_gradient': scen['drop_gradient'],
            'optimisation': scen['optimisation'],
            'EIG_method': scen['EIG_method'],
            'N_rec': i_rec,
            'design': optimal_design.tolist(),
            'EIG': eig[eig.argmax()].item() if i_rec > 3 else eig_optimal,
            'EIG_ref': eig_reference.item(),
            'EIG_N': scen['EIG_N'],
            'runtime': total_time,
            'EIG_candidates': eig.tolist(),
        })
        save_design_data(design_data, csv_filename)


def run_random_or_sobol_optimization(scen: Dict[str, Any], BED_class, design_space: torch.Tensor, method_kwargs: Dict[str, Dict[str, Any]], 
                                    design_data: List[Dict[str, Any]], csv_filename: str, base_path: str, num_workers: int) -> None:
    """Run random or Sobol optimization."""
    N_random = 1000
    
    if scen['optimisation'] == 'random':
        torch.manual_seed(0)
        random_indices = torch.randint(len(design_space), (N_random, scen['N_rec_max']))
        random_design_list = design_space[random_indices].squeeze(-2)
    elif scen['optimisation'] == 'sobol':
        design_space_dist = concave_hull2D_prior_dist_constructor(
                design_space[..., :3], topo_data,
                base_dist='sobol',
                buffer=20, depth=0, ratio=0.05,
            )
        
        torch.manual_seed(0)
        random_design_list = torch.stack([
            design_space_dist.sample(scen['N_rec_max']) for _ in range(N_random)
        ]).float()  
           
        # For each design in the list, find closest design in design_space
        indices = torch.cdist(random_design_list, design_space[..., :3]).argmin(dim=-1)
        random_design_list = design_space[indices]
    
    for i_rec in tqdm(range(2, scen['N_rec_max']+1), disable=True):        
        if not is_scenario_in_design_data(pd.DataFrame(design_data), scen, i_rec):
            print_iteration_header(scen, i_rec)

            start_time = time.time()
            
            eig, info = BED_class.calculate_EIG(
                random_design_list[:, :i_rec],
                eig_method=scen['EIG_method'],
                eig_method_kwargs=method_kwargs[str(scen['EIG_method'])],
                random_seed=0,
                progress_bar=False,
                num_workers=num_workers,
                parallel_library='mpire',
                filename=f'{base_path}/EIG_random_{N_random}_{scen["EIG_method"]}_{i_rec}.pt',
            )
                
            optimal_design = random_design_list[:, :i_rec][eig.argmax()]
            eig = eig.detach()
            
            end_time = time.time()
            total_time = end_time - start_time
            
            has_nuisance = (scen['drop_mean'] != 0) and (scen['drop_gradient'] != 0)
            eig_reference = calculate_reference_eig(
                BED_class, optimal_design, has_nuisance, base_path, scen["EIG_method"], i_rec
            )
            
            design_data.append({
                'study_area': scen['study_area'],
                'model_prior': scen['model_prior'],
                'velocity_model': scen['velocity_model'],
                'vel_sigma': scen['vel_sigma'],
                'noise_correlation': scen['noise_correlation'],
                'drop_mean': scen['drop_mean'],
                'drop_gradient': scen['drop_gradient'],
                'optimisation': scen['optimisation'],
                'EIG_method': scen['EIG_method'],
                'N_rec': i_rec,
                'design': optimal_design.tolist(),
                'EIG': eig[eig.argmax()].item(),
                'EIG_ref': eig_reference.item(),
                'EIG_N': scen['EIG_N'],
                'runtime': total_time,
                'EIG_candidates': eig.tolist(),
            })
            save_design_data(design_data, csv_filename)


def run_genetic_optimization(scen: Dict[str, Any], BED_class, design_space: torch.Tensor, method_kwargs: Dict[str, Dict[str, Any]], 
                            design_data: List[Dict[str, Any]], csv_filename: str, base_path: str, num_workers: int) -> None:
    """Run genetic algorithm optimization."""
    for i_rec in tqdm(range(2, scen['N_rec_max']+1), disable=True):
            
        if not is_scenario_in_design_data(pd.DataFrame(design_data), scen, i_rec):
            print_iteration_header(scen, i_rec)
            start_time = time.time()

            if i_rec == 1:
                raise ValueError("Genetic algorithm not implemented for 1 receiver")
            else:
                N_pop = max(min(int(i_rec*2), 100), 10)
                N_generations = 1000
                N_design_space = design_space.shape[0]
                
                def fitness_func(ga_instance, solution, solution_idx):
                    if solution.ndim == 1:
                        design = design_space[list(solution)].unsqueeze(0)
                    elif solution.ndim == 2:
                        design = torch.stack([design_space[sol] for sol in solution])

                    eig = BED_class.calculate_EIG(
                        design,
                        eig_method=scen['EIG_method'],
                        eig_method_kwargs=method_kwargs[str(scen['EIG_method'])],                        
                        num_workers=num_workers,
                        parallel_library='mpire',
                        random_seed=0,
                        progress_bar=False,
                    )[0]
                                            
                    if solution.ndim == 1:
                        return eig.item()
                    else:
                        if eig.ndim == 1:
                            return eig.numpy()
                        else:
                            return eig.numpy()[None]

                torch.manual_seed(0)
                design_space_dist = concave_hull2D_prior_dist_constructor(
                    design_space[..., :3], topo_data,
                    base_dist='sobol',
                    buffer=100, depth=0, ratio=0.05,
                )
                random_design_list = torch.stack([
                    design_space_dist.sample(i_rec) for _ in range(N_pop)
                ]).float()  
                # For each design in the list, find closest design in design_space
                initial_population = torch.cdist(
                    random_design_list, design_space[..., :3]).argmin(dim=-1).tolist()
                                    
                # Initialize tqdm progress bar
                pbar = tqdm(total=N_generations, desc=f"GA Generation (N_rec={i_rec})")

                # Define the callback function for tqdm
                def on_generation_callback(ga_instance):
                    pbar.update(1)
                    # Optionally update description with best fitness
                    pbar.set_postfix(best_fitness=f"{ga_instance.best_solution()[1]:.4f}")

                ga = pygad.GA(
                    num_generations=N_generations,
                    num_parents_mating=N_pop//2,
                    fitness_func=fitness_func,
                    sol_per_pop=N_pop,
                    num_genes=i_rec,
                    keep_elitism=2,
                    initial_population=initial_population,
                    gene_type=int,
                    gene_space={'low': 0, 'high': N_design_space-1},
                    allow_duplicate_genes=False,
                    random_mutation_min_val=0,
                    random_mutation_max_val=N_design_space-1,
                    mutation_type="adaptive",
                    mutation_probability=(0.5, 0.1),
                    stop_criteria=f'saturate_{N_generations//10}',
                    random_seed=0,
                    save_solutions=True,
                    on_generation=on_generation_callback, # Add the callback here
                )

                # Run GA (progress bar updates inside)
                ga.run()

                # Close the progress bar
                pbar.close()
            
                ga.run()
            
                eig_optimal = np.array(ga.best_solutions_fitness).max()
                eig = np.array(ga.best_solutions_fitness)
                
                optimal_design_idx = ga.best_solution(ga.last_generation_fitness)[0]
                optimal_design = design_space[optimal_design_idx]
                
            end_time = time.time()
            total_time = end_time - start_time
            
            has_nuisance = (scen['drop_mean'] != 0) and (scen['drop_gradient'] != 0)
            eig_reference = calculate_reference_eig(
                BED_class, optimal_design, has_nuisance, base_path, scen["EIG_method"], i_rec
            )
            
            design_data.append({
                'study_area': scen['study_area'],
                'model_prior': scen['model_prior'],
                'velocity_model': scen['velocity_model'],
                'vel_sigma': scen['vel_sigma'],
                'noise_correlation': scen['noise_correlation'],
                'drop_mean': scen['drop_mean'],
                'drop_gradient': scen['drop_gradient'],
                'optimisation': scen['optimisation'],
                'EIG_method': scen['EIG_method'],
                'N_rec': i_rec,
                'design': optimal_design.tolist(),
                'EIG': eig_optimal,
                'EIG_ref': eig_reference.item(),
                'EIG_N': scen['EIG_N'],
                'runtime': total_time,
                'EIG_candidates': eig.tolist(),
            })
            save_design_data(design_data, csv_filename)
        else:
            print(f"Current scenario with {i_rec} receivers already in design data")


def main():
    """Main function to run the design generation process."""
    # Parse command line arguments
    scen = parse_arguments()

    # Print scenario parameters
    print_scenario(scen)
    
    # Set number of workers
    num_workers = 1
    
    # Setup base path and design data
    base_path = f'generate_designs/data/scenario_{scen["study_area"]}_{scen["model_prior"]}_{scen["velocity_model"]}_{scen["vel_sigma"]:.3f}_{scen["noise_correlation"]:.1f}_{scen["drop_mean"]:.1f}_{scen["drop_gradient"]:.1f}_{scen["optimisation"]}_{scen["EIG_method"]}_{scen["EIG_N"]}'
    os.makedirs(base_path, exist_ok=True)
    csv_filename = f'{base_path}/design_data.csv'
    
    # Setup design data
    design_data_df = setup_design_data(base_path)
    design_data = design_data_df.to_dict(orient='records')

    # Setup environment
    setup_environment()
    
    # Get design space
    design_space = get_design_space(scen['study_area'])
    
    # Get model prior samples
    model_prior_samples = get_model_prior_samples(scen['study_area'], scen['model_prior'])
    
    # Get forward function
    forward_function = get_forward_function(scen['velocity_model'], scen['model_prior'], 
                                          scen['study_area'], model_prior_samples)
    
    # Setup data likelihood and nuisance distribution
    data_likelihood, nuisance_dist = setup_data_likelihood(forward_function, scen)
    
    # Setup BED class
    BED_class = setup_bed_class(data_likelihood, model_prior_samples, nuisance_dist)
    
    # Setup method keyword arguments
    method_kwargs = setup_method_kwargs(scen)
    
    # Run optimization based on chosen method
    if scen['optimisation'] == 'iterative':
        run_iterative_optimization(scen, BED_class, design_space, method_kwargs, 
                                 design_data, csv_filename, base_path, num_workers)
    elif scen['optimisation'] in ['random', 'sobol']:
        run_random_or_sobol_optimization(scen, BED_class, design_space, method_kwargs, 
                                       design_data, csv_filename, base_path, num_workers)
    elif scen['optimisation'] == 'genetic':
        run_genetic_optimization(scen, BED_class, design_space, method_kwargs, 
                               design_data, csv_filename, base_path, num_workers)
    else:
        raise ValueError("Optimisation method not implemented. Choose 'iterative', 'random', 'sobol', or 'genetic'")


if __name__ == '__main__':
    main()
    print("Design generation completed successfully.")
    print_scenario(parse_arguments())  # Print the scenario again at the end for confirmation