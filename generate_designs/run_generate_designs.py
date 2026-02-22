import subprocess

from scenarios import scenarios
# from scenarios_iterative import scenarios

import time

N_parallel = 16

# Check if slurm is available
def is_slurm_available():
    # try:
    #     subprocess.run(['scontrol', 'ping'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #     return True
    # except (subprocess.CalledProcessError, FileNotFoundError):
    return False

# Run generate designs
def run_generate_designs(scen, i):
    argument_list = []
    for k, v in scen.items():
        argument_list.append(f'--{k}')
        argument_list.append(str(v))
        
    if is_slurm_available():
        memory = "6G"
        num_cpus = 1
        job_name = f"ga_{i}"
        output_log = f"generate_designs/slurm_out/{job_name}.out"
        error_log = f"generate_designs/slurm_out/{job_name}.err"

        print(f">>>>>> Running scenario {i} with slurm")
        print(f"Command: {' '.join(['srun', '--mem', memory, '--cpus-per-task', str(num_cpus), '--job-name', job_name, '--output', output_log, '--error', error_log, 'python', 'generate_designs/generate_designs.py'] + argument_list)}")
        print()


        return subprocess.Popen(['srun', '--mem', memory, '--cpus-per-task', str(num_cpus), '--job-name', job_name, '--output', output_log, '--error', error_log, 'python', 'generate_designs/generate_designs.py'] + argument_list)
    else:
        return subprocess.Popen(['/scratch/dstrutz/.conda/envs/cdv_oed_benchmark/bin/python', 'generate_designs/generate_designs.py'] + argument_list)

if __name__ == "__main__":
    processes = []
    # for i, scen in enumerate(scenarios):
    
    # run setfacl -R -m u:s2272341:rwx * using the command line, ignoring errors
    subprocess.run(['setfacl', '-R', '-m', 'u:s2272341:rwx', '*'], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    for i, scen in enumerate(scenarios):
        while len(processes) >= N_parallel:
            time.sleep(1)
            processes = [(p, scen, i) for p, scen, i in processes if p.poll() is None]
    
        p = run_generate_designs(scen, i)
        processes.append((p, scen, i))

    # while processes:
    #     for p, scen, i in processes:
    #         if p.poll() is not None:
    #             if p.returncode == 0:
    #                 print(f"Scenario {i} completed successfully.")
    #                 processes.remove((p, scen, i))
    #             else:
    #                 print(f"Scenario {i} failed. Retrying...")
    #                 new_p = run_generate_designs(scen, i)
    #                 processes.append((new_p, scen, i))
    #                 processes.remove((p, scen, i))
    #     time.sleep(1)
    
    for p, scen, i in processes:
        p.wait()
        if p.returncode == 0:
            print(f"Scenario {i} completed successfully.")
        else:
            print(f"Scenario {i} failed.")