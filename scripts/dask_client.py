from __future__ import annotations
from dask.distributed import Client
import time
import numpy as np
import socket


def get_dask_client(client_type: str, cluster_name: str, workers_amount_cpus: int, night_mode=False, nodes=1, slurm_script_commands=None,
                    slurm_memory_per_core=3.6, time_limit_hours=72, slurm_partition="debug", **kwargs):
    if cluster_name is None:
        cluster_name = "unknown"
    if not night_mode:
        print("Preparing workers")
    if client_type == "local":
        client = get_local_client(workers_amount_cpus)
    elif client_type == "slurm":
        client = get_slurm_cluster(workers_amount_cpus, nodes, slurm_memory_per_core,
                                   script_commands=slurm_script_commands, time_limit_hours=time_limit_hours,
                                   slurm_partition=slurm_partition, **kwargs)
    else:
        raise ValueError("client_type must be either local or slurm")
    if not night_mode:
        print(client)

    host = client.run_on_scheduler(socket.gethostname)
    port = client.scheduler_info()['services']['dashboard']
    if not night_mode:
        print(f"Assuming that the cluster is ran at {cluster_name} (change in config if not the case)")

        print(f"ssh -N -L {port}:{host}:{port} {cluster_name}")
        print(f"Then go to http://localhost:{port}/status to check the status of the workers")

        print("Worker preparation complete")

    return client


def get_local_client(workers_amount, **kwargs):
    if workers_amount >= 1:
        client = Client(threads_per_worker=1, n_workers=workers_amount, **kwargs)
    else:
        client = Client(threads_per_worker=1, **kwargs)
    return client


def get_slurm_cluster(cores_per_job: int, jobs_nodes: int, memory_per_core_gb: int, script_commands=None,
                      time_limit_hours=72, slurm_partition='debug', **kwargs):
    from dask_jobqueue import SLURMCluster
    if script_commands is None:
        script_commands = [            # Additional commands to run before starting dask worker
            'module purge',
            'module load basic-path',
            'module load intel',
            'module load anaconda3-py3.10']
    # Create a SLURM cluster object
    # split into days, hours in format: days-hh:mm:ss
    days = time_limit_hours // 24
    hours = time_limit_hours % 24
    if days == 0:
        time_limit_string = f"{int(hours):02d}:00:00"
    else:
        time_limit_string = f"{int(days)}-{int(hours):02d}:00:00"
    print(time_limit_string)
    cluster = SLURMCluster(
        queue=slurm_partition,                      # Which queue/partition to submit jobs to
        cores=cores_per_job,                     # Number of cores per job (so like cores/workers per node)
        memory=f"{memory_per_core_gb * cores_per_job}GB",         # Amount of memory per job (also per node)
        job_script_prologue=script_commands,     # Additional commands to run before starting dask worker
        walltime=time_limit_string                      # Time limit for each job
    )
    cluster.scale(jobs=jobs_nodes)      # How many nodes
    client = Client(cluster)

    return client
