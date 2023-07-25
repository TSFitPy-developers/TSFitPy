from dask.distributed import Client
from dask_jobqueue import SLURMCluster
import time
import numpy as np
import socket


def get_client(client_type: str, cluster_name: str, workers_amount: int, **kwargs):
    if cluster_name is None:
        cluster_name = "unknown"
    print("Preparing workers")
    if client_type == "local":
        client = get_local_client(workers_amount)
    elif client_type == "slurm":
        client = get_slurm_cluster(kwargs['nodes'], **kwargs)
    else:
        raise ValueError("client_type must be either local or slurm")

    print(client)

    host = client.run_on_scheduler(socket.gethostname)
    port = client.scheduler_info()['services']['dashboard']
    print(f"Assuming that the cluster is ran at {cluster_name} (change in config if not the case)")

    print(f"ssh -N -L {port}:{host}:{port} {cluster_name}")
    print(f"Then go to http://localhost:{port}/status to check the status of the workers")

    print("Worker preparation complete")

    return client


def get_local_client(workers_amount, **kwargs):
    if workers_amount > 1:
        client = Client(threads_per_worker=1, n_workers=workers_amount, **kwargs)
    else:
        client = Client(threads_per_worker=1, **kwargs)
    return client


def get_slurm_cluster(cores_per_job: int, jobs: int, memory_per_job_gb: int, script_commands=None, **kwargs):
    if script_commands is None:
        script_commands = [            # Additional commands to run before starting dask worker
            'module purge',
            'module load basic-path',
            'module load intel',
            'module load anaconda3-py3.10']
    # Create a SLURM cluster object
    cluster = SLURMCluster(
        cores=cores_per_job,                     # Number of cores per job (so like cores/workers per node)
        memory=f"{memory_per_job_gb}GB",         # Amount of memory per job (also per node)
        job_script_prologue=script_commands,     # Additional commands to run before starting dask worker
        **kwargs
    )
    cluster.scale(jobs=jobs)      # How many nodes
    client = Client(cluster)

    return client
