import os
from pathlib import Path
import subprocess

import submitit


huggingface_fix = f"TRANSFORMERS_OFFLINE=1 CURL_CA_BUNDLE=''"


class SubmititLauncher:
    def __init__(self, args):
        self.args = args

    def __call__(self):
        host_name = os.popen(
            "scontrol show hostnames $SLURM_JOB_NODELIST"
        ).read().split("\n")[0]
        self._set_gpu_args()
        # Using Accelerate for launching
        multi_gpu = "--multi_gpu" if self.args.num_nodes * self.args.gpu_per_node > 1 else ""
        opts = " ".join(self.args.opts) if len(self.args.opts) > 0 else ""
        opts += f" num_gpu={self.args.num_nodes * self.args.gpu_per_node} "
        full_cfg_path = Path(self.args.config)
        cfg_path, cfg_file = str(full_cfg_path.parent), str(full_cfg_path.name)
        cmd = f"{huggingface_fix} accelerate launch --num_machines {self.args.num_nodes} \
                        --mixed_precision {self.args.mixed_precision} {multi_gpu} \
                        --num_processes {self.args.gpu_per_node * self.args.num_nodes} \
                        --num_cpu_threads_per_process {self.args.cpu_per_task} \
                        --main_process_ip {host_name} \
                        --main_process_port {self.args.port} \
                        --machine_rank {self.args.node_id} \
                        --dynamo_backend no \
                        {self.args.run_file} \
                        --config-path {cfg_path} \
                        --config-name {cfg_file} \
                        num_gpu={self.args.num_nodes * self.args.gpu_per_node} \
                        hydra.run.dir=. \
                        hydra.output_subdir=null \
                        hydra/job_logging=disabled \
                        hydra/hydra_logging=disabled {opts}"
        subprocess.run(cmd, shell=True)
    
    def _set_gpu_args(self):
        job_env = submitit.JobEnvironment()
        self.args.job_dir = str(self.args.job_dir).replace("%j", job_env.job_id)
        self.args.node_id = int(job_env.global_rank / self.args.gpu_per_node)


def submitit_launch(args):
    """
    Multi node script launching with Submitit
    """
    additional_parameters = {}
    if args.nodelist != "":
        # if specifying node id
        nodelist = f"{str(args.nodelist)}"
        additional_parameters["nodelist"] = nodelist

    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)
    executor.update_parameters(
        name=args.name,
        mem_gb=args.mem_per_gpu * args.gpu_per_node * args.num_nodes,
        gpus_per_node=args.gpu_per_node,
        tasks_per_node=1,
        cpus_per_task=args.gpu_per_node * args.cpu_per_task,
        nodes=args.num_nodes,
        slurm_qos=args.qos,
        slurm_partition=args.partition,
        slurm_account=args.account,
        slurm_time=args.time * 60,
        slurm_signal_delay_s=120,
        slurm_additional_parameters=additional_parameters
    )
    launcher = SubmititLauncher(args)
    job = executor.submit(launcher)
    print(f"submitted job: {job.job_id}")


def accelerate_launch(args):
    """
    Single node script launching with Accelerate
    """
    opts = " ".join(args.opts) if len(args.opts) > 0 else ""
    opts += f" num_gpu={args.num_nodes * args.gpu_per_node} "
    multi_gpu = "--multi_gpu" if args.num_nodes * args.gpu_per_node > 1 else ""
    full_cfg_path = Path(args.config)
    cfg_path, cfg_file = str(full_cfg_path.parent), str(full_cfg_path.name)
    cmd = f"{huggingface_fix} accelerate launch --num_machines {args.num_nodes} \
        {multi_gpu} \
        --mixed_precision {args.mixed_precision} \
        --num_processes {args.gpu_per_node * args.num_nodes} \
        --num_cpu_threads_per_process {args.cpu_per_task} \
        --dynamo_backend no \
        {args.run_file} \
        --config-path {cfg_path} \
        --config-name {cfg_file} \
        num_gpu={args.num_nodes * args.gpu_per_node} \
        hydra.run.dir=. \
        hydra.output_subdir=null \
        hydra/job_logging=disabled \
        hydra/hydra_logging=disabled {opts}"
    subprocess.run(cmd, shell=True)


def python_launch(args):
    """
    Vanilla python launcher for degbugging purposes
    """
    opts = " ".join(args.opts) if len(args.opts) > 0 else ""
    full_cfg_path = Path(args.config)
    cfg_path, cfg_file = str(full_cfg_path.parent), str(full_cfg_path.name)
    cmd = f"{huggingface_fix} python {args.run_file} " \
          f"--config-path {cfg_path} " \
          f"--config-name {cfg_file} " \
          f"num_gpu=1 " \
          f"hydra.run.dir=. " \
          f"hydra.output_subdir=null " \
          f"hydra/job_logging=disabled " \
          f"hydra/hydra_logging=disabled {opts}"
    subprocess.run(cmd, shell=True)