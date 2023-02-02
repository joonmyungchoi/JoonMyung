import argparse
from joonmyung.script import GPU_Worker, Process_Worker
from joonmyung.utils import str2list

parser = argparse.ArgumentParser(description="tuning")
# a. Setting
parser.add_argument('--gpus', default=[0, 1, 2, 3, 4, 5], type=str2list)
parser.add_argument('--server', default='152', type=str)
parser.add_argument('--wandb_table', default='OURS', type=str)
parser.add_argument('--wandb_version', default='1.1.0', type=str, help="version")
parser.add_argument('--wandb_project', default='2023ICCV', type=str, help="version")
parser.add_argument('--wandb_entity',  default='joonmyung', type=str, help="version")

# b. Setting
parser.add_argument('--seed', default=[20170922], type=str2list)
parser.add_argument('--dataset', default=["cifar100"], help="", type=str2list)

# c. Tunning
parser.add_argument('--lr', default=[0.1], help="", type=str2list)
args = parser.parse_args()

processes = []
for seed in args.seed:
    for dataset in args.dataset:
        for lr in args.lr:

            process = f"python train.py \
                  --server {args.server}  --seed {seed} --dataset {dataset} \
                  --use_wandb --wandb_entity {args.wandb_entity} --wandb_project {args.wandb_project} --wandb_table {args.wandb_table} --wandb_version {args.wandb_version} \
                  --lr {lr}"
            processes.append(process)



gpuWorker = GPU_Worker(args.gpus, 30, 120)
Process_Worker(processes, gpuWorker)