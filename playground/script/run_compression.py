# import time
# time.sleep(3600)

import sys
sys.path.append('/home/joonmyung/project/HanWha/InternVideo-HanWha/InternVideo2/multi_modality')
from utils.script import find_lists, find_lists_full, resultManager, find_highest_index
from joonmyung.script import GPU_Worker, Process_Worker
from joonmyung.utils import str2list, str2bool
from joonmyung.meta_data import data2path
from operator import add
import argparse
import random
import socket
import copy
import os

parser = argparse.ArgumentParser(description="tuning")
parser.add_argument('--activate',        default=True, type=str2bool)
parser.add_argument('--gpus',            default=[1,2,3], type=str2list)
parser.add_argument('--need_gpu',        default=1,   type=int)
parser.add_argument('--max_run_num',     default=4,   type=int)
parser.add_argument('--checkType',       default=0,   type=int)
parser.add_argument('--utilRatio',       default=0,   type=int)

parser.add_argument('--output_dir',      default="/hub_data1/joonmyung/project/HanWha", type=str)
parser.add_argument('--result_name',     default="1.0.0", type=str)
parser.add_argument('--batch_size',      default='1',   type=int)


parser.add_argument('--settings', default=[
                                        ["ret", "msrvtt", "tasks/pretrain.py", "scripts/evaluation/stage2/zero_shot/1B/config_msrvtt.py", "./pretrained/InternVideo2-stage2_1b-224p-f4.pt",  [500, 800]]
                    ], type=str2list)

parser.add_argument('--search_space',    default=[[1000, 100], [500,  50]], type=str2list)
parser.add_argument('--compression',     default=[[1, 0, 10, 0, 1, 1], [1, 10, 25, 1], [0]], type=str2list)
parser.add_argument('--r_merge',         default=[[0] * 40], type=str2list)
args = parser.parse_args()


processes, count, ex, ex_prev = [], 1, 0, 0
hostname = socket.gethostname()
server = data2path("imagenet")[3]

for task, dataset, run, config, pretrained_path, flop_range in args.settings:
    result_path = os.path.join(args.output_dir, server, task)
    r_merges = []
    for r_merge in args.r_merge:
        for ss in args.search_space:
            add_rs = find_lists(40, *ss)
            add_rs = [l for l in add_rs if find_highest_index(l) < 4]
            for r_add in add_rs:
                r = copy.deepcopy(r_merge)
                r = list(map(add, r, r_add))

                log_stats = {"r_merge": r, "mctf": args.compression[1], "vid_TLDR": args.compression[2], "task": task, "dataset" : dataset}
                if resultManager(file_name=f"{args.version}.pkl", folder_path = result_path, new_result=log_stats, checkColumns=list(log_stats.keys()), duplicate_check=True):
                    r_merges.append(r)
                    ex += 1
                else:
                    ex_prev += 1

    gpu_len = min(len(args.gpus) // args.need_gpu, len(r_merges))
    r_merges_split = [r_merges[i::gpu_len] for i in range(gpu_len)]

    for i, r_merges in enumerate(r_merges_split):
        r_merges_str    = str(r_merges).replace(' ', '').replace('\'', '')
        flop_range      = str(flop_range).replace(' ', '').replace('\'', '')
        compression_str = str(args.compression).replace(' ', '').replace('\'', '')
        port = random.randint(15000, 20000)
        process = f"python {run} {config} pretrained_path {pretrained_path} output_dir {result_path} batch_size_test {args.batch_size} use_bf16 False \
                model.vision_encoder.use_flash_attn False model.vision_encoder.use_fused_rmsnorm False model.vision_encoder.use_fused_mlp False \
                model.vision_encoder.compression {compression_str} r_merges {r_merges_str} \
                version {args.version} analysis True server {server} task {task} dataset {dataset} flop_range {flop_range}"
        processes.append(process)
        count += 1
# torch.distributed.launch --nproc_per_node=1
#  --master_port=25032 --use_env tasks/pretrain.py scripts/evaluation/stage2/zero_shot/1B/config_msrvtt.py output_dir ./output evaluate True pretrained_path /hub_data1/joonmyung/weights/InternVideo2/zero_shot/InternVideo2-stage2_1b-224p-f4.pt batch_size_test 3 batch_size 1
if args.activate:
    print(f"New / Exist / ALL : {ex} / {ex_prev} / {ex + ex_prev}")
    gpuWorker = GPU_Worker(args.gpus, 60, 60, checkType=args.checkType, utilRatio=args.utilRatio, need_gpu=args.need_gpu, max_run_num=args.max_run_num)
    Process_Worker(processes, gpuWorker, True)
else:
    print(processes)

