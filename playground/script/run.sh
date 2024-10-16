cuda=${1}
nproc=${2}
port=${3}
server=${4}
wandb_version=${5}
wandb_table=${6}
save_every=${7}
model=${8}
embed_dim=${9}
batch_size=${10}
reg_lambda=${11}




dataset="IMNET"

if [ $server = 149 ]
then
  data_path="/hub_data1/imagenet/imagenet-pytorch"
  output_dir="/hub_data1/joonmyung/2023ICLR"
elif [ $server = kakao ]
then
  data_path="/data/opensets/imagenet-pytorch"
  output_dir="/data/project/rw/joonmyung/conference/2023ICLR"
elif [ $server = 148 ]
then
  data_path="/hub_data/imagenet/imagenet-pytorch"
  output_dir="/hub_data/joonmyung/2023ICLR"

elif [ $server = 137 ]
then
  data_path="/hub_data1/joonmyung/data/imagenet"
  output_dir="/hub_data/joonmyung/2023ICLR"
elif [ $server = 154 ]
then
  data_path="/data1/imagenet/imagenet-pytorch"
  output_dir="/data1/joonmyung/conference/2023ICLR"
elif [ $server = 151 ]
then
  data_path="/hub_data1/joonmyung/data/imagenet"
  output_dir="/hub_data1/joonmyung/conference/2023ICLR"
elif [ $server = 113 ]
then
  data_path="/hub_data1/joonmyung/data/imagenet"
  output_dir="/hub_data/joonmyung/conference/2023ICLR"

elif [ $server = 65 ]
then
  data_path="/hub_data1/joonmyung/data/imagenet"
  output_dir="/hub_data1/joonmyung/conference/2023ICLR"
elif [ $server = 67 ]
then
  data_path="/hub_data1/joonmyung/data/imagenet"
  output_dir="/hub_data/joonmyung/conference/2023ICLR"
elif [ $server = 64 ]
then
  data_path="/hub_data/imagenet/imagenet-pytorch"
  output_dir="/hub_data/joonmyung/2023ICLR"
elif [ $server = kisti ]
then
  data_path="/scratch/x2487a05/data/imagenet"
  output_dir="/scratch/x2487a05/data/2023ICLR"
else
  data_path="/hub_data/imagenet/imagenet-pytorch"
  output_dir="None"
fi



echo "CUDA_VISIBLE_DEVICES=${cuda} python -m torch.distributed.launch --nproc_per_node=${nproc} --use_env main.py --port ${port} --server ${server} --output_dir ${output_dir} \
  --use_wandb --wandb_entity joonmyung --wandb_project 2023ICCV --wandb_version ${wandb_version} --wandb_table ${wandb_table} \
  --data-path ${data_path} --dataset ${dataset} --seed 0 --save_every ${save_every} \
  --model ${model} --embed_dim ${embed_dim} --batch-size ${batch_size} \
  --reg_lambda ${reg_lambda}"


CUDA_VISIBLE_DEVICES=${cuda} python -m torch.distributed.launch --nproc_per_node=${nproc} --use_env main.py --port ${port} --server ${server} --output_dir ${output_dir} \
  --use_wandb --wandb_entity joonmyung --wandb_project 2023ICCV --wandb_version ${wandb_version} --wandb_table ${wandb_table} \
  --data-path ${data_path} --dataset ${dataset} --seed 0 --save_every ${save_every} \
  --model ${model} --embed_dim ${embed_dim} --batch-size ${batch_size} \
  --reg_lambda ${reg_lambda}

