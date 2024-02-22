EXP_NAME=FinalOVFinetune
CFG_MODE=$1
NOTE=$2
PRETRAIN_CKPT=$3
SOLVER_EPOCHS=$4
SOLVER_WARMUPS=$5
SOLVER_LR=$6

random_number=$RANDOM
random_number=$((random_number % 50))
PORT=$((random_number+4321))

CONFIG="configs/final/finetune/${CFG_MODE}.yaml"

if [ -n "$SOLVER_EPOCHS" ]; then
  MODEL_OPTS="${MODEL_OPTS} $SOLVER_EPOCHS $SOLVER_WARMUPS $SOLVER_LR"
fi

python launch.py --time 8 --name $CFG_MODE --qos lv2 --port $PORT --partition DGX --mem_per_gpu 100 --gpu_per_node 8 \
        --config $CONFIG dataloader.batchsize=64 note=$NOTE name=$EXP_NAME $MODEL_OPTS pretrain_ckpt_path=$PRETRAIN_CKPT \
#        ckpt_path="/scratch/masaccio/results/AllOVPretrain_b512_Pretrain_all_all_pretrain_refined_scene/2023-11-10-16:44:05.916327/ckpt/best.pth" \
#        resume=True \
#        dataloader.num_workers=2