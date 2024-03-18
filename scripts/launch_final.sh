EXP_NAME=FinalOVPretrain
CFG_MODE=$1
RESUME_CKPT=$2
OBJ_CKPT=$3

random_number=$RANDOM
random_number=$((random_number % 50))
PORT=$((random_number+4321))

CONFIG="configs/final/${CFG_MODE}.yaml"

# Run batchsize 512
if [ -n "$OBJ_CKPT" ]; then
    MODEL_OPTS="${MODEL_OPTS} model.vision.args.path=$OBJ_CKPT"
fi

if [ -n "$RESUME_CKPT" ]; then
  RESUME_OPTS="resume=True exp_dir=$RESUME_CKPT"
fi

python launch.py --time 72 --name $CFG_MODE --qos lv1 --port $PORT --partition HGX --mem_per_gpu 80 --gpu_per_node 8 \
        --config $CONFIG dataloader.batchsize=64 note=$CFG_MODE name=$EXP_NAME $MODEL_OPTS $RESUME_OPTS solver.epochs=400