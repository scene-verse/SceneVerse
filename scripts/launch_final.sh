EXP_NAME=FinalOVPretrain
CFG_MODE=$1
OBJ_CKPT=$2

random_number=$RANDOM
random_number=$((random_number % 50))
PORT=$((random_number+4321))

CONFIG="configs/final/${CFG_MODE}.yaml"

# Run batchsize 512
if [ -n "$OBJ_CKPT" ]; then
    MODEL_OPTS="${MODEL_OPTS} model.vision.args.path=$OBJ_CKPT"
fi
python launch.py --time 8 --name $CFG_MODE --qos lv0b --port $PORT --partition HGX --mem_per_gpu 80 --gpu_per_node 4 \
        --config $CONFIG dataloader.batchsize=64 note=$CFG_MODE name=$EXP_NAME $MODEL_OPTS