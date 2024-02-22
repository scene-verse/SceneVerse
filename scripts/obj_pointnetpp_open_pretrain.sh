EXP_NAME=ALLObjPretrain

#TRAIN_SET_REAL_ALL="['ScanNetPretrainObj','RScanPretrainObj','ARKitScenePretrainObj','MultiScanPretrainObj','HMPretrainObj']"
#TRAIN_SET_WSYN="['ScanNetPretrainObj','RScanPretrainObj','ARKitScenePretrainObj','MultiScanPretrainObj','HMPretrainObj','S3DPretrainObj']"
#TRAIN_SET_ALL="['ScanNetPretrainObj']"
#TRAIN_SET_ALL="['ScanNetPretrainObj','RScanPretrainObj']"
#TRAIN_SET_ALL="['ScanNetPretrainObj','RScanPretrainObj','MultiScanPretrainObj']"
#TRAIN_SET_NOSCANNET="['RScanPretrainObj','ARKitScenePretrainObj','MultiScanPretrainObj','HMPretrainObj']"
#TRAIN_SET_NOMULTISCAN="['ScanNetPretrainObj','RScanPretrainObj','ARKitScenePretrainObj','HMPretrainObj']"
#TRAIN_SET_WSYN="['ScanNetPretrainObj','RScanPretrainObj','ARKitScenePretrainObj','MultiScanPretrainObj','HMPretrainObj','ProcThorPretrainObj']"
TRAIN_SET_PROCTHOR="['ProcThorPretrainObj']"

#TRAIN_SETS=($TRAIN_SET_ALL $TRAIN_SET_NOSCANNET $TRAIN_SET_NOMULTISCAN $TRAIN_SET_WSYN)
#TRAIN_SETS=($TRAIN_SET_ALL $TRAIN_SET_REAL_ALL)
# TRAIN_SETS=($TRAIN_SET_REAL_ALL $TRAIN_SET_WSYN)
TRAIN_SETS=($TRAIN_SET_PROCTHOR)

#VAL_SET_ALL="['ScanNetPretrainObj']"
#VAL_SET_NOSCANNET="['ScanNetPretrainObj']"
#VAL_SET_NOMULTISCAN="['MultiScanPretrainObj']"
VAL_SET_PROCTHOR="['ProcThorPretrainObj']"
#VAL_SETS=($VAL_SET_ALL $VAL_SET_NOSCANNET $VAL_SET_NOMULTISCAN)
VAL_SETS=($VAL_SET_PROCTHOR)
#TEST_SETS=($VAL_SET_ALL $VAL_SET_NOSCANNET $VAL_SET_NOMULTISCAN)
TEST_SETS=($VAL_SET_PROCTHOR)

SOLVER_OPTS="solver.epochs=1500 solver.lr=1e-2 solver.sched.args.warmup_steps=0"

PORTS=("1350" "1456" "1567" "1678")
#NOTES=("procthor" "obj_ov_nomultiscan" "obj_ov_wsyn")
NOTES=("procthor")

# Run batchsize 512
for ((i=0; i<${#TRAIN_SETS[@]}; i++)); do
        train_set=${TRAIN_SETS[i]}
        val_set=${VAL_SETS[i]}
        test_set=${TEST_SETS[i]}
        port=${PORTS[i]}
        note=${NOTES[i]}
        if [ -n "$1" ]; then
            PARTITION=$1
        else
            PARTITION="HGX"
        fi
        python launch.py --name $note --time 8 --qos lv0a --port $port --partition $PARTITION --mem_per_gpu 100 --gpu_per_node 4 \
        --config configs/pretrain/obj_cls_pretrain.yaml dataloader.batchsize=128 \
         data.train=$train_set data.val=$val_set data.test=$test_set note=1115${note} name=$EXP_NAME \
         logger.entity=bigai-gvl \
        $SOLVER_OPTS \
        dataloader.num_workers=16
done