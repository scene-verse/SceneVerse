# Getting Started

## Environment Setup
To install the environment requirements needed for SceneVerse, you can run the installation scripts provided by:
```bash
$ conda env create -n sceneverse python=3.9
$ conda activate sceneverse
$ pip install --r requirements.txt
```
Meanwhile, SceneVerse depends on an efficient implementation of PointNet2  which is located in ```modules```. Remember to install it with
```bash
$ cd modules/third_party/pointnet2
$ python setup.py install
$ cd ../..
```

## Model Configurations
### 1. Experiment Setup
We provide all experiment configurations in ```configs/final```, you can find the experiment setting in the top of comment
each experiment file. To correctly use the configuration files, you need to change the following fields in the configuration
file to load paths correctly: 
- ```base_dir```: save path for model checkpoints, configurations, and logs.
- ```logger.entity```: we used W&B for logging experiments, change it to your corresponding account.
- ```data.{DATASET}_familiy_base```: path to ```{Dataset}``` related data.
- ```model.vision.args.path```: path to the pre-trained object encoder (PointNet++).
- ```model.vision.args.lang_path```: deprecated, but basically text embeddings of the 607 classes in ScanNet.

You can walk through the ```configs/final/all_pretrain.yaml``` and compare it with other files to see how we controlled
data and objectives used in training. 

### 2. Checkpoints
We will release the checkpoints and logs of experiments reported in the paper shortly.

[//]: # (### 1. 3D visual grounding experiments)

[//]: # (We provide all available experimental configurations in ```configs/```.)

[//]: # ()
[//]: # (| Setting | Dataset | Model |)

[//]: # (|---------|---------|-------|)

[//]: # (|   Pretrain      | ScanNet/ArkitScenes/3RScan/RScan/HM3D/MultiScan |  TBD     |)

[//]: # (|   Pretrain      | ScanNet/ArkitScenes/3RScan/RScan/HM3D/MultiScan<br>Structured3D | TBD   |)

[//]: # (|   Pretrain      | ScanNet/ArkitScenes/3RScan/RScan/HM3D/MultiScan<br>Structured3D/ProcTHOR | TBD  |)

[//]: # (|   ScanNet      |  Scratch      |  TBD |)

[//]: # (|   ScanNet      |  Finetuned    |  TBD |)

[//]: # (|   SceneVerse   |  Scratch      |  TBD |)

[//]: # (|   SceneVerse   |  Finetuned    |  TBD |)

[//]: # ()
[//]: # (### 2. Zero-shot transfer experiments)

[//]: # (### 3. 3D question answering)

[//]: # (To be released, stay tuned)

[//]: # (### 4. Open-vocabulary 3D segmentation)

[//]: # (To be released, stay tuned)

## Experiments
### 1. Training and Inference
This codebase leverages [Huggingface Accelerate](https://huggingface.co/docs/accelerate/index) package and 
[Facebook Submitit](https://github.com/facebookincubator/submitit) package for efficient model training on multi-node clusters.
We provide a launcher file ```launch.py``` which provides three ways of launching experiment:
```bash
# Launching using submitit on a SLURM cluster (e.g. 10 hour 1 node 4 GPU experiment with config file $CONFIG)
$ python launch.py --mode submitit --time 10 --qos $QOS --partition $PARTITION --mem_per_gpu 80 \
                   --gpu_per_node 4 --config $CONFIG note=$NOTE name=$EXP_NAME
                   
# Launching using accelerator with a multi-gpu instance
$ python launch.py --mode accelerate --gpu_per_node 4 --num_nodes 1 -- config $CONFIG note=$NOTE name=$EXP_NAME 
```
Basically, ```launch.py``` set up process(es) to run the main entry point ```run.py``` under multi GPU settings. You can
directly overwrite configurations in the configuration file ```$CONFIG``` by setting property fields using ```=``` after
all command line arguments. (e.g., ```name=$EXP_NAME```,```solver.epochs=400```,```dataloader.batchsize=4```)

For testing and inference, remember to set up the testing data correctly under each configuration files and switch the
```mode``` field in the configurations into ```test``` (i.e., ```mode=test```).

### 2. Debugging
If you want to debug your code without an additional job launcher, you can also directly run the file ```run.py``` . 
As an example, you can directly run the file for debugging with
```bash
# Single card direct run for debugging purposes
$ python run.py --config-path ${PROJ_PATH}/configs/final/ --config-name ${EXP_CONFIG_NAME}.yaml \
                num_gpu=1 hydra.run.dir=. hydra.output_subdir=null hydra/job_logging=disabled hydra/hydra_logging=disabled \
                debug.flag=True debug.debug_size=1 dataloader.batchsize=2 debug.hard_debug=True name=Debug_test
```

