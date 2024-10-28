# Training and Inference

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

## Checkpoints
We provide all available checkpoints under the same data directory, named after ```Checkpoints```. Here we provide detailed
descriptions of checkpoint in the table below:

| Setting              | Description                                                             | Corresponding Experiment                            | Checkpoint based on experiment setting                                                                                                                                                                                                                                                                           |
|----------------------|-------------------------------------------------------------------------|-----------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ```pre-trained```    | GPS model pre-trained on SceneVerse                                     | 3D-VL grounding (Tab.2)                             | [Model](https://drive.google.com/drive/folders/1FDjVaYZxHdMJgxB8stSHfI34Q7crItJc?usp=sharing)                                                                                                                                                                                                                                                                                                        |
| ```scratch```        | GPS model trained on datasets from scratch                              | 3D-VL grounding (Tab.2)<br/>SceneVerse-val (Tab. 3) | [ScanRefer](https://drive.google.com/drive/folders/1d7sGm_D7kyj6Fmo0f8b6DPrhWYUCtWVq?usp=sharing), [Sr3D](https://drive.google.com/drive/folders/1bKGgXot8Sc6BB2MWAfW_OGdu0iq0RWZt?usp=sharing), [Nr3D](https://drive.google.com/drive/folders/14K-UaIeg0GHWFoaonIFHTHZZbDotukzV?usp=sharing), [SceneVerse-val](https://drive.google.com/drive/folders/1CeWwLIPEuK0b35I_gbiwu_OiaUEE42jD?usp=drive_link)                                                                                                                                                                                                                                                            |
| ```fine-tuned```     | GPS model fine-tuned on datasets with grounding heads                   | 3D-VL grounding (Tab.2)                             | [ScanRefer](https://drive.google.com/drive/folders/1P5YprjIlBMAl0OQ38jgTDJyuFVIGiCMS?usp=sharing), [Sr3D](https://drive.google.com/drive/folders/1-LMYW6jy5wpqL_KlQQuvuSM7TDyo7M3g?usp=sharing), [Nr3D](https://drive.google.com/drive/folders/1sw-_hhF2__JgGCHE1yfyAQeNZ7jSrID0?usp=sharing)                                                                                                                                                                                                                                                                                |
| ```zero-shot```      | GPS model trained on SceneVerse without data from ScanNet and MultiScan | Zero-shot Transfer (Tab.3)                          | [Model](https://drive.google.com/drive/folders/11824oiZnaU8ChsNpH8zZKIT2i1PdJWSA?usp=sharing)                                                                                                                                                                                                                                                                                                        |
| ```zero-shot text``` | GPS                                                                     | Zero-shot Transfer (Tab.3)                          | [ScanNet](https://drive.google.com/drive/folders/1TKIhb7xgGzwDiAdvznwTpKzkcJnG7GD0?usp=sharing), [SceneVerse-val](https://drive.google.com/drive/folders/18f65Q6313sa-blLCyspqjZRmWpKJPh3M?usp=sharing)                                                                                                                                                                                              |
| ```text-ablation```  | Ablations on the type of language used during pre-training              | Ablation on Text (Tab.7)                            | [Template only](https://drive.google.com/drive/folders/1Xo6FkbThHP3uLUJMblt3zgJiM0n3RbVK?usp=sharing), [Template+LLM](https://drive.google.com/drive/folders/1w9Oi8nWKZXOW3BcA0eiC1bgp7snk8ZKS?usp=sharing)                                                                                                      |
| ```scene-ablation``` | Ablations on the use of synthetic scenes during pre-training            | Ablation on Scene (Tab.8)                           | [Real only](https://drive.google.com/drive/folders/1WZDf2BS7eG36NgGEdTuChICmVHF377is?usp=sharing), [S3D only](https://drive.google.com/drive/folders/1Zh4QfCs6l67ZeltvzOPZtokKkgkvxATc?usp=sharing), [ProcTHOR only](https://drive.google.com/drive/folders/1H9zm7vYxVn_zd2HYi49Js9R34AHnGi1d?usp=sharing)                                                                                                                                                                                                                                                                   |
| ```model-ablation``` | Ablations on the use of losses during pre-training                      | Ablation on Model Design (Tab.9)                    | [Refer only](https://drive.google.com/drive/folders/1yKF8dVPlcbKb-COcfUZbwcqWxt_uvzuc?usp=sharing), [Refer+Obj-lvl](https://drive.google.com/drive/folders/1C5L20UvTQj2my2t0BnqHZPsb_VaXxVjX?usp=sharing), [w/o Scene-lvl](https://drive.google.com/drive/folders/14jR43ils1-jop6K84hu1AqPqU9DcHucx?usp=sharing) |

To properly use the pre-trained checkpoints, you can use the ```pretrain_ckpt_path``` key in the configs:
```shell
# Directly testing the checkpoint
$ python launch.py --mode submitit --qos $QOS --partition $PARTITION --mem_per_gpu 80 \
                   --gpu_per_node 4 --config $CONFIG note=$NOTE name=$EXP_NAME mode=test \
                   pretrain_ckpt_path=$PRETRAIN_CKPT

# Fine-tuning with pre-trained checkpoint
$ python launch.py --mode submitit --qos $QOS --partition $PARTITION --mem_per_gpu 80 \
                   --gpu_per_node 4 --config $CONFIG note=$NOTE name=$EXP_NAME \
                   pretrain_ckpt_path=$PRETRAIN_CKPT
```
For fine-tuning the pre-trained checkpoint on datasets, you can use the fine-tuning config files provided under 
```configs/final/finetune```.