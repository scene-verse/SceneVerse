<h2 align="center">
  <span><img src="assets/logo025.png" width="4%" style="transform: translate(0,9px)"></span>
  <b>SceneVerse: Scaling 3D Vision-Language Learning for Grounded Scene Understanding</b>
</h2>

<div align="center" margin-bottom="6em">
<a target="_blank" href="https://buzz-beater.github.io/">Baoxiong Jia<sup>✶</sup></a>,
<a target="_blank" href="https://yixchen.github.io/">Yixin Chen<sup>✶</sup></a>,
<a target="_blank" href="https://scholar.google.com/citations?user=fKRgnIMAAAAJ/">Huangyue Yu</a>,
<a target="_blank" href="https://github.com/jetpackfirstme">Yan Wang</a>,
<a target="_blank" href="https://nxsedson.github.io/">Xuesong Niu</a>,
<a target="_blank" href="https://tengyu.ai/">Tengyu Liu</a>,
<a target="_blank" href="https://liqing-ustc.github.io/">Qing Li</a>,
<a target="_blank" href="https://siyuanhuang.com/">Siyuan Huang</a>

</div>
&nbsp;

<div align="center">
    <a href="https://arxiv.org/abs/2401.09340" target="_blank">
    <img src="https://img.shields.io/badge/Paper-arXiv-deepgreen" alt="Paper arXiv"></a>
    <a href="https://scene-verse.github.io" target="_blank">
    <img src="https://img.shields.io/badge/Project-Page-9cf" alt="Project Page"></a>
    <a href="https://youtu.be/UnujS0EVxKU" target="_blank">
    <img src="https://img.shields.io/badge/Video-YouTube-9966ff" alt="Video"></a>
    <a href="https://scene-verse.github.io" target="_blank">
    <img src="https://img.shields.io/badge/Data-SceneVerse-blue" alt="Data"></a>
    <a href="https://scene-verse.github.io" target="_blank">
    <img src="https://img.shields.io/badge/Model-GPS-darkorange" alt="Model"></a>
</div>
&nbsp;

<div align="left">
<img src="assets/overview.png" width="99%" alt="SceneVerse Teaser">
</div>

We propose SceneVerse, the first million-scale 3D vision-language dataset with 68K 3D indoor scenes and 2.5M vision-language pairs.  We demonstrate the scaling effect by (i) achieving state-of-the-art on all existing 3D visual grounding benchmarks and (ii) showcasing zero-shot transfer capabilities with our GPS (Grounded Pre-training for Scenes) model.

## News
- ![](https://img.shields.io/badge/New!-8A2BE2) [2024-10] Pre-trained checkpoints are now available, find detailed instructions in [TRAIN.md](TRAIN.md)!
- ![](https://img.shields.io/badge/New!-8A2BE2) [2024-09] The scripts for scene graph generation are released.
- [2024-07] Training & Inference code as well as preprocessing code is released and checkpoints & logs are on the way!
- [2024-07] Preprocessing codes for scenes used in SceneVerse are released.
- [2024-07] SceneVerse is accepted by ECCV 2024! Training and inference codes/checkpoints will come shortly, stay tuned!
- [2024-03] We release the data used in SceneVerse. Fill out the [form](https://forms.gle/AXMk7MH6bFXpCqd99) for the download link!
- [2024-01] We release SceneVerse on ArXiv. Checkout our [paper](https://arxiv.org/abs/2401.09340) and [website](https://scene-verse.github.io/).

## Data
See [DATA.md](DATA.md) for detailed instructions on data download, processing, visualization. The data inventory is listed below:

|   Dataset    | Object Caption | Scene Caption | Ref-Annotation   | Ref-Pairwise<br>```rel2``` | Ref-MultiObject<br>```relm``` | Ref-Star<br>```star``` | Ref-Chain (Optional)<br>```chain``` |
|:------------:|:--------------:|:-------------:|------------------|-------------------------|-------------------------------|-----------------------|------------------------------------|
|   ScanNet    |       ✅        |       ✅       | ScanRefer<br>Nr3D | ✅              | ✅                             | ✅           | ✅       |
|  MultiScan   |       ✅        |       ✅       | ✅ | ✅              | ✅                             | ✅           | ✅       |
| ARKitScenes  |       ✅        |       ✅       | ✅ | ✅              | ✅                             | ✅           | ✅       |
|     HM3D     |  ```template```   |       ✅       | ✅ | ✅              | ✅                             | ✅           | ✅       |
|    3RScan    |       ✅        |       ✅       | ❌ | ✅              | ✅                             | ✅           | ✅       |
| Structured3D | ```template``` |       ✅       | ❌ | ✅              | ✅                             | ✅           |    ❌     |
|   ProcTHOR   | ```template``` |    ❌     | ❌ | ```template```              | ```template```                   | ```template```            |    ❌     |


## Training and Inference
See [TRAIN.md](TRAIN.md) for the inventory of available checkpoints and detailed instructions on training and testing 
with pre-trained checkpoints. The checkpoint inventory is listed below:


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


## BibTex
```bibtex
@inproceedings{jia2024sceneverse,
  title={Sceneverse: Scaling 3d vision-language learning for grounded scene understanding},
  author={Jia, Baoxiong and Chen, Yixin and Yu, Huangyue and Wang, Yan and Niu, Xuesong and Liu, Tengyu and Li, Qing and Huang, Siyuan},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024}
}
```

## Acknowledgements
We thank the authors from [ScanRefer](https://github.com/daveredrum/ScanRefer), 
[ScanNet](https://github.com/ScanNet/ScanNet), 
[3RScan](https://github.com/WaldJohannaU/3RScan), [ReferIt3D](https://github.com/referit3d/referit3d), 
[Structured3D](https://github.com/bertjiazheng/Structured3D), 
[HM3D](https://github.com/matterport/habitat-matterport-3dresearch),
[ProcTHOR](https://github.com/allenai/procthor),
[ARKitScenes](https://github.com/apple/ARKitScenes), [MultiScan](https://github.com/smartscenes/multiscan) for
open-sourcing their awesome datasets. We also heavily adapted codes from [ScanQA](https://github.com/ATR-DBI/ScanQA), 
[SQA3D](https://github.com/SilongYong/SQA3D), and 
[3D-VisTA](https://github.com/3d-vista/3D-VisTA) for training and inference.
