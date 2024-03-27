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
- [2024-03] We release the data used in SceneVerse. Fill out the [form](https://docs.google.com/forms/d/1x8cCAkn86d6MyyY5PMvvS_qRrH7dV8_RKHeBX9sE3KU/edit?usp=drive_web) for the download link! As we are still under submission, training and inference codes/checkpoints will come shortly, stay tuned!
- [2024-01] We release SceneVerse on ArXiv. Checkout our [paper](https://arxiv.org/abs/2401.09340) and [website](https://scene-verse.github.io/).

## TODO


## Data Download
We currently host our data on G-drive and request all applicants to fill out the form from [here](https://docs.google.com/forms/d/1x8cCAkn86d6MyyY5PMvvS_qRrH7dV8_RKHeBX9sE3KU/edit?usp=drive_web).

For each dataset we provided, you should see one or multiple zip file segments. For datasets with multiple segments (e.g., ARKitScenes), you can unzip the files with:

```shell
# Directories with multiple zip segments
$ ls ARKitScenes/
  -> ARKitScenes.zip  ARKitScenes.z01

# Unzip from all zip segments
$ cd ARKitScenes/
$ zip -F ARKitScenes.zip --out combined.zip
$ unzip combined.zip
```
After unzipping, files are organized as:
```shell
ARKitScenes/
|-- scan_data                   # Point cloud data
  |-- instance_id_to_label      # Reorganized instance id to label mapping
  |-- pcd_with_global_alignment # Aligned scene point clouds
|-- annotations                 # Language annotations
  |-- splits
    |-- train_split.txt         # For all datasets, we provide training split
    |-- val_split.txt           # For datasets with evaluation sets
  |-- <language_type>.json      # For datasets except for ScanNet, language for ScanNet is located at annotations/refer
```

We also provide a short script for visualizing scene and language data, you can use it like
```shell
# Visualize scene and instance data
$ python visualize_data.py --root <PATH_TO_DOWNLOAD> --dataset <DATASET>
# Visualize language data
$ python visualize_data.py --root <PATH_TO_DOWNLOAD> --dataset <DATASET> --vis_refer
```

As our data contains scenes from existing datasets, please read carefully about the term of use for each dataset in the license information we provided.


## BibTex
```bibtex
@article{jia2024sceneverse,
  title={SceneVerse: Scaling 3D Vision-Language Learning for Grounded Scene Understanding},
  author={Jia, Baoxiong and Chen, Yixin and Yu, Huangyue and Wang, Yan and Niu, Xuesong and Liu, Tengyu and Li, Qing and Huang, Siyuan},
  journal={arXiv preprint arXiv:2401.09340},
  year={2024}
}
```
