
## Data

* Note: As some of our users requested the mapping between HM3D object id in SceneVerse to HM3D-semantics, we have added an additional file ([HM3D_tgtID2objID.zip](assets/HM3D_tgtID2objID.zip)) to obtain this mapping. The json file for each scene contains a dictionary of ```{<sceneverse_objid>:[hm3d_objid, hm3d_label]}```.

### Data Processing

We release a data preprocessing exemplar for 3RScan, MultiScan, ARKitScenes and Structured3D, with more details [here](preprocess/README.md).

We also release the [scripts](preprocess/ssg/README.md) for scene graph generation.

### Data Download
We currently host our data on G-drive and request all applicants to fill out the form from [here](https://forms.gle/AXMk7MH6bFXpCqd99).

You should see one or multiple zip file segments for each dataset we provided. For datasets with multiple segments (e.g., ARKitScenes), you can unzip the files with:

```shell
# Directories with multiple zip segments
$ ls ARKitScenes/
  -> ARKitScenes.zip  ARKitScenes.z01

# Unzip from all zip segments
$ cd ARKitScenes/
$ zip -F ARKitScenes.zip --out combined.zip
$ unzip combined.zip
```

After unzipping, the files are organized as:
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

### Data Visualization
For data browsing, we experimented with NVIDIA CUDA 11.8 on Ubuntu 22.04 and require the following steps:
```shell
$ conda create -n sceneverse python=3.9
$ pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118
$ pip install numpy open3d
```

We provide a short script for visualizing scene and language data, you can use it with:
```shell
# Visualize scene and instance data
$ python visualize_data.py --root <PATH_TO_DOWNLOAD> --dataset <DATASET>
# Visualize language data
$ python visualize_data.py --root <PATH_TO_DOWNLOAD> --dataset <DATASET> --vis_refer
```

As our data contains scenes from existing datasets, please read carefully about the term of use for each dataset we provided in the form.

### Provided Language Types

We list the available data in the current version of SceneVerse in the table below:

|   Dataset    | Object Caption | Scene Caption | Ref-Annotation   | Ref-Pairwise<br>```rel2``` | Ref-MultiObject<br>```relm``` | Ref-Star<br>```star``` | Ref-Chain (Optional)<br>```chain``` |
|:------------:|:--------------:|:-------------:|------------------|-------------------------|-------------------------------|-----------------------|------------------------------------|
|   ScanNet    |       ✅        |       ✅       | ScanRefer<br>Nr3D | ✅              | ✅                             | ✅           | ✅       |
|  MultiScan   |       ✅        |       ✅       | ✅ | ✅              | ✅                             | ✅           | ✅       |
| ARKitScenes  |       ✅        |       ✅       | ✅ | ✅              | ✅                             | ✅           | ✅       |
|     HM3D     |  ```template```   |       ✅       | ✅ | ✅              | ✅                             | ✅           | ✅       |
|    3RScan    |       ✅        |       ✅       | ❌ | ✅              | ✅                             | ✅           | ✅       |
| Structured3D | ```template``` |       ✅       | ❌ | ✅              | ✅                             | ✅           |    ❌     |
|   ProcTHOR   | ```template``` |    ❌     | ❌ | ```template```              | ```template```                   | ```template```            |    ❌     |

For the generated object referrals, we provide both the direct template-based generations ```template``` and the LLM-refined versions ```gpt```.
Please refer to our supplementary for the description of selected ```pair-wise``` / ```multi-object``` / ```star``` types. We also
provide the ```chain``` type which contains language using obejct A to refer B and then B to refer the target object C. As we found 
the ```chain``` type could sometimes lead to unnatural descriptions, we did not discuss it in the main paper. Feel free to inspect
and use it in your projects.

For the remaining data, we hope to further refine and update our data in the following weeks, stay tuned!
