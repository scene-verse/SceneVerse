## Data Processing

We have released a preprocessing script for 3RScan designed to provide a comprehensive framework for data preparation. The process involves the following steps:

- Import raw meshes and annotations from each dataset.  
- Extract vertices from the mesh and assign both instance and semantic labels to them.  
- Map the dataset-specific semantic labels to ScanNet 607. This is optional for SceneVerse training but may be required for closed-vocab training ([example](https://github.com/scene-verse/SceneVerse/blob/b936f96b61614bec32282e5eed7de844d1a7a330/preprocess/rscan.py#L58)).
- Axis Alignment: Rotate the 3D point clouds so that most 3D object bounding boxes are axis-aligned. This follows ScanRefer, and is currently implemented as a heuristic search ([example](https://github.com/scene-verse/SceneVerse/blob/b936f96b61614bec32282e5eed7de844d1a7a330/preprocess/rscan.py#L95)).  
- Translation Alignment: Translate the 3D point clouds so that its origin at the center on the floor ([example](https://github.com/scene-verse/SceneVerse/blob/b936f96b61614bec32282e5eed7de844d1a7a330/preprocess/rscan.py#L102)).  
- Color Alignment: The color value should be within the [0, 255] range ([example](https://github.com/scene-verse/SceneVerse/blob/b936f96b61614bec32282e5eed7de844d1a7a330/preprocess/rscan.py#L98)).
- Point subsampling: subsample the point clouds if the number of points exceeds 240K.
    ```python
    PTS_LIMIT = 240000
    if out_points.shape[0] > PTS_LIMIT:
        pcd_idxs = np.random.choice(out_points.shape[0], size=PTS_LIMIT, replace=False)
        out_points = out_points[pcd_idxs]
        out_colors = out_colors[pcd_idxs]
        instance_labels = instance_labels[pcd_idxs]
    ```

The detailed steps may vary between datasets. Please note the translation and color alignment are critical as they can significantly impact performance. Axis alignment, which requires 3D bounding box annotations, may result in slight but not severe degradation.

### 3RScan
To reproduce the data preprocessing, download [3RScan](https://waldjohannau.github.io/RIO/) and run:
```shell
# Preprocess 3RScan 
$ python rscan.py
```
Adjust the `data_root`, `save_root` and `num_workers` accordingly.

### HM3D
We release a script to generate the mapping between HM3D object id in SceneVerse to HM3D-semantics in ```sceneverse2hmsemantic.py```. When running with the target HM3D-semantics scenes, it will generate a dictionary of ```{<sceneverse_objid>:[hm3d_objid, hm3d_label]}``` for each scene.


## Prepare for your custom datasets
To prepare your custom data for inference, follow the previous steps and  the example script for 3RScan. A convenient way for verification is to use the `visualize_data.py`. If everything is correct, you should observe the colored point clouds displayed similarly to those in the released version of SceneVerse.
