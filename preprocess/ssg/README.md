## Scene Graph Generation

We have released the scripts to generate 3D scene graphs for the datasets released in SceneVerse. 

### Example Usage
Construct the following OmegaConfig and run ```python ssg_main.py```
```python
    cfg = OmegaConf.create({
        'dataset': 'MultiScan',
        'scene_path': 'path/to/SceneVerse',
        'rels_save_path': './tmp',
        'visualize': True,
        'num_workers': 1,
    })
```
Note that the current implementation of scene graph generation assumes a default viewing direction of "+y" from outside the 3D scan. Therefore, it can be adapted for situated understanding by allowing manual presetting of the position and viewing direction.