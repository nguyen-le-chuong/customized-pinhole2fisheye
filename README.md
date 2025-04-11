# Customized Fisheye Augmentation 

This scripts to conver the pinholde dataset to fiseye dataset include image, semantic segmentation, instance segmentation, polygons and 2d bouding box. Modify for Cityscape and BDD100K (working) dataset
## Requirements

## Convert to Fisheye

```bash
python DataGenerator.py --base_dir "path_contain_datasets" --mode ['train', 'val'] --dataset ['cityscape', 'bdd100k']
``` 


## Convert 2D bouding box follow COCO format

create directory:
```bash
mkir /path_of_fisheye_dataset/annotations
```
convert:
```bash
python data2coco.py --datadir "path_of_fisheye_dataset" --outdir "path_of_fisheye_dataset/annotations"
```

## Visualization

```bash
mkdir outputs

python inspect_coco.py --coco_dir "path_of_fisheye_dataset" --num_examples 5
```

<!-- | Original image        | Fisheye        |
|-----------------|----------------|
| ![Image 1](outputs/frankfurt_000001_038418_leftImg8bit.png) | ![Image 2](outputs/image(2).png) |
| Segmentation        | 2d Bouding box        |
| ![Image 3](outputs/image(1).png) | ![Image 4](outputs/image(3).png) | -->

<table>
  <tr>
    <th style="width: 200px; height: 200px; text-align: center;">Original Image</th>
    <th style="width: 200px; height: 200px; text-align: center;">Fisheye Transformation</th>
  </tr>
  <tr>
    <td style="width: 200px; height: 200px; text-align: center; vertical-align: middle;">
      <img src="outputs/frankfurt_000001_038418_leftImg8bit.png" alt="Original" style="width: 100%; height: auto;">
    </td>
    <td style="width: 200px; height: 200px; text-align: center; vertical-align: middle;">
      <img src="outputs/image(2).png" alt="Fisheye" style="width: 100%; height: auto;">
    </td>
  </tr>
  <tr>
    <th style="width: 200px; height: 200px; text-align: center;">Segmentation Output</th>
    <th style="width: 200px; height: 200px; text-align: center;">2D Bounding Box</th>
  </tr>
  <tr>
    <td style="width: 200px; height: 200px; text-align: center; vertical-align: middle;">
      <img src="outputs/image(1).png" alt="Segmentation" style="width: 100%; height: auto;">
    </td>
    <td style="width: 200px; height: 200px; text-align: center; vertical-align: middle;">
      <img src="outputs/image(3).png" alt="Bounding Box" style="width: 100%; height: auto;">
    </td>
  </tr>
</table>