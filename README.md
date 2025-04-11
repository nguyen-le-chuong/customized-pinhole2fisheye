# Customized Fisheye Augmentation 

This script converts a pinhole dataset to a fisheye dataset, including images, semantic segmentation, instance segmentation, polygons, and 2D bounding boxes. It is compatible with the Cityscape and BDD100K (pending) datasets.



## Convert to Fisheye

```bash
python DataGenerator.py --base_dir "path_contain_datasets" --mode <train|val> --dataset <cityscape|bdd100k>
```

## Convert 2D Bounding Box to COCO Format

Create the directory:
```bash
mkdir /path_of_fisheye_dataset/annotations
```

Convert:
```bash
python data2coco.py --datadir "path_of_fisheye_dataset" --outdir "path_of_fisheye_dataset/annotations"
```

## Visualization

Create the output directory:
```bash
mkdir outputs
```

Run visualization:
```bash
python inspect_coco.py --coco_dir "path_of_fisheye_dataset" --num_examples 5
```

### Sample

<table>
  <tr>
    <th style="text-align: center;">Original Image</th>
    <th style="text-align: center;">Fisheye Transformation</th>
  </tr>
  <tr>
    <td style="text-align: center;">
      <img src="outputs/frankfurt_000001_038418_leftImg8bit.png" alt="Original" style="width: 60%; height: auto;">
    </td>
    <td style="text-align: center;">
      <img src="outputs/image(2).png" alt="Fisheye" style="width: 100%; height: auto;">
    </td>
  </tr>
  <tr>
    <th style="text-align: center;">Segmentation Output</th>
    <th style="text-align: center;">2D Bounding Box</th>
  </tr>
  <tr>
    <td style="text-align: center;">
      <img src="outputs/image(1).png" alt="Segmentation" style="width: 60%; height: auto;">
    </td>
    <td style="text-align: center;">
      <img src="outputs/image(3).png" alt="Bounding Box" style="width: 100%; height: auto;">
    </td>
  </tr>
</table>

## Acknowledgements

This repo builds upon the contributions of the following repositories:
- [FisheyeSeg](https://github.com/Yaozhuwa/FisheyeSeg/tree/master)
- [cityscapes-to-coco-conversion](https://github.com/TillBeemelmanns/cityscapes-to-coco-conversion)

