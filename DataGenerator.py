from data.FishEyeGenerator import FishEyeGenerator
import os
import cv2
import json
import argparse
# import os
from math import tan, sqrt
import numpy as np
from tqdm import tqdm
import PIL.Image     as Image
# Paths for BDD dataset
# dst_dir = "/home/vinai/Workspace/chuongnl/data/bdd100k_fisheye/images/train/"
# lane_new = "/home/vinai/Workspace/chuongnl/data/bdd100k_fisheye/lane/train/"
# da_new = "/home/vinai/Workspace/chuongnl/data/bdd100k_fisheye/da/train/"
# det_new = "/home/vinai/Workspace/chuongnl/data/bdd100k_fisheye/det/train/"
# img_dir = "/home/vinai/Workspace/chuongnl/data/bdd10k/bdd100k/images/100k/train/"
# lane_dir = "/home/vinai/Workspace/chuongnl/data/bdd10k/bdd_lane_gt/train/"
# da_dir = "/home/vinai/Workspace/chuongnl/data/bdd10k/bdd_seg_gt/train/"
# det_dir = "/home/vinai/Workspace/chuongnl/data/bdd10k/data2/zwt/bdd/bdd100k/labels/100k/train/"

class FESetsGenerator:

    def __init__(self, dst_shape, focal_len=350):
        self._generator = FishEyeGenerator(focal_len, dst_shape)

        self._F_RAND_FLAG = False
        self._F_RANGE = [200, 400]
        self._focal_len = focal_len
        self._EXT_RAND_FLAG = False
        self._EXT_PARAM_RANGE = [5, 5, 10, 0.3, 0.3, 0.4]
        self._generator.set_ext_param_range(self._EXT_PARAM_RANGE)

    def set_ext_param_range(self, ext_param):
        for i in range(6):
            self._EXT_PARAM_RANGE[i] = ext_param[i]
        self._generator.set_ext_param_range(self._EXT_PARAM_RANGE)

    def rand_ext_params(self):
        self._EXT_RAND_FLAG = True

    def set_ext_params(self, ext_params):
        self._generator.set_ext_params(ext_params)
        self._EXT_RAND_FLAG = False

    def set_f(self, focal_len):
        self._generator.set_f(focal_len)
        self._F_RAND_FLAG = False

    def rand_f(self, f_range=[200, 400]):
        self._F_RANGE = f_range
        self._F_RAND_FLAG = True
    def generate_bdd100k(self, src_img_dir, src_ann_dir, dst_img_dir, dst_ann_dir, prefix=""):
        """
        Generate fisheye transformations for the BDD100K dataset.

        Args:
            src_img_dir (str): Source directory for images.
            src_ann_dir (str): Source directory for annotations (e.g., labels).
            dst_img_dir (str): Destination directory for transformed images.
            dst_ann_dir (str): Destination directory for transformed annotations.
            prefix (str): Prefix to add to the transformed file names.
        """
        # Ensure destination directories exist
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_ann_dir, exist_ok=True)

        # Process each image in the dataset
        for image_file in tqdm(sorted(os.listdir(src_img_dir))):
            if not image_file.endswith((".jpg", ".png")):  # Adjust based on BDD100K image format
                continue

            base_name = os.path.splitext(image_file)[0]
            annotation_file = f"{base_name}.json"

            src_image_path = os.path.join(src_img_dir, image_file)
            src_annotation_path = os.path.join(src_ann_dir, annotation_file)

            if not os.path.exists(src_annotation_path):
                print(f"Annotation file {annotation_file} not found for image {image_file}. Skipping...")
                continue

            # Skip if already processed
            if os.path.exists(os.path.join(dst_ann_dir, prefix + annotation_file)):
                continue

            # Load image and annotation
            src_image = cv2.imread(src_image_path)
            with open(src_annotation_path, 'r') as f:
                json_data = json.load(f)

            if self._F_RAND_FLAG:
                self._generator.rand_f(self._F_RANGE)
            if self._EXT_RAND_FLAG:
                self._generator.rand_ext_params()

            # Prepare transformed annotation
            transformed_objects = []
            image_height, image_width = src_image.shape[:2]
            for obj in json_data["frames"][0]["objects"]:
                # Skip objects without polygons
                if "poly2d" not in obj:
                    continue

                # Draw the polygon on a black background
                black_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
                for poly in obj["poly2d"]:
                    points = np.array(poly["vertices"], dtype=np.int32)
                    cv2.polylines(black_image, [points], isClosed=True, color=(0, 0, 255), thickness=2)

                # Apply fisheye transformation
                transformed_polygon_image = self._generator.transFromColor(black_image)

                # Extract transformed points
                red_mask = np.all(transformed_polygon_image == [0, 0, 255], axis=-1)
                transformed_points = np.column_stack(np.where(red_mask))

                # Recreate the object with transformed polygons
                transformed_objects.append({
                    "category": obj["category"],
                    "poly2d": [{
                        "vertices": transformed_points.tolist(),
                        "closed": poly["closed"],
                        "types": poly["types"]
                    }]
                })

            # Update JSON metadata
            json_data["frames"][0]["objects"] = transformed_objects
            json_data["frames"][0]["size"] = {"width": 640, "height": 640}  # Adjust if resizing

            json_out_path = os.path.join(dst_ann_dir, prefix + annotation_file)
            with open(json_out_path, 'w') as json_out_file:
                json.dump(json_data, json_out_file, indent=4)

            # Apply fisheye transformation to the image
            transformed_image = self._generator.transFromColor(src_image)

            # Save transformed outputs
            cv2.imwrite(os.path.join(dst_img_dir, prefix + image_file), transformed_image)

    def generate_ctsp(self, src_img_dir, src_ann_dir, dst_img_dir, dst_ann_dir, prefix=""):
        """
        Generate fisheye transformations for Cityscapes dataset.

        Args:
            src_img_dir (str): Source directory for images (e.g., leftImg8bit/train).
            src_ann_dir (str): Source directory for annotations (e.g., gtFine/train).
            dst_img_dir (str): Destination directory for transformed images.
            dst_ann_dir (str): Destination directory for transformed annotations.
            prefix (str): Prefix to add to the transformed file names.
        """
        # Ensure destination directories exist
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_ann_dir, exist_ok=True)

        # Process each city in the dataset
        for city in os.listdir(src_img_dir):
            city_img_dir = os.path.join(src_img_dir, city)
            city_ann_dir = os.path.join(src_ann_dir, city)
            
            # Skip non-directory entries
            if not os.path.isdir(city_img_dir):
                continue
            
            # Create city-level subdirectories in destination
            dst_city_img_dir = os.path.join(dst_img_dir, city)
            dst_city_ann_dir = os.path.join(dst_ann_dir, city)
            os.makedirs(dst_city_img_dir, exist_ok=True)
            os.makedirs(dst_city_ann_dir, exist_ok=True)
            
            # Process each image in the city directory
            for image_file in tqdm(sorted(os.listdir(city_img_dir))):
                if not image_file.endswith(".png"):
                    continue

                base_name = image_file.replace("_leftImg8bit.png", "")
                annotation_file = f"{base_name}_gtFine_instanceIds.png"
                annotation_file_color = f"{base_name}_gtFine_color.png"

                src_image_path = os.path.join(city_img_dir, image_file)
                src_annotation_path = os.path.join(city_ann_dir, annotation_file)
                src_annotation_path_color = os.path.join(city_ann_dir, annotation_file_color)
                if os.path.exists(os.path.join(dst_city_ann_dir, prefix + annotation_file_color)):
                    continue
                if not os.path.exists(src_annotation_path):
                    print(f"Annotation file {annotation_file} not found for image {image_file}. Skipping...")
                    continue

                #Skip if already processed
                # if os.path.exists(os.path.join(dst_city_ann_dir, prefix + annotation_file)):
                #     continue

                # Load image and annotation
                src_image = cv2.imread(src_image_path)
                src_annotation_color = cv2.imread(src_annotation_path_color)
                # print(src_annotation_path_color)
                src_annotation = np.asarray(Image.open(src_annotation_path))

                if self._F_RAND_FLAG:
                    self._generator.rand_f(self._F_RANGE)
                if self._EXT_RAND_FLAG:
                    self._generator.rand_ext_params()
                # # for 2d bbox
                image_height, image_width = src_image.shape[:2]
                json_file = f"{base_name}_gtFine_polygons.json"
                src_json_path = os.path.join(city_ann_dir, json_file)
                with open(src_json_path, 'r') as f:
                    json_data = json.load(f)
                black_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
                transformed_objects = []
                # Draw the polygons on the black image
                for obj in json_data["objects"]:
                    black_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
                    for point in obj["polygon"]:
                        x, y = point  # Unpack the point tuple (x, y)
                        cv2.circle(black_image, (x, y), radius=5, color=(0, 0, 255), thickness=-1)

                # Apply fisheye transformation to the black image with polygons
                    transformed_polygon_image = self._generator.transFromColor(black_image)
                    transformed_polygon_points = []
                    red_mask = np.all(transformed_polygon_image == [0, 0, 255], axis=-1)

                    # Use np.where to get the coordinates of red points
                    red_points = np.column_stack(np.where(red_mask))

                    # red_points now contains the transformed polygon points (x, y)
                    transformed_polygon_points = red_points.tolist()
                    transformed_objects.append({
                        "label": obj["label"],
                        "polygon": transformed_polygon_points
                    })
                json_data["imgHeight"] = 960
                json_data["imgWidth"] = 1280
                json_data["objects"] = transformed_objects
                json_out_path = os.path.join(dst_city_ann_dir, prefix + json_file)
                with open(json_out_path, 'w') as json_out_file:
                    json.dump(json_data, json_out_file, indent=4)  

                # Apply fisheye transformation
                transformed_image = self._generator.transFromColor(src_image)
                transformed_annotation_color = self._generator.transFromColor(src_annotation_color)
                transformed_annotation = self._generator.transFromGray(src_annotation)

                          
                # Save transformed outputs
                cv2.imwrite(os.path.join(dst_city_img_dir, prefix + image_file), transformed_image)
                cv2.imwrite(os.path.join(dst_city_ann_dir, prefix + annotation_file), transformed_annotation.astype(np.uint16))
                cv2.imwrite(os.path.join(dst_city_ann_dir, prefix + annotation_file_color), transformed_annotation_color)

                # print(f"Processed {image_file} for city {city}.")

    def generate_bdd100k(self, src_img_dir, src_ann_dir, dst_img_dir, dst_ann_dir, prefix=""):
        """
        Generate fisheye transformations for the BDD100K dataset.

        Args:
            src_img_dir (str): Source directory for images.
            src_ann_dir (str): Source directory for annotations (e.g., labels).
            dst_img_dir (str): Destination directory for transformed images.
            dst_ann_dir (str): Destination directory for transformed annotations.
            prefix (str): Prefix to add to the transformed file names.
        """
        # Ensure destination directories exist
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_ann_dir, exist_ok=True)

        # Process each image in the dataset
        for image_file in tqdm(sorted(os.listdir(src_img_dir))):
            if not image_file.endswith((".jpg", ".png")):  # Adjust based on BDD100K image format
                continue

            base_name = os.path.splitext(image_file)[0]
            annotation_file = f"{base_name}.json"

            src_image_path = os.path.join(src_img_dir, image_file)
            src_annotation_path = os.path.join(src_ann_dir, annotation_file)

            if not os.path.exists(src_annotation_path):
                print(f"Annotation file {annotation_file} not found for image {image_file}. Skipping...")
                continue

            # Skip if already processed
            if os.path.exists(os.path.join(dst_ann_dir, prefix + annotation_file)):
                continue

            # Load image and annotation
            src_image = cv2.imread(src_image_path)
            with open(src_annotation_path, 'r') as f:
                json_data = json.load(f)

            if self._F_RAND_FLAG:
                self._generator.rand_f(self._F_RANGE)
            if self._EXT_RAND_FLAG:
                self._generator.rand_ext_params()

            # Prepare transformed annotation
            transformed_objects = []
            image_height, image_width = src_image.shape[:2]
            for obj in json_data["frames"][0]["objects"]:
                # Skip objects without polygons
                if "poly2d" not in obj:
                    continue

                # Draw the polygon on a black background
                black_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
                for poly in obj["poly2d"]:
                    points = np.array(poly["vertices"], dtype=np.int32)
                    cv2.polylines(black_image, [points], isClosed=True, color=(0, 0, 255), thickness=2)

                # Apply fisheye transformation
                transformed_polygon_image = self._generator.transFromColor(black_image)

                # Extract transformed points
                red_mask = np.all(transformed_polygon_image == [0, 0, 255], axis=-1)
                transformed_points = np.column_stack(np.where(red_mask))

                # Recreate the object with transformed polygons
                transformed_objects.append({
                    "category": obj["category"],
                    "poly2d": [{
                        "vertices": transformed_points.tolist(),
                        "closed": poly["closed"],
                        "types": poly["types"]
                    }]
                })

            # Update JSON metadata
            json_data["frames"][0]["objects"] = transformed_objects
            json_data["frames"][0]["size"] = {"width": 640, "height": 640}  # Adjust if resizing

            json_out_path = os.path.join(dst_ann_dir, prefix + annotation_file)
            with open(json_out_path, 'w') as json_out_file:
                json.dump(json_data, json_out_file, indent=4)

            # Apply fisheye transformation to the image
            transformed_image = self._generator.transFromColor(src_image)

            # Save transformed outputs
            cv2.imwrite(os.path.join(dst_img_dir, prefix + image_file), transformed_image)


def test(dataset, src_img_dir, src_ann_dir, dst_img_dir, dst_ann_dir):
    DT = FESetsGenerator([960, 1280], focal_len=350)
    DT.set_ext_param_range([10, 10, 10, 0, 0, 0])
    DT.rand_ext_params()

    if dataset == "cityscape":
        print("Processing Cityscape dataset...")
        DT.generate_ctsp(src_img_dir, src_ann_dir, dst_img_dir, dst_ann_dir)
    elif dataset == "bdd100k":
        print("Processing BDD100K dataset...")
        DT.generate_bdd100k(src_img_dir, src_ann_dir, dst_img_dir, dst_ann_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate transformed images and annotations.")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory for data")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "val"], help="Mode (train or val)")
    parser.add_argument("--dataset", type=str, required=True, choices=["cityscape", "bdd100k"], help="Dataset type")

    args = parser.parse_args()

    # Adjust directory paths based on dataset structure
    if args.dataset == "cityscape":
        src_img_dir = os.path.join(args.base_dir, "cityscape/leftImg8bit", args.mode)
        src_ann_dir = os.path.join(args.base_dir, "cityscape/gtFine", args.mode)
        dst_img_dir = os.path.join(args.base_dir, "cityscape_fs/leftImg8bit", args.mode)
        dst_ann_dir = os.path.join(args.base_dir, "cityscape_fs/gtFine", args.mode)
    elif args.dataset == "bdd100k":
        src_img_dir = os.path.join(args.base_dir, "bdd100k/images", args.mode)
        src_ann_dir = os.path.join(args.base_dir, "bdd100k/labels", args.mode)
        dst_img_dir = os.path.join(args.base_dir, "bdd100k_fs/images", args.mode)
        dst_ann_dir = os.path.join(args.base_dir, "bdd100k_fs/labels", args.mode)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Log the paths being used
    print(f"Dataset: {args.dataset}")
    print(f"Source image directory: {src_img_dir}")
    print(f"Source annotation directory: {src_ann_dir}")
    print(f"Destination image directory: {dst_img_dir}")
    print(f"Destination annotation directory: {dst_ann_dir}")

    # Call the test function with the computed paths
    test(args.dataset, src_img_dir, src_ann_dir, dst_img_dir, dst_ann_dir)

