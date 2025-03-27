from data.FishEyeGenerator import FishEyeGenerator
import os
import cv2
import json
from math import tan, sqrt
import numpy as np
from tqdm import tqdm
import PIL.Image     as Image
# Paths for BDD dataset
dst_dir = "/home/vinai/Workspace/chuongnl/data/bdd100k_fisheye/images/train/"
lane_new = "/home/vinai/Workspace/chuongnl/data/bdd100k_fisheye/lane/train/"
da_new = "/home/vinai/Workspace/chuongnl/data/bdd100k_fisheye/da/train/"
det_new = "/home/vinai/Workspace/chuongnl/data/bdd100k_fisheye/det/train/"
img_dir = "/home/vinai/Workspace/chuongnl/data/bdd10k/bdd100k/images/100k/train/"
lane_dir = "/home/vinai/Workspace/chuongnl/data/bdd10k/bdd_lane_gt/train/"
da_dir = "/home/vinai/Workspace/chuongnl/data/bdd10k/bdd_seg_gt/train/"
det_dir = "/home/vinai/Workspace/chuongnl/data/bdd10k/data2/zwt/bdd/bdd100k/labels/100k/train/"

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

    def generate(self, src_dir, src_annot_dir, src_annot_dir_l, src_det, dst_dir, dst_annot_dir, dst_annot_dir_l, dst_det, prefix):
        # Ensure destination directories exist
        os.makedirs(dst_dir, exist_ok=True)
        os.makedirs(dst_annot_dir, exist_ok=True)
        os.makedirs(dst_annot_dir_l, exist_ok=True)

        # Get list of images in the source directory
        image_list = sorted([image for image in os.listdir(src_dir) if image.endswith(".jpg")])

        for count, image in enumerate(image_list):
            # Construct corresponding annotation filenames
            base_name = image.replace(".jpg", "")
            annotation_name = f"{base_name}.png"
            det_name = f"{base_name}.json"

            src_image_path = os.path.join(src_dir, image)
            src_annot_path = os.path.join(src_annot_dir, annotation_name)
            src_annot_path_l = os.path.join(src_annot_dir_l, annotation_name)
            src_det_path = os.path.join(src_det, det_name)

            if not os.path.exists(src_annot_path) or not os.path.exists(src_annot_path_l):
                print(f"Annotation file {annotation_name} not found for image {image}. Skipping...")
                continue
            # with open(src_det_path, "r") as file:
            #     data = json.load(file)
            if os.path.exists(os.path.join(dst_annot_dir_l, prefix + annotation_name)):
                continue
            src_image = cv2.imread(src_image_path)
            src_annot_image = cv2.imread(src_annot_path, 0)
            src_annot_image_l = cv2.imread(src_annot_path_l, 0)

            if self._F_RAND_FLAG:
                self._generator.rand_f(self._F_RANGE)
            if self._EXT_RAND_FLAG:
                self._generator.rand_ext_params()
            # Apply fisheye transformation to the image
            result1 = self._generator.transFromColor(src_image)
            cv2.imwrite(os.path.join(dst_dir, prefix + image), result1)

            # Apply fisheye transformation to lane annotation
            result2 = self._generator.transFromGray(src_annot_image)
            cv2.imwrite(os.path.join(dst_annot_dir, prefix + annotation_name), result2)

            # Apply fisheye transformation to segmentation annotatio

        # print("All images and annotations processed!")
    def transform_polygon(self, polygon, img_shape):
        """
        Apply fisheye transformation to a polygon.

        Args:
            polygon (list of list of int): Original polygon points as [[x1, y1], [x2, y2], ...].
            img_shape (tuple): Shape of the transformed image (height, width, channels).

        Returns:
            list of list of int: Transformed polygon points.
        """
        transformed_polygon = []
        img_height, img_width = img_shape[:2]
        for x, y in polygon:
            # Normalize coordinates to the center of the image
            normalized_x = x - img_width / 2
            normalized_y = y - img_height / 2

            # Apply fisheye transformation
            radius = sqrt(normalized_x**2 + normalized_y**2)
            theta = radius / self._focal_len
            if radius != 0:
                transformed_x = tan(theta) * normalized_x / radius * self._focal_len
                transformed_y = tan(theta) * normalized_y / radius * self._focal_len
            else:
                transformed_x, transformed_y = 0, 0

            # Map back to original coordinate system
            new_x = int(transformed_x * self._generator._ratio + img_width / 2)
            new_y = int(transformed_y * self._generator._ratio + img_height / 2)
            transformed_polygon.append([new_x, new_y])
        return transformed_polygon
    def generate_ctsp_v2(self, src_img_dir, src_ann_dir, dst_img_dir, dst_ann_dir, prefix=""):
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
            for image_file in sorted(os.listdir(city_img_dir)):
                if not image_file.endswith(".png"):
                    continue

                base_name = image_file.replace("_leftImg8bit.png", "")
                annotation_file = f"{base_name}_gtFine_labelIds.png"
                annotation_file_color = f"{base_name}_gtFine_color.png"

                src_image_path = os.path.join(city_img_dir, image_file)
                src_annotation_path = os.path.join(city_ann_dir, annotation_file)
                src_annotation_path_color = os.path.join(city_ann_dir, annotation_file_color)

                if not os.path.exists(src_annotation_path):
                    print(f"Annotation file {annotation_file} not found for image {image_file}. Skipping...")
                    continue

                # Load image and annotation
                src_image = cv2.imread(src_image_path)
                src_annotation_color = cv2.imread(src_annotation_path_color)
                src_annotation = cv2.imread(src_annotation_path, 0)

                if self._F_RAND_FLAG:
                    self._generator.rand_f(self._F_RANGE)
                if self._EXT_RAND_FLAG:
                    self._generator.rand_ext_params()

                # For 2D bbox
                json_file = f"{base_name}_gtFine_polygons.json"
                src_json_path = os.path.join(city_ann_dir, json_file)
                with open(src_json_path, 'r') as f:
                    json_data = json.load(f)

                # Apply fisheye transformation
                transformed_image = self._generator.transFromColor(src_image)
                transformed_annotation_color = self._generator.transFromColor(src_annotation_color)
                transformed_annotation = self._generator.transFromGray(src_annotation)

                # Transform polygons
                transformed_objs = []
                for obj in json_data["objects"]:
                    obj["polygon"] = self.transform_polygon(obj["polygon"], transformed_image.shape)
                    transformed_objs.append(obj)
                json_data["objects"] = transformed_objs

                # Save transformed JSON file
                json_out_path = os.path.join(dst_city_ann_dir, prefix + json_file)
                with open(json_out_path, 'w') as json_out_file:
                    json.dump(json_data, json_out_file, indent=4)

                # Save transformed outputs
                cv2.imwrite(os.path.join(dst_city_img_dir, prefix + image_file), transformed_image)
                cv2.imwrite(os.path.join(dst_city_ann_dir, prefix + annotation_file), transformed_annotation)
                cv2.imwrite(os.path.join(dst_city_ann_dir, prefix + annotation_file_color), transformed_annotation_color)
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
                    # Each "polygon" field is a list of points forming the polygon
                    # Draw the polygon on the black image (white lines on a black background)
                    # cv2.polylines(black_image, [polygon_points], isClosed=True, color=(255, 255, 255), thickness=2)
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

    def save_transformed_images_with_polygons(self, src_image_path, src_json_path, dst_img_path, dst_polygon_path):
        """
        Save both the transformed original image and the transformed image with polygons drawn on a black background.

        Args:
            src_image_path (str): Path to the original image.
            src_json_path (str): Path to the JSON file containing polygon annotations.
            dst_img_path (str): Path where the transformed original image will be saved.
            dst_polygon_path (str): Path where the transformed image with polygons will be saved.
        """
        # Read the original image
        src_image = cv2.imread(src_image_path)
        if src_image is None:
            raise ValueError(f"Unable to read the image from {src_image_path}")

        # Get the original image size
        image_height, image_width = src_image.shape[:2]

        # Apply fisheye transformation to the original image
        transformed_image = self._generator.transFromColor(src_image)

        # Create a black image of the same size as the source image
        black_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

        # Load the JSON annotation
        with open(src_json_path, 'r') as f:
            json_data = json.load(f)
        transformed_objects = []
        # Draw the polygons on the black image
        for obj in json_data["objects"]:
            black_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

            # Each "polygon" field is a list of points forming the polygon
            polygon_points = np.array(obj["polygon"], np.int32).reshape((-1, 1, 2))
            # Draw the polygon on the black image (white lines on a black background)
            # cv2.polylines(black_image, [polygon_points], isClosed=True, color=(255, 255, 255), thickness=2)
            for point in obj["polygon"]:
                x, y = point  # Unpack the point tuple (x, y)
                cv2.circle(black_image, (x, y), radius=5, color=(255, 255, 255), thickness=-1)

        # Apply fisheye transformation to the black image with polygons
            transformed_polygon_image = self._generator.transFromColor(black_image)
            transformed_polygon_points = []
            for point in polygon_points:
                x, y = point
                # In the transformed point image, extract the color of the point (assuming it is red)
                if transformed_polygon_image[y, x][0] == 255:  # Check if the point is red (255, 0, 0)
                    transformed_polygon_points.append([x, y])
            transformed_objects.append({
                "label": obj["label"],
                "polygon": transformed_polygon_points
             })
        json_data["objects"] = transformed_objects
        # Save both transformed images
        cv2.imwrite(dst_img_path, transformed_image)  # Transformed original image
        cv2.imwrite(dst_polygon_path, transformed_polygon_image)  # Transformed polygon image

src_img_dir = "/mnt/mmlab2024nas/huycq/chuong/vinai/YOLOP/data/cityscape/leftImg8bit/train/"
src_ann_dir = "/mnt/mmlab2024nas/huycq/chuong/vinai/YOLOP/data/cityscape/gtFine/train/"
dst_img_dir = "/mnt/mmlab2024nas/huycq/chuong/vinai/YOLOP/data/cityscape_fs/leftImg8bit/train/"
dst_ann_dir = "/mnt/mmlab2024nas/huycq/chuong/vinai/YOLOP/data/cityscape_fs/gtFine/train/"
def test():
    DT = FESetsGenerator([960, 1280], focal_len=350)
    DT.set_ext_param_range([10, 10, 10, 0, 0, 0])
    DT.rand_ext_params()
    # Uncomment to randomize focal length
    # DT.rand_f()
    # DT.generate(img_dir, lane_dir, da_dir, det_dir, dst_dir, lane_new, da_new, det_new, prefix='')
    DT.generate_ctsp(src_img_dir, src_ann_dir, dst_img_dir, dst_ann_dir)
    # DT.save_transformed_image_with_polygons()
#     src_image_path = "../data/cityscape/leftImg8bit/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png"
#     src_json_path = "/home/vinai/Workspace/chuongnl/data/cityscape/gtFine/val/frankfurt/frankfurt_000000_000294_gtFine_polygons.json"
#     dst_img_path = "transformed_image.png"  # Output path for transformed original image
#     dst_polygon_path = "transformed_polygons.png"  # Output path for transformed polygon image

# # Call the function
#     DT.save_transformed_images_with_polygons(src_image_path, src_json_path, dst_img_path, dst_polygon_path)

if __name__ == "__main__":
    test()
