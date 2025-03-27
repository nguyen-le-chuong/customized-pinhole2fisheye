import os
import cv2
import numpy as np
import json

class FisheyeTransformation:
    def __init__(self, generator, rotate_trans_matrix):
        self._generator = generator
        self._rotate_trans_matrix = rotate_trans_matrix

    def transform_polygon(self, polygon, rotate_matrix):
        """
        Transforms polygon points with a rotation matrix.
        Args:
            polygon (list): List of polygon points (x, y).
            rotate_matrix (np.array): 3x3 rotation matrix.
        Returns:
            transformed_polygon (list): Transformed polygon points.
        """
        # Convert polygon points to homogeneous coordinates (x, y, 1)
        transformed_polygon = []
        for point in polygon:
            homogenous_point = np.array([point[0], point[1], 1])
            transformed_point = np.matmul(rotate_matrix, homogenous_point)
            transformed_polygon.append((transformed_point[0], transformed_point[1]))
        return transformed_polygon

    def visualize_transformed_polygon(self, image, polygon, output_dir, output_filename):
        """
        Visualizes the transformed polygon on the image and saves it.
        Args:
            image (np.array): The transformed image.
            polygon (list): List of transformed polygon points.
            output_dir (str): Directory to save the output image.
            output_filename (str): Filename for the saved image.
        """
        # Convert polygon points to a numpy array of shape (n, 2)
        polygon = np.array(polygon, np.int32)
        polygon = polygon.reshape((-1, 1, 2))

        # Draw the polygon on the image (color: red, thickness: 2)
        cv2.polylines(image, [polygon], isClosed=True, color=(0, 0, 255), thickness=2)

        # Save the output image
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, image)
        print(f"Saved transformed image with polygon to {output_path}")

    def test_transformation_with_one_file(self, src_img_path, src_ann_path, src_json_path, dst_img_dir):
        """
        Test fisheye transformation and polygon visualization for a single image and save the result.
        Args:
            src_img_path (str): Path to the source image.
            src_ann_path (str): Path to the source annotation image.
            src_json_path (str): Path to the source JSON annotation file containing polygons.
            dst_img_dir (str): Directory to save the output image.
        """
        # Ensure the annotation and JSON files exist
        if not os.path.exists(src_ann_path) or not os.path.exists(src_json_path):
            print(f"Annotation or JSON file not found for image {src_img_path}. Skipping...")
            return

        # Load the source image, annotations, and polygons
        src_image = cv2.imread(src_img_path)
        src_annotation_color = cv2.imread(src_ann_path)
        with open(src_json_path, 'r') as f:
            json_data = json.load(f)

        # Apply fisheye transformation
        transformed_image = self._generator.transFromColor(src_image)
        transformed_annotation_color = self._generator.transFromColor(src_annotation_color)

        # Visualize and save polygons on the transformed image
        for obj in json_data["objects"]:
            transformed_polygon = self.transform_polygon(obj["polygon"], self._rotate_trans_matrix)

            # Visualize the transformed polygon
            output_filename = os.path.basename(src_img_path).replace(".png", "_transformed_polygon.png")
            self.visualize_transformed_polygon(transformed_image, transformed_polygon, dst_img_dir, output_filename)

        print(f"Test completed for {os.path.basename(src_img_path)}. Check {dst_img_dir} for saved result.")
# Example paths (adjust to your actual file locations)
src_img_path = "/home/vinai/Workspace/chuongnl/data/cityscape/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png"  # Replace with the actual image file path
src_ann_path = "/path/to/your/annotation.png"  # Replace with the actual annotation file path
src_json_path = "/path/to/your/polygon_annotations.json"  # Replace with the actual JSON file path
dst_img_dir = "/path/to/your/output/directory"  # Replace with the output directory

# Assuming _generator and _rotate_trans_matrix are already defined in your context
fisheye_transformer = FisheyeTransformation(generator, rotate_trans_matrix)

# Call the method to test the transformation with the image
fisheye_transformer.test_transformation_with_one_file(src_img_path, src_ann_path, src_json_path, dst_img_dir)
