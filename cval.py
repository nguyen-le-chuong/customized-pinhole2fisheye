from data.FishEyeGenerator import FishEyeGenerator
import os
import cv2
import json
# Paths for BDD dataset
dst_dir = "/home/vinai/Workspace/chuongnl/data/bdd100k_fisheye_te/images/val/"
lane_new = "/home/vinai/Workspace/chuongnl/data/bdd100k_fisheye_te/lane/val/"
da_new = "/home/vinai/Workspace/chuongnl/data/bdd100k_fisheye_te/da/val/"
det_new = "/home/vinai/Workspace/chuongnl/data/bdd100k_fisheye_te/det/val/"
img_dir = "/home/vinai/Workspace/chuongnl/data/bdd10k/bdd100k/images/100k/val/"
lane_dir = "/home/vinai/Workspace/chuongnl/data/bdd10k/bdd_lane_gt/val/"
da_dir = "/home/vinai/Workspace/chuongnl/data/bdd10k/bdd_seg_gt/val/"
det_dir = "/home/vinai/Workspace/chuongnl/data/bdd10k/data2/zwt/bdd/bdd100k/labels/100k/val/"

class FESetsGenerator:

    def __init__(self, dst_shape, focal_len=350):
        self._generator = FishEyeGenerator(focal_len, dst_shape)

        self._F_RAND_FLAG = False
        self._F_RANGE = [200, 400]

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

            # Apply fisheye transformation to segmentation annotation
            result3 = self._generator.transFromGray(src_annot_image_l)
            cv2.imwrite(os.path.join(dst_annot_dir_l, prefix + annotation_name), result3)
            # for frame in data.get("frames", []):
            #     for obj in frame.get("objects", []):
            #         if "box2d" in obj:
            #             bbox = [
            #                 obj["box2d"]["x1"],
            #                 obj["box2d"]["y1"],
            #                 obj["box2d"]["x2"],
            #                 obj["box2d"]["y2"],
            #             ]
            #             transformed_bbox = self._generator.trans_bbox(bbox)
            #             obj["box2d"]["x1"] = transformed_bbox[0]
            #             obj["box2d"]["y1"] = transformed_bbox[1]
            #             obj["box2d"]["x2"] = transformed_bbox[2]
            #             obj["box2d"]["y2"] = transformed_bbox[3]

            # # Save the updated annotation
            # output_path = os.path.join(dst_det, prefix + det_name)
            # os.makedirs(os.path.dirname(output_path), exist_ok=True)
            # with open(output_path, "w") as file:
            #     json.dump(data, file, indent=4)

            # print(f"Processed: {image} ({count + 1}/{len(image_list)})")

        # print("All images and annotations processed!")

def test():
    DT = FESetsGenerator([640, 640], focal_len=350)
    DT.set_ext_param_range([10, 10, 10, 0, 0, 0])
    DT.rand_ext_params()
    # Uncomment to randomize focal length
    # DT.rand_f()
    DT.generate(img_dir, lane_dir, da_dir, det_dir, dst_dir, lane_new, da_new, det_new, prefix='')

if __name__ == "__main__":
    test()
