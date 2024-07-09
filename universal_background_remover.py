import logging
import cv2
import numpy as np
from PIL import Image
import os
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    AutoModelForImageSegmentation,
)
import torch
from torchvision.ops import nms
from torch.nn import functional as F
from torchvision.transforms.functional import normalize
from typing import List, Tuple, Optional


def setup_logging():
    """
    Setup logging configuration for the script.
    """
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("processing.log"), logging.StreamHandler()],
    )


setup_logging()


class BackgroundRemover:
    def __init__(self, obj_detector_name: str, seg_model_name: str, cache_dir="cache"):
        """
        Initialize the BackgroundRemover class with object detection and segmentation models.
        """
        self.cache = {}
        self.cache_dir = cache_dir

        # Initialize the object detection model
        self.processor = AutoImageProcessor.from_pretrained(
            obj_detector_name, cache_dir=f"{cache_dir}/processor"
        )
        self.detector_model = AutoModelForObjectDetection.from_pretrained(
            obj_detector_name, cache_dir=f"{cache_dir}/detector_model"
        )

        # Initialize the segmentation model
        self.seg_model = AutoModelForImageSegmentation.from_pretrained(
            seg_model_name, trust_remote_code=True
        )
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.seg_model.to(self.device)

        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def compute_image_hash(self, image_path: str) -> str:
        """
        Compute MD5 hash of an image file.
        """
        try:
            with open(image_path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logging.error(f"Error computing hash for {image_path}: {e}")
            return ""

    def detect_objects_with_yolo(self, image_path: str) -> Optional[List[dict]]:
        """
        Detect objects in an image using YOLO object detection model.
        """
        try:
            logging.debug(f"Opening image for processing: {image_path}")
            image = Image.open(image_path).convert("RGB")
            image_np = np.array(image)
            inputs = self.processor(images=image_np, return_tensors="pt")
            outputs = self.detector_model(**inputs)

            thresholds = [0.4, 0.5, 0.6]
            all_detected_objects = []

            # Process object detection results at different thresholds
            for threshold in thresholds:
                results = self.processor.post_process_object_detection(
                    outputs,
                    target_sizes=torch.tensor([image.size[::-1]]),
                    threshold=threshold,
                )[0]

                for idx, box in enumerate(results["boxes"]):
                    label = results["labels"][idx].item()
                    score = results["scores"][idx].item()
                    box = box.tolist()
                    all_detected_objects.append(
                        {"label": label, "box": box, "score": score}
                    )

            # Apply Non-Maximum Suppression (NMS) to filter overlapping boxes
            boxes = torch.tensor([obj["box"] for obj in all_detected_objects])
            scores = torch.tensor([obj["score"] for obj in all_detected_objects])
            indices = nms(boxes, scores, iou_threshold=0.5)

            detected_objects = [all_detected_objects[i] for i in indices]

            if detected_objects:
                return detected_objects
            else:
                logging.warning("No objects detected")
                return None
        except Exception as e:
            logging.error(f"An error occurred during YOLO detection: {e}")
            return None

    def preprocess_image(self, im: np.ndarray, model_input_size: list) -> torch.Tensor:
        """
        Preprocess the image for segmentation model input.
        """
        if len(im.shape) < 3:
            im = im[:, :, np.newaxis]
        im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
        im_tensor = F.interpolate(
            torch.unsqueeze(im_tensor, 0), size=model_input_size, mode="bilinear"
        )
        image = torch.divide(im_tensor, 255.0)
        image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
        return image

    def postprocess_image(self, result: torch.Tensor, im_size: list) -> np.ndarray:
        """
        Postprocess the segmentation model output to create a mask.
        """
        result = torch.squeeze(F.interpolate(result, size=im_size, mode="bilinear"), 0)
        ma = torch.max(result)
        mi = torch.min(result)
        result = (result - mi) / (ma - mi)
        im_array = (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
        im_array = np.squeeze(im_array)
        return im_array

    def apply_background(
        self, image: np.ndarray, mask: np.ndarray, bg_color: Tuple[int, int, int]
    ) -> Image.Image:
        """
        Apply the background color to the image using the mask.
        """
        pil_im = Image.fromarray(mask)
        no_bg_image = Image.new("RGBA", pil_im.size, (*bg_color, 255))
        orig_image = Image.open(image)
        no_bg_image.paste(orig_image, mask=pil_im)
        return no_bg_image

    def save_debug_images(self, initial_mask, final_mask, mask_visual, orig_image_path):
        """
        Save debug images for initial mask, final mask, and mask visualization.
        """
        try:
            initial_mask_path = os.path.join(
                self.cache_dir, f"initial_mask_{os.path.basename(orig_image_path)}"
            )
            final_mask_path = os.path.join(
                self.cache_dir, f"final_mask_{os.path.basename(orig_image_path)}"
            )
            mask_visual_path = os.path.join(
                self.cache_dir, f"mask_visual_{os.path.basename(orig_image_path)}"
            )

            # Convert images to RGB if they are in RGBA
            if initial_mask.mode == "RGBA":
                initial_mask = initial_mask.convert("RGB")
            if final_mask.mode == "RGBA":
                final_mask = final_mask.convert("RGB")
            if mask_visual.mode == "RGBA":
                mask_visual = mask_visual.convert("RGB")

            initial_mask.save(initial_mask_path, "JPEG")
            final_mask.save(final_mask_path, "JPEG")
            mask_visual.save(mask_visual_path, "JPEG")

            logging.debug(f"Initial mask saved to {initial_mask_path}")
            logging.debug(f"Final mask saved to {final_mask_path}")
            logging.debug(f"Mask visual saved to {mask_visual_path}")
        except Exception as e:
            logging.error(f"Error saving debug images: {e}")

    def process_image(
        self, image_path: str, bg_color: Tuple[int, int, int] = (0, 0, 0)
    ) -> Optional[str]:
        """
        Process an image to remove its background.
        """
        try:
            image_hash = self.compute_image_hash(image_path)
            if image_hash in self.cache:
                return self.cache[image_hash]

            detected_objects = self.detect_objects_with_yolo(image_path)
            if not detected_objects:
                return None

            orig_im = np.array(Image.open(image_path))
            orig_im_size = orig_im.shape[0:2]

            combined_mask = np.zeros(orig_im_size, dtype=np.uint8)

            for obj in detected_objects:
                box = obj["box"]
                x1, y1, x2, y2 = map(int, box)
                roi = orig_im[y1:y2, x1:x2]
                preprocessed_roi = self.preprocess_image(roi, [640, 640]).to(
                    self.device
                )

                result = self.seg_model(preprocessed_roi)
                if isinstance(result, tuple):
                    result = result[0]
                if isinstance(result, list):
                    result = result[0]

                mask = self.postprocess_image(result, [y2 - y1, x2 - x1])
                combined_mask[y1:y2, x1:x2] = np.maximum(
                    combined_mask[y1:y2, x1:x2], mask
                )

            # Apply morphological operations to refine the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

            no_bg_image = self.apply_background(image_path, combined_mask, bg_color)

            output_filename = f"processed_{os.path.basename(image_path)}"
            output_path = os.path.join(self.cache_dir, output_filename)

            if no_bg_image.mode == "RGBA" and output_path.lower().endswith(".jpeg"):
                no_bg_image = no_bg_image.convert("RGB")
                output_path = output_path.replace(".jpeg", ".png")

            no_bg_image.save(output_path)

            # Save debug images
            initial_mask_image = Image.fromarray(combined_mask)
            final_mask_image = no_bg_image
            mask_visual_image = no_bg_image.copy()
            mask_visual_image.putalpha(Image.fromarray(combined_mask))

            self.save_debug_images(
                initial_mask_image, final_mask_image, mask_visual_image, image_path
            )

            logging.debug(f"Processed image saved to {output_path}")

            self.cache[image_hash] = output_path
            return output_path
        except Exception as e:
            logging.error(f"Unexpected error processing image {image_path}: {e}")
            return None

    def process_images_in_parallel(
        self, image_paths: List[str], bg_color: Tuple[int, int, int] = (0, 0, 0)
    ) -> List[Optional[str]]:
        """
        Process multiple images in parallel to remove their backgrounds.
        """
        output_paths = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self.process_image, path, bg_color): path
                for path in image_paths
            }
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        output_paths.append(result)
                except Exception as e:
                    logging.error(f"Error processing image in parallel: {e}")
        return output_paths


if __name__ == "__main__":
    # Default model names for object detection and segmentation
    obj_detector_name = "hustvl/yolos-base"
    seg_model_name = "briaai/RMBG-1.4"
    remover = BackgroundRemover(obj_detector_name, seg_model_name)

    # List of image paths to process
    image_paths = ["Original_pic.jpeg"]
    remover.process_images_in_parallel(image_paths)
