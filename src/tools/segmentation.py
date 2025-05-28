import numpy as np
from autogen_core.tools import FunctionTool
from transformers import SamModel, SamProcessor
from PIL import Image
import matplotlib.pyplot as plt
segmentation_model_id = "facebook/sam-vit-base"
sam_processor = SamProcessor.from_pretrained(segmentation_model_id)
sam_model = SamModel.from_pretrained(segmentation_model_id)

def run_segmentation(image_path: str):
  img = Image.open(image_path).convert("RGB")
  inputs = sam_processor(images=img, return_tensors="pt")
  outputs = sam_model(**inputs)

  logits = outputs.pred_masks[0]                     # (num_masks, H, W)
  probs = logits.sigmoid().cpu().detach().numpy()
  binary = (probs > 0.5).astype(np.uint8)

  # take the very first mask
  single_mask = binary[0]                           # (H, W)
  bin_mask = (single_mask * 255).astype(np.uint8)  # 0 or 255

  return {"masks": bin_mask}  # still a numpy array


segmentation_tool = FunctionTool(
    run_segmentation,
    description="Generate pixel-accurate binary mask from an image using SAM.")


if __name__ == "__main__":
  image_path = r"D:\Asus\AIO\Project\ai-agent-cv-tasks\static\animals.jpg"
  result = run_segmentation(image_path)
