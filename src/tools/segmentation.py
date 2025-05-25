from autogen_core.tools import FunctionTool
from transformers import SamModel, SamProcessor
from PIL import Image

segmentation_model_id = "facebook/sam-vit-base"
sam_processor = SamProcessor.from_pretrained(segmentation_model_id)
sam_model = SamModel.from_pretrained(segmentation_model_id)

def run_segmentation(image_path: str):
  """SAM: return binary masks as nested lists"""
  img = Image.open(image_path).convert("RGB")
  inputs = sam_processor(images=img, return_tensors="pt")
  outputs = sam_model(**inputs)

  masks = outputs.pred_masks.squeeze(0).cpu().detach().numpy().tolist()
  return {"masks": masks}


segmentation_tool = FunctionTool(
    run_segmentation,
    description="Generate pixel-accurate binary masks from an image using SAM.")
