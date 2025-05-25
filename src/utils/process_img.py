import os
import base64
import json
import uuid
from pathlib import Path

def image_to_base64(image_path: str):
  with open(image_path, "rb") as f:
    return base64.b64encode(f.read()).decode("utf-8")

def save_uploaded_image(pil_img) -> Path:
  """Save PIL image to ./static and return its path."""
  Path("static").mkdir(exist_ok=True)
  filename = f"upload_{uuid.uuid4().hex[:8]}.png"
  path = Path("static") / filename
  pil_img.save(path)
  return path
