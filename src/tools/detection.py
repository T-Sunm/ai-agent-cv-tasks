from ultralytics import YOLO
from autogen_core.tools import FunctionTool
from typing_extensions import Annotated
detection_model_id = "yolo11n.pt"
detection_model = YOLO(detection_model_id)


def detect_and_count_object_tool(
    image_path_or_url: Annotated[str, "Path or URL to the image"]
):
  """Detect objects within the image using YOLOv11 model"""

  results = detection_model(image_path_or_url, verbose=False)

  detections = []
  counting = {}

  # Process each result
  for result in results:
    boxes = result.boxes
    class_names = result.names

    for box in boxes:
      class_id = int(box.cls[0])
      class_name = class_names[class_id]
      confidence = float(box.conf[0])
      x1, y1, x2, y2 = map(int, box.xyxy[0])

      detections.append({
          'class': class_name,
          'confidence': confidence,
          'bbox': (x1, y1, x2, y2)
      })

      counting[class_name] = counting.get(class_name, 0) + 1

  return str({'counting': counting, 'detections': detections})


detection_tool = FunctionTool(
    detect_and_count_object_tool,
    description="Detect and count objects within the image. The return will be a dictionary, containing the counting dictionary (counting how many instance of each object class) and a list of dictionaries, containing the object names, confidence scores, and location in the image (in (x1, x2, y1, y2) format)."
)
