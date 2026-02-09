# Fine-Grained Coral Instance Segmentation using YOLOv8

[This repository implements a high-precision instance segmentation system designed for the automated monitoring of coral reef ecosystems in Indonesia. Situated within the **Coral Triangle**, Indonesia's reefs are "marine rainforests" that support critical biodiversity but face significant threats from climate change and human activity.

## Key Technical Breakthroughs
* **Top Performance**: The **YOLOv8m-Pp** (Medium with Preprocessing) model achieved a precision of **96.7%**, a recall of **95.9%**, and a mean Average Precision ($mAP_{50}$) of **98.2%**.
* [cite_start]**Optimal Preprocessing**: Our research identified **Histogram Equalization (HE)** as the most effective technique for underwater environments, successfully balancing high accuracy with a training efficiency of **31.62 minutes**.
* [cite_start]**Architectural Edge**: By utilizing YOLOv8's anchorless detection and CSPNet/PANet innovations, the system provides superior boundary delineation for complex reef structures.

## Segmented Coral Species
The model is optimized to identify and segment six distinct coral and marine categories:
1.  **Favites**
2.  **Feather-star**
3.  **Goniastrea**
4.  **Gorgonian**
5.  **Porites**
6.  **Turbinaria**

## Comparative Analysis
Our study compared several YOLO architectures to determine the optimal balance between accuracy and resource demands:

| Model | Precision | Recall | $mAP_{50}$ | Params (M) | Training Time (Min) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLOv8m-Pp** | **0.967** | **0.954** | **0.982** | **27.2** | **31.62** |
| YOLOv7-seg | 0.952 | 0.958 | 0.973 | 37.8 | 22.20 |
| YOLOv5-seg | 0.937 | 0.956 | 0.961 | 7.6 | 14.46 |
| YOLOv8m-Wp | 0.953 | 0.840 | 0.884 | 27.2 | 19.74 |

## 🛠️ Usage & Inference
To run instance segmentation on underwater video data (e.g., `coraltest.mp4`), the following inference logic is utilized:

```python
from ultralytics import YOLO

# Load the best-performing model weights (YOLOv8m-Pp)
model = YOLO('weights/best.pt')

# Perform inference on video with a 0.25 confidence threshold
results = model.predict(
    source='output/coraltest.mp4', 
    conf=0.25, 
    save=True,
    imgsz=640
)
