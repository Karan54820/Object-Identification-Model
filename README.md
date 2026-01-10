# Object-Identification-Model

## Dataset 1 
https://universe.roboflow.com/waste-object-detection/waste-detection-ctmyy/dataset/12
 
Best mAP@0.5: 0.49 (Target: 0.55-0.6)

| Model | Architecture | Backbone / Variant | Training Stability | Best Reported mAP@0.5 | Overall Ranking |
|---|---|---|---|---:|---|
| RetinaNet | One-stage (anchor-based) | TorchVision RetinaNet | Stable loss, poor generalization | 0.012 | Worst |
| Faster R-CNN | Two-stage (proposal-based) | TorchVision Faster R-CNN | Fast convergence, consistent validation gains | 0.117 | Middle |
| YOLOv8m | One-stage (anchor-free) | Ultralytics YOLOv8-Medium | Very stable, well-regularized | 0.49 | Best |
## Dataset 2
https://www.kaggle.com/datasets/spellsharp/garbage-data