from ultralytics import YOLO
import torch

# Verify GPU
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

model = YOLO('yolov8m.pt')  # Start with nano for faster initial testing

# Train the model
results = model.train(
    data='Dataset/data.yaml',  # Update this path
    epochs=100,                   # Number of training epochs
    imgsz=640,                    # Image size
    batch=16,                     # Batch size (adjust based on GPU memory)
    device=0,                     # GPU device (0 for first GPU)
    patience=20,                  # Early stopping patience
    save=True,                    # Save checkpoints
    project='scrapuncle_models',  # Project folder name
    name='waste_detector_v1',     # Experiment name
    exist_ok=True,                # Overwrite existing
    pretrained=True,              # Use pretrained weights
    optimizer='auto',             # Optimizer (SGD/Adam/auto)
    verbose=True,                 # Verbose output
    seed=42,                      # Random seed for reproducibility
    deterministic=True,           # Deterministic training
    plots=True,                   # Generate plots
    val=True,                     # Validate during training
)

# Print training results
print("\n=== Training Complete ===")
print(f"Best mAP@0.5: {results.results_dict['metrics/mAP50(B)']}")
print(f"Model saved at: {results.save_dir}")
