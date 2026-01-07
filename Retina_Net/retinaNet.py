# retinanet_complete.py (Pure PyTorch - No Detectron2)
import torch
import torchvision
from torchvision.models.detection import retinanet_resnet50_fpn_v2
from torchvision.models.detection.retinanet import RetinaNetHead, RetinaNetClassificationHead
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import yaml
import numpy as np

print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")

# Dataset class
class WasteDataset(Dataset):
    def __init__(self, img_dir, label_dir, transforms=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.imgs = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + '.txt')
        
        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size
        
        boxes = []
        labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, x_center, y_center, width, height = map(float, parts)
                        
                        x_min = (x_center - width/2) * img_w
                        y_min = (y_center - height/2) * img_h
                        x_max = (x_center + width/2) * img_w
                        y_max = (y_center + height/2) * img_h
                        
                        if x_max > x_min and y_max > y_min:
                            boxes.append([x_min, y_min, x_max, y_max])
                            labels.append(int(class_id) + 1)
        
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.tensor([0.0])
        target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)
        
        if self.transforms:
            img = self.transforms(img)
        
        return img, target

# Load config
with open('Dataset/data.yaml', 'r') as f:
    data_config = yaml.safe_load(f)
    num_classes = data_config['nc']
    class_names = data_config['names']

print(f"\nNumber of classes: {num_classes}")

# Create RetinaNet model
def get_retinanet_model(num_classes):
    # Load pretrained RetinaNet with ResNet50 FPN backbone
    model = retinanet_resnet50_fpn_v2(pretrained=True)
    
    # Get number of anchors per location
    num_anchors = model.head.classification_head.num_anchors
    
    # Replace classification head for our number of classes
    model.head.classification_head.num_classes = num_classes + 1  # +1 for background
    
    # Reinitialize the classification head
    in_channels = model.backbone.out_channels
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes + 1,
        norm_layer=None
    )
    
    return model

# mAP calculation functions
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

@torch.no_grad()
def evaluate_map(model, data_loader, device, num_classes, verbose=True):
    model.eval()
    all_predictions = []
    all_targets = []
    
    if verbose:
        print("  Running inference...")
    
    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        outputs = model(images)
        
        for output, target in zip(outputs, targets):
            keep = output['scores'] > 0.5
            all_predictions.append({
                'boxes': output['boxes'][keep].cpu().numpy(),
                'scores': output['scores'][keep].cpu().numpy(),
                'labels': output['labels'][keep].cpu().numpy()
            })
            all_targets.append({
                'boxes': target['boxes'].numpy(),
                'labels': target['labels'].numpy()
            })
    
    # Calculate mAP@0.5
    aps = []
    class_stats = []
    
    for class_id in range(1, num_classes + 1):
        class_preds = []
        class_gts = []
        
        for pred, gt in zip(all_predictions, all_targets):
            if len(pred['labels']) > 0:
                class_mask = pred['labels'] == class_id
                for box, score in zip(pred['boxes'][class_mask], pred['scores'][class_mask]):
                    class_preds.append({'box': box, 'score': score})
            
            if len(gt['labels']) > 0:
                gt_mask = gt['labels'] == class_id
                for box in gt['boxes'][gt_mask]:
                    class_gts.append({'box': box, 'matched': False})
        
        if len(class_gts) == 0:
            continue
        
        class_preds = sorted(class_preds, key=lambda x: x['score'], reverse=True)
        tp = fp = 0
        
        for pred in class_preds:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(class_gts):
                if gt['matched']:
                    continue
                iou = calculate_iou(pred['box'], gt['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= 0.5:
                tp += 1
                class_gts[best_gt_idx]['matched'] = True
            else:
                fp += 1
        
        if tp + fp > 0:
            precision = tp / (tp + fp)
            recall = tp / len(class_gts)
            ap = precision * recall if recall > 0 else 0
            aps.append(ap)
            
            if verbose and ap > 0.1:
                class_name = class_names[class_id-1][:20]
                class_stats.append(f"    {class_name:20s}: AP={ap:.3f} (P={precision:.2f}, R={recall:.2f})")
    
    if verbose and class_stats:
        print("\n  Per-Class Performance (AP > 0.1):")
        for stat in class_stats[:15]:  # Show top 15
            print(stat)
        if len(class_stats) > 15:
            print(f"    ... and {len(class_stats)-15} more classes")
    
    map50 = sum(aps) / len(aps) if len(aps) > 0 else 0.0
    return map50

# Training function
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for i, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Check for NaN
        if torch.isnan(losses):
            print(f"  Warning: NaN loss at batch {i+1}, skipping...")
            continue
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        
        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += losses.item()
        num_batches += 1
        
        if (i + 1) % 50 == 0:
            print(f"  Batch {i+1}/{len(data_loader)}, Loss: {losses.item():.4f}")
    
    return total_loss / num_batches if num_batches > 0 else 0

# Main training
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = T.Compose([T.ToTensor()])
    
    train_dataset = WasteDataset('Dataset/train/images', 'Dataset/train/labels', transform)
    val_dataset = WasteDataset('Dataset/valid/images', 'Dataset/valid/labels', transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=4,  # Smaller batch for RetinaNet
        shuffle=True, 
        num_workers=4,
        collate_fn=lambda x: tuple(zip(*x))
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    print(f"\nTrain: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    print("\nCreating RetinaNet model...")
    model = get_retinanet_model(num_classes).to(device)
    print("✓ Model created successfully!")
    
    # Optimizer with lower learning rate (RetinaNet is sensitive)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, 
        lr=0.002,  # Lower LR than Faster R-CNN
        momentum=0.9, 
        weight_decay=0.0005
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    os.makedirs('retinanet_models', exist_ok=True)
    
    num_epochs = 100
    best_map50 = 0.0
    patience = 20
    patience_counter = 0
    
    print("\n" + "="*60)
    print("Training RetinaNet")
    print("="*60 + "\n")
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch + 1)
        lr_scheduler.step()
        
        print(f"  Avg Loss: {train_loss:.4f}")
        
        # Validate every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"\n  Validating...")
            map50 = evaluate_map(model, val_loader, device, num_classes, verbose=True)
            print(f"\n  mAP@0.5: {map50:.4f}\n")
            
            if map50 > best_map50:
                best_map50 = map50
                patience_counter = 0
                try:
                    if os.path.exists('retinanet_models/best.pth'):
                        os.remove('retinanet_models/best.pth')
                    torch.save(model.state_dict(), 'retinanet_models/best.pth')
                    print(f"  ✓ Best model saved: {map50:.4f}")
                except Exception as e:
                    print(f"  ⚠️  Save error: {e}")
            else:
                patience_counter += 1
            
            if patience_counter >= patience // 10:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        print()
    
    # Final validation
    print("\n" + "="*60)
    print("Final Evaluation")
    print("="*60)
    final_map50 = evaluate_map(model, val_loader, device, num_classes, verbose=True)
    
    print("\n" + "="*60)
    print("RetinaNet Training Complete")
    print("="*60)
    print(f"\nBest mAP@0.5:  {best_map50:.4f}")
    print(f"Final mAP@0.5: {final_map50:.4f}")
    
    print("\n" + "="*60)
    print("Model Comparison")
    print("="*60)
    print(f"YOLOv8m:        mAP@0.5 = 0.4720")
    print(f"Faster R-CNN:   mAP@0.5 = 0.2712")
    print(f"RetinaNet:      mAP@0.5 = {final_map50:.4f}")
    
    improvement_vs_yolo = ((final_map50 - 0.47) / 0.47) * 100
    print(f"\nvs YOLOv8m:     {improvement_vs_yolo:+.1f}%")
    
    print(f"\nModel saved in: retinanet_models/")

if __name__ == '__main__':
    main()
