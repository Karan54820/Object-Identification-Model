import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import yaml
import shutil

print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# ============================================
# STEP 1: Check and Convert Labels 
# ============================================

def check_label_format(label_dir, num_samples=10):
    """Check if labels are in detection (5 values) or segmentation (>5 values) format"""
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')][:num_samples]
    
    segmentation_count = 0
    detection_count = 0
    
    for label_file in label_files:
        with open(os.path.join(label_dir, label_file), 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 5:
                    segmentation_count += 1
                elif len(parts) == 5:
                    detection_count += 1
    
    if segmentation_count > detection_count:
        return "segmentation"
    elif detection_count > 0:
        return "detection"
    else:
        return "unknown"

def polygon_to_bbox(polygon_coords):
    """Convert polygon coordinates to bounding box"""
    x_coords = [polygon_coords[i] for i in range(0, len(polygon_coords), 2)]
    y_coords = [polygon_coords[i] for i in range(1, len(polygon_coords), 2)]
    
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)
    
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    
    return x_center, y_center, width, height

def convert_labels(input_dir, output_dir):
    """Convert segmentation labels to detection labels"""
    os.makedirs(output_dir, exist_ok=True)
    
    label_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    converted = 0
    total_boxes = 0
    
    for label_file in label_files:
        input_path = os.path.join(input_dir, label_file)
        output_path = os.path.join(output_dir, label_file)
        
        with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
            for line in f_in:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                class_id = parts[0]
                
                if len(parts) == 5:
                    # Already in detection format
                    f_out.write(line)
                else:
                    # Convert from segmentation to detection
                    polygon_coords = [float(x) for x in parts[1:]]
                    x_center, y_center, width, height = polygon_to_bbox(polygon_coords)
                    f_out.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
                
                total_boxes += 1
        
        converted += 1
        if converted % 500 == 0:
            print(f"  Converted {converted}/{len(label_files)} files...")
    
    return converted, total_boxes

# Check and convert labels
print("\n" + "="*60)
print("STEP 1: Checking Label Format")
print("="*60)

train_label_dir = 'Dataset/train/labels'
val_label_dir = 'Dataset/valid/labels'

train_format = check_label_format(train_label_dir)
val_format = check_label_format(val_label_dir)

print(f"Train labels format: {train_format}")
print(f"Validation labels format: {val_format}")

if train_format == "segmentation" or val_format == "segmentation":
    print("\n⚠️  Segmentation format detected! Converting to detection format...\n")
    
    # Backup originals
    if not os.path.exists('Dataset/train/labels_original'):
        print("Backing up original labels...")
        shutil.copytree(train_label_dir, 'Dataset/train/labels_original')
        shutil.copytree(val_label_dir, 'Dataset/valid/labels_original')
    
    # Convert
    print("Converting train labels...")
    train_converted, train_boxes = convert_labels(train_label_dir, 'Dataset/train/labels_converted')
    
    print("Converting validation labels...")
    val_converted, val_boxes = convert_labels(val_label_dir, 'Dataset/valid/labels_converted')
    
    # Replace
    shutil.rmtree(train_label_dir)
    shutil.rmtree(val_label_dir)
    shutil.move('Dataset/train/labels_converted', train_label_dir)
    shutil.move('Dataset/valid/labels_converted', val_label_dir)
    
    print(f"\n✓ Conversion complete!")
    print(f"  Train: {train_converted} files, {train_boxes} boxes")
    print(f"  Val: {val_converted} files, {val_boxes} boxes")
    print(f"  Originals backed up to labels_original/")
else:
    print("✓ Labels already in detection format!")

# ============================================
# STEP 2: Dataset and Model Setup
# ============================================

print("\n" + "="*60)
print("STEP 2: Setting Up Dataset and Model")
print("="*60)

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

# Load data.yaml
with open('Dataset/data.yaml', 'r') as f:
    data_config = yaml.safe_load(f)
    num_classes = data_config['nc']
    class_names = data_config['names']

print(f"Number of classes: {num_classes}")
print(f"Class names: {class_names[:10]}...")  # Show first 10

def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    return model

# ============================================
# STEP 3: mAP Calculation Functions
# ============================================

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
def evaluate_map(model, data_loader, device, num_classes, class_names):
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    print("  Running inference on validation set...")
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
    print("\n  Per-Class Performance:")
    
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
        
        tp = 0
        fp = 0
        
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
            
            if ap > 0.1:  # Only show classes with some performance
                print(f"    {class_names[class_id-1][:20]:20s}: AP={ap:.3f} (P={precision:.2f}, R={recall:.2f})")
    
    map50 = sum(aps) / len(aps) if len(aps) > 0 else 0.0
    return map50

# ============================================
# STEP 4: Training
# ============================================

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for i, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
        num_batches += 1
        
        if (i + 1) % 50 == 0:
            print(f"  Batch {i+1}/{len(data_loader)}, Loss: {losses.item():.4f}")
    
    return total_loss / num_batches if num_batches > 0 else 0

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform = T.Compose([T.ToTensor()])
    
    train_dataset = WasteDataset(
        img_dir='Dataset/train/images',
        label_dir='Dataset/train/labels',
        transforms=transform
    )
    
    val_dataset = WasteDataset(
        img_dir='Dataset/valid/images',
        label_dir='Dataset/valid/labels',
        transforms=transform
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=4,
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
    
    print(f"\nTrain dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    # Quick dataset check
    print("\nVerifying dataset...")
    sample_img, sample_target = train_dataset[0]
    print(f"  Sample image shape: {sample_img.shape}")
    print(f"  Sample boxes: {len(sample_target['boxes'])}")
    
    if len(sample_target['boxes']) == 0:
        print("\n❌ ERROR: Still no boxes found! Check label conversion.")
        return
    
    print("  ✓ Dataset looks good!")
    
    model = get_model(num_classes)
    model.to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    num_epochs = 100
    best_map50 = 0.0
    patience = 20
    patience_counter = 0
    
    os.makedirs('faster_rcnn_models', exist_ok=True)
    
    print("\n" + "="*60)
    print("STEP 3: Training Faster R-CNN")
    print("="*60 + "\n")
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch + 1)
        lr_scheduler.step()
        
        print(f"  Avg Loss: {train_loss:.4f}")
        
        # Validate every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"\n  Validating...")
            map50 = evaluate_map(model, val_loader, device, num_classes, class_names)
            print(f"\n  mAP@0.5: {map50:.4f}\n")
            
            if map50 > best_map50:
                best_map50 = map50
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss,
                    'map50': map50,
                }, 'faster_rcnn_models/best.pth')
                print(f"  ✓ Best model saved with mAP@0.5: {map50:.4f}")
            else:
                patience_counter += 1
        
        if patience_counter >= patience // 5:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'faster_rcnn_models/epoch_{epoch+1}.pth')
        
        print()
    
    # Final validation
    print("\n" + "="*60)
    print("STEP 4: Final Evaluation")
    print("="*60)
    final_map50 = evaluate_map(model, val_loader, device, num_classes, class_names)
    
    torch.save(model.state_dict(), 'faster_rcnn_models/final.pth')
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best mAP@0.5:  {best_map50:.4f}")
    print(f"Final mAP@0.5: {final_map50:.4f}")
    print(f"\nModel Comparison:")
    print(f"  YOLOv8m:       mAP@0.5 = 0.4720")
    print(f"  Faster R-CNN:  mAP@0.5 = {final_map50:.4f}")
    improvement = ((final_map50 - 0.47) / 0.47) * 100
    print(f"  Improvement:   {improvement:+.1f}%")
    print(f"\nModels saved in: faster_rcnn_models/")
    print("="*60)

if __name__ == '__main__':
    main()
