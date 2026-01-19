#!/usr/bin/env python3
"""
Class Comparison Visualizations
Shows examples and statistics for each crop class in the dataset
"""

import json
import cv2
import numpy as np
import os
from pathlib import Path
from pycocotools import mask as maskUtils
import matplotlib.pyplot as plt
from collections import defaultdict

def analyze_classes(json_file, images_dir, output_dir):
    """Analyze and visualize each class separately"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load COCO JSON
    with open(json_file, 'r') as f:
        coco_data = json.load(f)
    
    images = coco_data['images']
    annotations = coco_data['annotations']
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    print(f"\n{'='*70}")
    print(f"CROP CLASS ANALYSIS & COMPARISON")
    print(f"{'='*70}\n")
    
    # Count annotations per class
    class_stats = defaultdict(lambda: {'count': 0, 'images': set()})
    for ann in annotations:
        cat_id = ann['category_id']
        cat_name = categories[cat_id]
        class_stats[cat_name]['count'] += 1
        class_stats[cat_name]['images'].add(ann['image_id'])
    
    # Display statistics
    print("📊 CLASS STATISTICS:")
    print("-" * 70)
    total_masks = sum(s['count'] for s in class_stats.values())
    
    for class_name in sorted(class_stats.keys()):
        stats = class_stats[class_name]
        count = stats['count']
        num_images = len(stats['images'])
        percentage = (count / total_masks) * 100
        
        print(f"\n{class_name}")
        print(f"  • Total masks: {count} ({percentage:.1f}%)")
        print(f"  • Found in: {num_images} images")
        print(f"  • Avg per image: {count/num_images:.1f}")
    
    # Color assignment for each class
    class_colors = {
        'dry_farming-building-paddy-other': (255, 100, 100),   # Light Red
        'dry_farming': (0, 200, 100),                            # Green
        'mix_Wood_Land': (100, 100, 200),                        # Purple
        'other_crop': (200, 150, 50),                            # Orange
        'paddy': (200, 50, 150),                                 # Magenta
    }
    
    print(f"\n{'='*70}")
    print("CREATING PER-CLASS VISUALIZATIONS")
    print(f"{'='*70}\n")
    
    # For each class, find best examples and create visualization
    for class_name in sorted(class_stats.keys()):
        print(f"\nProcessing class: {class_name}")
        
        # Get all images with this class
        class_image_ids = class_stats[class_name]['images']
        
        # Find images with most instances of this class
        class_mask_counts = defaultdict(int)
        for ann in annotations:
            if ann['category_id'] == next(k for k, v in categories.items() if v == class_name):
                class_mask_counts[ann['image_id']] += 1
        
        # Get top 3 images for this class
        top_images = sorted(
            [(img_id, count) for img_id, count in class_mask_counts.items()],
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        # Create comparison figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"Class: {class_name} - Top 3 Examples", fontsize=16, fontweight='bold')
        
        for idx, (img_id, mask_count) in enumerate(top_images):
            img_info = next((img for img in images if img['id'] == img_id), None)
            if not img_info:
                continue
            
            # Find image file
            img_path = None
            for root, dirs, files in os.walk(images_dir):
                if img_info['file_name'] in files:
                    img_path = os.path.join(root, img_info['file_name'])
                    break
            
            if not img_path or not os.path.exists(img_path):
                continue
            
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            
            # Create mask overlay for this class only
            overlay = img_rgb.copy().astype(float)
            mask_vis = np.zeros((h, w, 3), dtype=np.uint8)
            
            cat_id = next(k for k, v in categories.items() if v == class_name)
            
            # Get annotations for this image and class
            img_annotations = [
                a for a in annotations 
                if a['image_id'] == img_id and a['category_id'] == cat_id
            ]
            
            color = class_colors.get(class_name, (128, 128, 128))
            
            for ann in img_annotations:
                if 'segmentation' in ann:
                    seg = ann['segmentation']
                    
                    if isinstance(seg, dict) and 'counts' in seg:
                        mask = maskUtils.decode(seg)
                    else:
                        mask = np.zeros((h, w), dtype=np.uint8)
                        if isinstance(seg, list) and seg:
                            for polygon in seg:
                                pts = np.array(polygon, dtype=np.int32).reshape(-1, 2)
                                cv2.fillPoly(mask, [pts], 1)
                    
                    mask_idx = mask > 0
                    overlay[mask_idx] = overlay[mask_idx] * 0.5 + np.array(color) * 0.5
                    mask_vis[mask_idx] = color
            
            overlay = overlay.astype(np.uint8)
            
            # Display
            axes[idx].imshow(overlay)
            axes[idx].set_title(f"{mask_count} instances of {class_name}", fontsize=12, fontweight='bold')
            axes[idx].axis('off')
        
        # Save figure
        output_file = os.path.join(output_dir, f"class_{class_name.replace('/', '_')}_examples.png")
        plt.tight_layout()
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {os.path.basename(output_file)}")
    
    # Create class comparison chart
    print(f"\nCreating class distribution chart...")
    
    class_names = sorted(class_stats.keys())
    mask_counts = [class_stats[cn]['count'] for cn in class_names]
    image_counts = [len(class_stats[cn]['images']) for cn in class_names]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Crop Class Distribution Analysis", fontsize=16, fontweight='bold')
    
    # Mask count bar chart
    colors_list = [class_colors.get(cn, (128, 128, 128)) for cn in class_names]
    colors_normalized = [(r/255, g/255, b/255) for r, g, b in colors_list]
    
    ax1.bar(range(len(class_names)), mask_counts, color=colors_normalized)
    ax1.set_xlabel("Crop Class", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Total Masks", fontsize=12, fontweight='bold')
    ax1.set_title("Total Segmentation Masks per Class", fontsize=12, fontweight='bold')
    ax1.set_xticks(range(len(class_names)))
    ax1.set_xticklabels(class_names, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(mask_counts):
        ax1.text(i, v + 20, str(v), ha='center', va='bottom', fontweight='bold')
    
    # Image count bar chart
    ax2.bar(range(len(class_names)), image_counts, color=colors_normalized)
    ax2.set_xlabel("Crop Class", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Number of Images", fontsize=12, fontweight='bold')
    ax2.set_title("Images Containing Each Class", fontsize=12, fontweight='bold')
    ax2.set_xticks(range(len(class_names)))
    ax2.set_xticklabels(class_names, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(image_counts):
        ax2.text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, "class_distribution_analysis.png")
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: class_distribution_analysis.png")
    
    # Create side-by-side class comparison
    print(f"\nCreating side-by-side class comparison...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Crop Class Comparison - One Example Per Class", fontsize=16, fontweight='bold')
    
    for class_idx, class_name in enumerate(sorted(class_stats.keys())):
        row = class_idx // 3
        col = class_idx % 3
        ax = axes[row, col]
        
        # Get one good example
        class_image_ids = list(class_stats[class_name]['images'])
        
        if not class_image_ids:
            continue
        
        img_id = class_image_ids[0]
        img_info = next((img for img in images if img['id'] == img_id), None)
        
        if not img_info:
            continue
        
        # Find image file
        img_path = None
        for root, dirs, files in os.walk(images_dir):
            if img_info['file_name'] in files:
                img_path = os.path.join(root, img_info['file_name'])
                break
        
        if not img_path or not os.path.exists(img_path):
            continue
        
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Create mask overlay for this class only
        overlay = img_rgb.copy().astype(float)
        
        cat_id = next(k for k, v in categories.items() if v == class_name)
        img_annotations = [
            a for a in annotations 
            if a['image_id'] == img_id and a['category_id'] == cat_id
        ]
        
        color = class_colors.get(class_name, (128, 128, 128))
        
        for ann in img_annotations:
            if 'segmentation' in ann:
                seg = ann['segmentation']
                
                if isinstance(seg, dict) and 'counts' in seg:
                    mask = maskUtils.decode(seg)
                else:
                    mask = np.zeros((h, w), dtype=np.uint8)
                    if isinstance(seg, list) and seg:
                        for polygon in seg:
                            pts = np.array(polygon, dtype=np.int32).reshape(-1, 2)
                            cv2.fillPoly(mask, [pts], 1)
                
                mask_idx = mask > 0
                overlay[mask_idx] = overlay[mask_idx] * 0.5 + np.array(color) * 0.5
        
        overlay = overlay.astype(np.uint8)
        
        ax.imshow(overlay)
        stats = class_stats[class_name]
        ax.set_title(f"{class_name}\n({stats['count']} masks in {len(stats['images'])} images)", 
                    fontsize=11, fontweight='bold')
        ax.axis('off')
    
    # Hide empty subplots
    if len(class_names) < 6:
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, "all_classes_comparison.png")
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: all_classes_comparison.png")
    
    print(f"\n{'='*70}")
    print(f"✓ CLASS ANALYSIS COMPLETE!")
    print(f"{'='*70}")
    print(f"\n📁 Output directory: {output_dir}")
    print(f"\nGenerated files:")
    print(f"  • class_*_examples.png - Top 3 examples per class")
    print(f"  • class_distribution_analysis.png - Bar charts with statistics")
    print(f"  • all_classes_comparison.png - Side-by-side comparison")

if __name__ == "__main__":
    json_file = "./data/annotations/train.json"
    images_dir = "./data/raw/roboflow_export"
    output_dir = "./class_analysis_outputs"
    
    if not os.path.exists(json_file):
        print(f"ERROR: {json_file} not found")
        exit(1)
    
    if not os.path.exists(images_dir):
        print(f"ERROR: {images_dir} not found")
        exit(1)
    
    analyze_classes(json_file, images_dir, output_dir)
