#!/usr/bin/env python3
"""
Generate beautiful segmentation visualizations like the reference image
Shows satellite imagery with colored masks overlaid
"""

import json
import cv2
import numpy as np
import os
from pathlib import Path
from pycocotools import mask as maskUtils
import sys

def create_beautiful_visualization(json_file, images_dir, output_dir, num_samples=5):
    """
    Create beautiful segmentation visualizations showing masks overlaid on satellite imagery
    Matches the reference style with colored masks and clear boundaries
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load COCO JSON
    with open(json_file, 'r') as f:
        coco_data = json.load(f)
    
    images = coco_data['images']
    annotations = coco_data['annotations']
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    print(f"Total images in dataset: {len(images)}")
    print(f"Creating {min(num_samples, len(images))} beautiful visualizations...\n")
    
    # Color palette - distinct colors for each class
    class_colors = {
        'dry_farming-building-paddy-other': (50, 200, 100),      # Teal
        'dry_farming': (0, 200, 100),                             # Green
        'mix_Wood_Land': (100, 100, 200),                         # Purple
        'other_crop': (200, 150, 50),                             # Orange/Yellow
        'paddy': (200, 50, 150),                                  # Magenta
    }
    
    # Find images with good mask density
    img_mask_count = {}
    for ann in annotations:
        img_id = ann['image_id']
        img_mask_count[img_id] = img_mask_count.get(img_id, 0) + 1
    
    # Sort by mask count (most masks first)
    sorted_imgs = sorted(img_mask_count.items(), key=lambda x: x[1], reverse=True)
    
    for idx, (img_id, mask_count) in enumerate(sorted_imgs[:num_samples]):
        img_info = next((img for img in images if img['id'] == img_id), None)
        if not img_info:
            continue
        
        img_filename = img_info['file_name']
        print(f"[{idx+1}/{min(num_samples, len(sorted_imgs))}] {img_filename} ({mask_count} masks)")
        
        # Find image file
        img_path = None
        for root, dirs, files in os.walk(images_dir):
            if img_filename in files:
                img_path = os.path.join(root, img_filename)
                break
        
        if not img_path or not os.path.exists(img_path):
            print(f"  ✗ Image not found")
            continue
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print(f"  ✗ Failed to load image")
            continue
        
        h, w = image.shape[:2]
        
        # Create output image (start with original)
        output_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get annotations for this image
        img_annotations = [a for a in annotations if a['image_id'] == img_id]
        
        class_counts = {}
        
        # Draw each mask
        for ann in img_annotations:
            cat_id = ann['category_id']
            cat_name = categories[cat_id]
            class_counts[cat_name] = class_counts.get(cat_name, 0) + 1
            
            color = class_colors.get(cat_name, (128, 128, 128))
            
            # Decode mask
            if 'segmentation' in ann:
                seg = ann['segmentation']
                
                if isinstance(seg, dict) and 'counts' in seg:
                    # RLE format
                    mask = maskUtils.decode(seg)
                else:
                    # Polygon format
                    mask = np.zeros((h, w), dtype=np.uint8)
                    if isinstance(seg, list) and seg:
                        for polygon in seg:
                            pts = np.array(polygon, dtype=np.int32).reshape(-1, 2)
                            cv2.fillPoly(mask, [pts], 1)
                
                # Apply mask to output image
                mask_idx = mask > 0
                
                # Blend: 40% mask color + 60% original
                output_image[mask_idx] = (
                    output_image[mask_idx].astype(float) * 0.6 +
                    np.array(color, dtype=np.uint8) * 0.4
                ).astype(np.uint8)
                
                # Add mask boundary (darker lines for visibility)
                contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                if contours:
                    boundary_color = tuple([int(c * 0.7) for c in color])
                    cv2.drawContours(output_image, contours, -1, boundary_color, 2)
        
        # Convert back to BGR for saving
        output_image_bgr = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        
        # Save visualization
        output_filename = f"segmentation_{idx+1:02d}_{mask_count}masks.png"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, output_image_bgr)
        
        print(f"  ✓ Saved: {output_filename}")
        print(f"    Classes: {class_counts}")
        print()
    
    print("="*70)
    print(f"✓ Generated {min(num_samples, len(sorted_imgs))} beautiful visualizations!")
    print(f"✓ Location: {output_dir}/")
    print("="*70)

if __name__ == "__main__":
    json_file = "./data/annotations/train.json"
    images_dir = "./data/raw/roboflow_export"
    output_dir = "./beautiful_segmentation_outputs"
    
    print("="*70)
    print("🎨 AGRIVISION - BEAUTIFUL SEGMENTATION VISUALIZATIONS")
    print("="*70)
    print()
    
    # Check files
    if not os.path.exists(json_file):
        print(f"ERROR: {json_file} not found")
        exit(1)
    
    if not os.path.exists(images_dir):
        print(f"ERROR: {images_dir} not found")
        exit(1)
    
    # Create visualizations
    create_beautiful_visualization(json_file, images_dir, output_dir, num_samples=10)
    
    print("\n🎨 OUTPUT STYLE:")
    print("   • Satellite imagery as background")
    print("   • Colored masks overlaid (40% mask, 60% original)")
    print("   • Dark boundary lines for clarity")
    print("   • Images with most masks prioritized")
    print("\n✨ Ready to view in: " + output_dir)
