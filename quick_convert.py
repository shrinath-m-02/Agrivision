#!/usr/bin/env python3
"""
Quick Roboflow Dataset Converter
Converts already-extracted Roboflow dataset to COCO format for training
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List
import cv2
import numpy as np

def convert_roboflow_dataset(dataset_path: str, output_dir: str = "./data") -> bool:
    """
    Convert extracted Roboflow dataset to COCO format
    
    Expected structure:
    dataset_path/
    ├── train/
    │   ├── images/
    │   └── _annotations.coco.json
    ├── valid/
    │   ├── images/
    │   └── _annotations.coco.json
    └── test/
        ├── images/
        └── _annotations.coco.json
    """
    
    dataset_path = Path(dataset_path)
    output_path = Path(output_dir)
    
    if not dataset_path.exists():
        print(f"❌ Dataset path not found: {dataset_path}")
        return False
    
    print(f"📁 Dataset path: {dataset_path}")
    
    # Create output directory
    annotations_dir = output_path / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    raw_dir = output_path / "raw" / "roboflow_export"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each split
    splits = {}
    for split_name in ["train", "valid", "test"]:
        split_path = dataset_path / split_name
        if not split_path.exists():
            print(f"⚠️  {split_name} split not found, skipping")
            continue
        
        # Images can be in split_path or split_path/images
        images_dir = split_path / "images"
        if not images_dir.exists():
            images_dir = split_path
        
        # Count images
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        if not image_files:
            print(f"⚠️  No images found in {split_name}")
            continue
        
        print(f"✅ {split_name}: {len(image_files)} images")
        
        # Copy images to output
        output_images_dir = raw_dir / split_name / "images"
        output_images_dir.mkdir(parents=True, exist_ok=True)
        
        for img_file in image_files:
            shutil.copy2(img_file, output_images_dir / img_file.name)
        
        # Load and convert annotations
        anno_file = split_path / "_annotations.coco.json"
        if anno_file.exists():
            with open(anno_file, 'r') as f:
                coco_data = json.load(f)
            
            # Save converted annotations
            output_anno = annotations_dir / f"{split_name}.json"
            with open(output_anno, 'w') as f:
                json.dump(coco_data, f, indent=2)
            
            print(f"   → Annotations: {len(coco_data.get('annotations', []))} masks")
            print(f"   → Classes: {len(coco_data.get('categories', []))}")
            
            splits[split_name] = coco_data
    
    print("\n" + "="*60)
    print("✅ CONVERSION COMPLETE!")
    print("="*60)
    print(f"\n📊 Dataset Summary:")
    print(f"   Output directory: {output_path}")
    print(f"   Annotations: {list(annotations_dir.glob('*.json'))}")
    
    # Create combined dataset
    if splits:
        combined = {
            "info": {"description": "Roboflow Agricultural Dataset"},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # Add categories from first split
        if splits:
            first_split = list(splits.values())[0]
            combined["categories"] = first_split.get("categories", [])
        
        image_id_offset = 0
        anno_id_offset = 0
        
        for split_name, coco_data in splits.items():
            # Remap image IDs
            id_map = {}
            for img in coco_data.get("images", []):
                old_id = img["id"]
                img["id"] = image_id_offset + old_id
                id_map[old_id] = img["id"]
                combined["images"].append(img)
            
            # Remap annotation IDs and image references
            for anno in coco_data.get("annotations", []):
                anno["id"] = anno_id_offset + anno["id"]
                anno["image_id"] = id_map.get(anno.get("image_id"), anno.get("image_id"))
                combined["annotations"].append(anno)
            
            image_id_offset += len(coco_data.get("images", []))
            anno_id_offset += len(coco_data.get("annotations", []))
        
        output_combined = annotations_dir / "dataset.json"
        with open(output_combined, 'w') as f:
            json.dump(combined, f, indent=2)
        
        print(f"\n📦 Combined dataset: {output_combined}")
        print(f"   Total images: {len(combined['images'])}")
        print(f"   Total annotations: {len(combined['annotations'])}")
        print(f"   Classes: {len(combined['categories'])}")
    
    print("\n✨ Ready for training!")
    return True

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python quick_convert.py <dataset_path> [output_dir]")
        print("\nExample:")
        print('  python quick_convert.py "C:\\Users\\91915\\Downloads\\field_2.v3-2025-03-04-2-04pm-g.coco-segmentation (1)"')
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./data"
    
    success = convert_roboflow_dataset(dataset_path, output_dir)
    sys.exit(0 if success else 1)
