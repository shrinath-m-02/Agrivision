#!/usr/bin/env python3
"""Web Server for Agrivision Segmentation Visualizations"""

from flask import Flask, render_template_string, send_file, jsonify
import os
from pathlib import Path
import json

app = Flask(__name__)

# Configuration
VIZ_DIR = "./beautiful_segmentation_outputs"
CLASS_ANALYSIS_DIR = "./class_analysis_outputs"
ANNO_FILE = "./data/annotations/train.json"

# HTML Template
HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agrivision - Segmentation Visualizations</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: Arial, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        header { background: white; padding: 30px; border-radius: 15px; margin-bottom: 40px; box-shadow: 0 10px 40px rgba(0,0,0,0.1); }
        h1 { color: #667eea; font-size: 2.5em; margin-bottom: 10px; }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-top: 20px; }
        .stat-box { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white; text-align: center; }
        .stat-box h3 { font-size: 2em; margin-bottom: 5px; }
        .gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 30px; }
        .image-card { background: white; border-radius: 15px; overflow: hidden; box-shadow: 0 10px 30px rgba(0,0,0,0.2); transition: transform 0.3s; cursor: pointer; }
        .image-card:hover { transform: translateY(-10px); }
        .image-card img { width: 100%; height: 300px; object-fit: cover; }
        .image-info { padding: 20px; }
        .image-info h3 { color: #333; margin-bottom: 10px; }
        .mask-badge { display: inline-block; background: #667eea; color: white; padding: 5px 15px; border-radius: 20px; margin-top: 10px; font-weight: bold; }
        .modal { display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.7); }
        .modal-content { background: white; margin: 5% auto; max-width: 900px; border-radius: 15px; overflow: hidden; }
        .modal-header { padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; display: flex; justify-content: space-between; }
        .modal-body { padding: 20px; max-height: 70vh; overflow-y: auto; }
        .modal-body img { width: 100%; border-radius: 10px; margin-bottom: 20px; }
        .close { color: white; font-size: 2em; cursor: pointer; }
        .close:hover { opacity: 0.7; }
        .download-btn { display: inline-block; background: #28a745; color: white; padding: 10px 20px; border-radius: 5px; text-decoration: none; margin-top: 10px; }
        footer { background: white; padding: 30px; border-radius: 15px; text-align: center; margin-top: 40px; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🌾 Agrivision - Segmentation Visualizations</h1>
            <p>Instance Segmentation Analysis of Agricultural Fields</p>
            <div class="stats">
                <div class="stat-box"><h3>{{ total_images }}</h3><p>Total Images</p></div>
                <div class="stat-box"><h3>{{ total_masks }}</h3><p>Total Masks</p></div>
                <div class="stat-box"><h3>{{ avg_masks }}</h3><p>Avg Masks/Image</p></div>
                <div class="stat-box"><h3>5</h3><p>Crop Classes</p></div>
            </div>
        </header>
        
        <div class="gallery" id="gallery"></div>
        
        <footer>
            <h2>✨ Dataset Ready for Production</h2>
            <p>200 images with 2,170 high-quality segmentation masks in COCO format</p>
        </footer>
    </div>
    
    <div id="imageModal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <span id="modalTitle"></span>
                <span class="close" onclick="closeModal()">&times;</span>
            </div>
            <div class="modal-body" id="modalBody"></div>
        </div>
    </div>
    
    <script>
        const gallery = document.getElementById('gallery');
        const modal = document.getElementById('imageModal');
        
        fetch('/api/images')
            .then(r => r.json())
            .then(data => {
                data.forEach(img => {
                    const card = document.createElement('div');
                    card.className = 'image-card';
                    card.innerHTML = `
                        <img src="/images/${img.filename}" onclick="openModal('${img.filename}', '${img.title}')">
                        <div class="image-info">
                            <h3>${img.title}</h3>
                            <span class="mask-badge">${img.masks} Masks</span>
                        </div>
                    `;
                    gallery.appendChild(card);
                });
            });
        
        function openModal(filename, title) {
            document.getElementById('modalTitle').textContent = title;
            document.getElementById('modalBody').innerHTML = `
                <img src="/images/${filename}">
                <a href="/download/${filename}" class="download-btn">📥 Download Full Resolution</a>
            `;
            modal.style.display = 'block';
        }
        
        function closeModal() {
            modal.style.display = 'none';
        }
        
        window.onclick = function(e) {
            if (e.target == modal) modal.style.display = 'none';
        }
    </script>
</body>
</html>"""

def get_image_metadata():
    images = []
    
    # Add segmentation visualizations
    if os.path.exists(VIZ_DIR):
        for img_file in sorted(os.listdir(VIZ_DIR)):
            if img_file.endswith('.png'):
                parts = img_file.replace('.png', '').split('_')
                mask_count = parts[-1].replace('masks', '')
                images.append({
                    'filename': img_file,
                    'title': img_file.replace('.png', ''),
                    'masks': mask_count,
                    'category': 'Segmentation'
                })
    
    # Add class analysis visualizations
    if os.path.exists(CLASS_ANALYSIS_DIR):
        for img_file in sorted(os.listdir(CLASS_ANALYSIS_DIR)):
            if img_file.endswith('.png'):
                images.append({
                    'filename': img_file,
                    'title': img_file.replace('.png', '').replace('_', ' '),
                    'masks': 'Analysis',
                    'category': 'Class Analysis'
                })
    
    return images

def get_dataset_stats():
    if os.path.exists(ANNO_FILE):
        with open(ANNO_FILE, 'r') as f:
            coco_data = json.load(f)
        total_images = len(coco_data.get('images', []))
        total_masks = len(coco_data.get('annotations', []))
        avg_masks = round(total_masks / total_images) if total_images > 0 else 0
        return total_images, total_masks, avg_masks
    return 0, 0, 0

@app.route('/')
def index():
    total_images, total_masks, avg_masks = get_dataset_stats()
    return render_template_string(HTML_TEMPLATE, total_images=total_images, total_masks=total_masks, avg_masks=avg_masks)

@app.route('/api/images')
def api_images():
    return jsonify(get_image_metadata())

@app.route('/images/<filename>')
def serve_image(filename):
    try:
        # Try beautiful_segmentation_outputs first
        if os.path.exists(os.path.join(VIZ_DIR, filename)):
            return send_file(os.path.join(VIZ_DIR, filename), mimetype='image/png')
        # Then try class_analysis_outputs
        elif os.path.exists(os.path.join(CLASS_ANALYSIS_DIR, filename)):
            return send_file(os.path.join(CLASS_ANALYSIS_DIR, filename), mimetype='image/png')
    except:
        pass
    return "Not found", 404

@app.route('/download/<filename>')
def download_image(filename):
    try:
        # Try beautiful_segmentation_outputs first
        if os.path.exists(os.path.join(VIZ_DIR, filename)):
            return send_file(os.path.join(VIZ_DIR, filename), as_attachment=True, download_name=filename)
        # Then try class_analysis_outputs
        elif os.path.exists(os.path.join(CLASS_ANALYSIS_DIR, filename)):
            return send_file(os.path.join(CLASS_ANALYSIS_DIR, filename), as_attachment=True, download_name=filename)
    except:
        pass
    return "Not found", 404

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🌾 AGRIVISION - WEB VISUALIZATION SERVER")
    print("="*70)
    print("\n✓ Server starting...\n")
    print("📍 Access your visualizations at:")
    print("   🔗 http://localhost:5000")
    print("\n✨ Features:")
    print("   • Beautiful gallery view")
    print("   • Click to enlarge images")
    print("   • Download full resolution")
    print("   • Dataset statistics")
    print("\n" + "="*70)
    print("Press CTRL+C to stop the server")
    print("="*70 + "\n")
    app.run(debug=False, host='0.0.0.0', port=5000)
