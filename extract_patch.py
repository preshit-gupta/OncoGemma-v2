import os
import sys
import json
import traceback
import numpy as np
from PIL import Image

# For Windows, openslide needs its DLLs loaded. We try to find it in the PATH.
if hasattr(os, 'add_dll_directory'):
    fallback_bin = r"C:\openslide-win64\openslide-win64-20231011\bin"
    if os.path.isdir(fallback_bin):
        try:
            os.add_dll_directory(fallback_bin)
        except Exception:
            pass
            
    for p in os.environ.get('PATH', '').split(os.pathsep):
        if 'openslide' in p.lower() and os.path.isdir(p):
            try:
                os.add_dll_directory(p)
                break
            except Exception:
                pass

try:
    import openslide
    import cv2
except ImportError as e:
    openslide = None
    cv2 = None

def extract_tissue_patches(svs_path: str, output_dir: str, target_size: int = 448) -> dict:
    """
    Opens a Whole Slide Image (SVS), runs tissue detection using Otsu's thresholding,
    and extracts 10 high-power field (HPF) patches from the densest regions.
    """
    if openslide is None or cv2 is None:
        return {
            "success": False,
            "error": "Missing OpenSlide or OpenCV. Ensure they are installed and binaries are configured."
        }

    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # 1. Open slide
        slide = openslide.OpenSlide(svs_path)
        
        # 2. Get magnification & downsample details
        level0_mag = float(slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER, 40))
        target_1x_downsample = level0_mag / 1.0
        
        # Find best lower level for macro tissue detection
        best_lower_level = slide.get_best_level_for_downsample(target_1x_downsample)
        thumb_level_dim = slide.level_dimensions[best_lower_level]
        
        # Memory optimization: limit thumbnail size
        MAX_THUMB_SIZE = 2048
        thumbnail_rgba = slide.read_region((0, 0), best_lower_level, thumb_level_dim)
        
        # Convert transparent to white background
        thumbnail = Image.new("RGB", thumbnail_rgba.size, (255, 255, 255))
        if len(thumbnail_rgba.split()) == 4:
            thumbnail.paste(thumbnail_rgba, mask=thumbnail_rgba.split()[3])
        else:
            thumbnail.paste(thumbnail_rgba)
        
        if max(thumb_level_dim) > MAX_THUMB_SIZE:
            scale = MAX_THUMB_SIZE / max(thumb_level_dim)
            new_size = (int(thumb_level_dim[0] * scale), int(thumb_level_dim[1] * scale))
            thumbnail = thumbnail.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy for OpenCV processing
        thumb_np = np.array(thumbnail)
        gray = cv2.cvtColor(thumb_np, cv2.COLOR_RGB2GRAY)
        
        # Fixed thresholding to segment tissue from background (glass is white/255, tissue is darker < 220)
        _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
        
        # Scale factor from threshold mask back to Level 0
        h_thresh, w_thresh = thresh.shape
        scale_w = slide.dimensions[0] / w_thresh
        scale_h = slide.dimensions[1] / h_thresh
        
        # Window sizes for scanning tissue density (representing 1x mag area)
        mask_window_w = int((target_size * level0_mag) / scale_w)
        mask_window_h = int((target_size * level0_mag) / scale_h)
        
        stride_x = max(50, mask_window_w // 2)
        stride_y = max(50, mask_window_h // 2)
        
        scored_regions = []
        for y in range(0, h_thresh - mask_window_h + 1, stride_y):
            for x in range(0, w_thresh - mask_window_w + 1, stride_x):
                window_mask = thresh[y:y+mask_window_h, x:x+mask_window_w]
                density = cv2.countNonZero(window_mask)
                if density > 0:
                    scored_regions.append({"x": x, "y": y, "score": density})
                    
        if not scored_regions:
            raise Exception("No cellular tissue detected on the slide.")
            
        # Sort regions by tissue density
        scored_regions.sort(key=lambda r: r["score"], reverse=True)
        
        # Non-Maximum Suppression (NMS) to select all distinct candidate regions (no limit)
        selected_regions = []
        for r in scored_regions:
            overlap = False
            for sel in selected_regions:
                if abs(r["x"] - sel["x"]) < mask_window_w and abs(r["y"] - sel["y"]) < mask_window_h:
                    overlap = True
                    break
            if not overlap:
                selected_regions.append(r)
                    
        # Extract exactly 20 patches with maximum cellularity and >=70% tissue coverage (threshold < 220)
        extracted_patches = []
        target_num_patches = 20
        
        # We try multiple passes with decreasing tissue thresholds to get the densest patches possible
        thresholds = [0.70, 0.50, 0.30, 0.10, 0.01, 0.0]
        
        # Avoid duplicate/highly overlapping coordinates (at least half patch size separation)
        min_distance = target_size // 2
        
        for min_tissue_pct in thresholds:
            if len(extracted_patches) >= target_num_patches:
                break
                
            for reg_idx, reg in enumerate(selected_regions):
                if len(extracted_patches) >= target_num_patches:
                    break
                    
                reg_x_lvl0 = int(reg["x"] * scale_w)
                reg_y_lvl0 = int(reg["y"] * scale_h)
                reg_size_lvl0 = int(target_size * level0_mag)
                
                # Make up to 30 attempts per region per pass to find suitable patches
                for _ in range(30):
                    if len(extracted_patches) >= target_num_patches:
                        break
                        
                    off_x = np.random.randint(0, max(1, reg_size_lvl0 - target_size))
                    off_y = np.random.randint(0, max(1, reg_size_lvl0 - target_size))
                    
                    patch_x = reg_x_lvl0 + off_x
                    patch_y = reg_y_lvl0 + off_y
                    
                    # Ensure no overlap with existing selected patches
                    duplicate = False
                    for p in extracted_patches:
                        if abs(p["x"] - patch_x) < min_distance and abs(p["y"] - patch_y) < min_distance:
                            duplicate = True
                            break
                    if duplicate:
                        continue
                        
                    rgba_patch = slide.read_region((patch_x, patch_y), 0, (target_size, target_size))
                    candidate_patch = Image.new("RGB", rgba_patch.size, (255, 255, 255))
                    if len(rgba_patch.split()) == 4:
                        candidate_patch.paste(rgba_patch, mask=rgba_patch.split()[3])
                    else:
                        candidate_patch.paste(rgba_patch)
                        
                    # Grayscale for tissue coverage check
                    patch_np = np.array(candidate_patch)
                    patch_gray = cv2.cvtColor(patch_np, cv2.COLOR_RGB2GRAY)
                    
                    # Grayscale threshold: stained tissue is < 220, glass background is >= 220
                    tissue_pixels = np.sum(patch_gray < 220)
                    tissue_pct = tissue_pixels / (target_size * target_size)
                    
                    if tissue_pct >= min_tissue_pct:
                        patch_name = f"patch_{len(extracted_patches)}.png"
                        patch_path = os.path.join(output_dir, patch_name)
                        candidate_patch.save(patch_path, "PNG")
                        
                        extracted_patches.append({
                            "path": patch_path,
                            "filename": patch_name,
                            "x": patch_x,
                            "y": patch_y,
                            "region_index": reg_idx,
                            "tissue_percentage": float(tissue_pct)
                        })
                
        # Get slide dimensions before closing
        slide_w, slide_h = slide.dimensions
        slide.close()
        
        # Save a downsampled thumbnail of the slide for UI rendering
        thumb_out_path = os.path.join(output_dir, "slide_thumbnail.png")
        thumbnail.save(thumb_out_path, "PNG")
        
        return {
            "success": True,
            "patches": extracted_patches,
            "thumbnail_path": thumb_out_path,
            "slide_width": slide_w,
            "slide_height": slide_h
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_patch.py <svs_path> <output_dir>")
        sys.exit(1)
        
    svs_file = sys.argv[1]
    out_dir = sys.argv[2]
    
    result = extract_tissue_patches(svs_file, out_dir)
    print(json.dumps(result, indent=2))
