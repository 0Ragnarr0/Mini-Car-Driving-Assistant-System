#Get images or frames from(video) cameras (omnidirectional,3D cameras) needed to be used in the project.
#The camera is used to capture the environment and provide datas for the Trained AI model to make decisions and control the mini car for lane adjusting and obstacle avoidance.

import numpy as np
import cv2

def process_3dCamera_data(raw_depth_map):
    """
    Process raw depth map into a 64-point virtual lidar array.
    """
    height, width = raw_depth_map.shape
    
    # Extract the middle row
    middle_row_index = height // 2
    middle_row = raw_depth_map[middle_row_index, :]

    # Linearly sample exactly 64 points for the new Fusion Network
    sample_indices = np.linspace(0, width - 1, num=64, dtype=int)
    virtual_lidar_mm = middle_row[sample_indices]

    # Convert mm to meters and clip to 30.0m max (as per Multimodal specs)
    virtual_lidar_meters = virtual_lidar_mm / 1000.0
    virtual_lidar_cleaned = np.clip(virtual_lidar_meters / 30.0, 0.0, 1.0)

    return virtual_lidar_cleaned.astype(np.float32)

def process_fisheye_data(raw_fisheye_rgb):
    """
    Process fisheye camera to (64, 64, 3) RGB normalized [0,1].
    """
    # Resize to 64x64
    resized_image = cv2.resize(raw_fisheye_rgb, (64, 64))
    
    # Normalize pixel values to 0.0 - 1.0
    normalized_image = resized_image / 255.0
    
    # Return the 3D tensor shape (Do NOT flatten!)
    return normalized_image.astype(np.float32)