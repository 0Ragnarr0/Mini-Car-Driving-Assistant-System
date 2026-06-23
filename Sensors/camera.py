#Get images or frames from(video) cameras (omnidirectional,3D cameras) needed to be used in the project.
#The camera is used to capture the environment and provide datas for the Trained AI model to make decisions and control the mini car for lane adjusting and obstacle avoidance.

import numpy as np
import cv2

def process_3dCamera_data(raw_depth_map):
    """
    Provess raw camera data into optimised format for ryzen 5 2500U
    Inputs:
    -raw_depth_map:2D numpy array(dtype=uint16)represent distance in mm
    """
    #Get the dimensions of the depth image
    height, width = raw_depth_map.shape
    
    #Extract a single horizontal line across he middle of depth map
    middle_row_index = height // 2
    middle_row = raw_depth_map[middle_row_index, :]

    #Linerallly sample exactly 10 points across the middle row
    sample_indices = np.linspace(0, width - 1, num=10, dtype=int)
    virtual_lidar_mm = middle_row[sample_indices]

    #Normalise the data:Convert from mm to meters and cap at 3.0 meters max for keeping the input values of AI model within 0 to 1 range
    virtual_lidar_meters = virtual_lidar_mm / 1000.0
    virtual_lidar_cleaned = np.clip(virtual_lidar_meters / 3.0, 0.0, 1.0)
    return virtual_lidar_cleaned

def process_fisheye_data(raw_fisheye_rgb):
    """
    Process raw fisheye RGB data into optimised format for ryzen 5 2500U
    Inputs:
    -raw_fisheye_rgb:3D numpy array(dtype=uint8)represent RGB image from fisheye camera
    """
    #Convert to grayscale(drops data from 3 channels to 1)
    gray_image = cv2 .cvtColor(raw_fisheye_rgb, cv2.COLOR_BGR2GRAY)

    #crop out the top half of the image (doesnt need to see the sky)
    f_height, f_width = gray_image.shape
    cropped_image = gray_image[f_height//2: f_height, 0: f_width]

    #resize to 64x64 pixels (reduces data size while keeping enough detail for lane detection)
    resized_image = cv2.resize(cropped_image, (64,64), interpolation=cv2.INTER_AREA)

    #Normalise pixel values from 0-255 to 0-1 range
    normalized_image = resized_image / 255.0

    #flatten the 2d image into 1d array of 4096 values(64x64)
    flattend_visuals = normalized_image.flatten()

    return flattend_visuals

def ai_model_input_processing(raw_depth_map, raw_fisheye_rgb):
    """
    Combines the processed depth and fisheye data into a single input vector for the AI model
    """
    virtual_lidar_input = process_3dCamera_data(raw_depth_map)
    fisheye_visual_input = process_fisheye_data(raw_fisheye_rgb)

    #Combine the two inputs into a single vector (10 lidar values + 4096 visual values = 4106 total)
    ai_input_vector = np.concatenate([virtual_lidar_input, fisheye_visual_input])
    return ai_input_vector

#Simulation Test:Test the pipeline with dummy data

if __name__ == "__main__":
    #simulate raw depth image from a camera(VGA resolution 640x480)
    #Filled with random objects distance between 0.5m to 3m(500mm to 3000mm)
    dummy_depth = np.random.randint(500, 3000, size=(480, 640),dtype=np.uint16)

    #simulate raw fisheye RGB image(480x640x3)
    dummy_fisheye = np.random.randint(0,256, size=(480,640,3), dtype=np.uint8)

    #Run the processing engine
    final_state = ai_model_input_processing(dummy_depth,dummy_fisheye)

    print(f"Final AI model input vector shape: {final_state.shape}")  
    print(f"Total numeric inputs to AI model: {final_state.size}")
    print(f"Frist 10 values (virtual lidar): {final_state[:10]}")
    print("Data processing complete. AI model input vector ready for inference.")