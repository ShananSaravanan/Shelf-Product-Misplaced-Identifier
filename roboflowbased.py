from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw
import numpy as np
import cv2

# Initialize the inference client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="yKu5It01y8vjzfiOTnXA"
)


def detect_products(image_path, model_id, confidence_threshold=0.05):
    result = CLIENT.infer(image_path, model_id=model_id)
    predictions = result['predictions']
    filtered_predictions = [pred for pred in predictions if pred['confidence'] >= confidence_threshold]
    return filtered_predictions

def normalize_coordinates(predictions, image_width, image_height):
    for prediction in predictions:
        prediction['x'] /= image_width
        prediction['y'] /= image_height
        prediction['width'] /= image_width
        prediction['height'] /= image_height
    return predictions

def calculate_color_difference(image1, image2, box1, box2):
    x0_1, y0_1, x1_1, y1_1 = box1
    x0_2, y0_2, x1_2, y1_2 = box2
    
    crop1 = image1.crop((x0_1, y0_1, x1_1, y1_1))
    crop2 = image2.crop((x0_2, y0_2, x1_2, y1_2))
    
    crop1 = cv2.cvtColor(np.array(crop1), cv2.COLOR_RGB2BGR)
    crop2 = cv2.cvtColor(np.array(crop2), cv2.COLOR_RGB2BGR)
    
    hist1 = cv2.calcHist([crop1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([crop2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()
    
    color_diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    return color_diff

def draw_bounding_boxes(image_path, predictions, save_path, differences=None):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    image_width, image_height = image.size

    for prediction in predictions:
        x_center = prediction['x'] * image_width
        y_center = prediction['y'] * image_height
        box_width = prediction['width'] * image_width
        box_height = prediction['height'] * image_height
        class_label = prediction['class']
        confidence = prediction['confidence']
        
        x0 = int(x_center - box_width / 2)
        y0 = int(y_center - box_height / 2)
        x1 = int(x_center + box_width / 2)
        y1 = int(y_center + box_height / 2)
        
        outline_color = "green"
        if differences is not None:
            for diff in differences:
                if diff['class'] == class_label and (diff['latest_position'][0] == prediction['x'] and diff['latest_position'][1] == prediction['y']):
                    if diff['status'] == 'slightly misplaced':
                        outline_color = "yellow"
                    elif diff['status'] == 'highly misplaced':
                        outline_color = "red"
                    elif diff['status'] == 'wrong product':
                        outline_color = "blue"
                    elif diff['color_status'] == 'slight color change':
                        outline_color = "cyan"
                    elif diff['color_status'] == 'medium color change':
                        outline_color = "orange"
                    elif diff['color_status'] == 'high color change':
                        outline_color = "purple"
        
        draw.rectangle([x0, y0, x1, y1], outline=outline_color)
        draw.text((x0, y0), f"{class_label} ({confidence:.2f})", fill=outline_color)
    
    image.save(save_path)
    image.show()

def compare_products(original_image, latest_image, original_predictions, latest_predictions, slight_threshold=0.2, high_threshold=0.5, slight_color_threshold=0.4, medium_color_threshold=0.3):
    differences = []
    
    original_dict = {pred['class']: pred for pred in original_predictions}
    latest_dict = {pred['class']: pred for pred in latest_predictions}
    
    matched_indices = set()
    
    for original in original_predictions:
        matched = False
        for i, latest in enumerate(latest_predictions):
            if i in matched_indices:
                continue
            original_x, original_y = original['x'], original['y']
            latest_x, latest_y = latest['x'], latest['y']
            distance = np.sqrt((original_x - latest_x) ** 2 + (original_y - latest_y) ** 2)
            
            box1 = (
                int((original['x'] - original['width'] / 2) * original_image.width),
                int((original['y'] - original['height'] / 2) * original_image.height),
                int((original['x'] + original['width'] / 2) * original_image.width),
                int((original['y'] + original['height'] / 2) * original_image.height),
            )
            box2 = (
                int((latest['x'] - latest['width'] / 2) * latest_image.width),
                int((latest['y'] - latest['height'] / 2) * latest_image.height),
                int((latest['x'] + latest['width'] / 2) * latest_image.width),
                int((latest['y'] + latest['height'] / 2) * latest_image.height),
            )
            
            color_diff = calculate_color_difference(original_image, latest_image, box1, box2)
            
            if original['class'] == latest['class'] and distance <= slight_threshold:
                status = 'correct'
                matched_indices.add(i)
                matched = True
                break
            elif original['class'] == latest['class'] and distance <= high_threshold:
                status = 'slightly misplaced'
                matched_indices.add(i)
                matched = True
                break
            elif original['class'] != latest['class'] and distance <= slight_threshold:
                status = 'wrong product'
                matched_indices.add(i)
                matched = True
                break
        
        if not matched:
            status = 'highly misplaced'
            latest_x, latest_y = None, None
        
        if color_diff <= slight_color_threshold:
            color_status = 'slight color change'
        elif color_diff <= medium_color_threshold:
            color_status = 'medium color change'
        else:
            color_status = 'high color change'
        
        differences.append({
            'class': original['class'],
            'original_position': (original['x'], original['y']),
            'latest_position': (latest_x, latest_y),
            'distance': distance,
            'color_difference': color_diff,
            'color_status': color_status,
            'status': status
        })
    
    return differences

# Paths to the original and latest images
original_image_path = "images/master.jpg"
latest_image_path = "images/shelf3.jpg"

# Load images to get dimensions
original_image = Image.open(original_image_path)
latest_image = Image.open(latest_image_path)
original_width, original_height = original_image.size
latest_width, latest_height = latest_image.size

# Detect products in both images with a lower confidence threshold for higher sensitivity
confidence_threshold = 0.05  # Adjust the threshold as needed
original_predictions = detect_products(original_image_path, "shelf-models-landscape-test-1/1", confidence_threshold=confidence_threshold)
latest_predictions = detect_products(latest_image_path, "shelf-models-landscape-test-1/1", confidence_threshold=confidence_threshold)

# Normalize coordinates
original_predictions = normalize_coordinates(original_predictions, original_width, original_height)
latest_predictions = normalize_coordinates(latest_predictions, latest_width, latest_height)

# Draw bounding boxes for visualization
draw_bounding_boxes(original_image_path, original_predictions, "original_with_boxes.jpg")
draw_bounding_boxes(latest_image_path, latest_predictions, "latest_with_boxes.jpg")

# Compare products and identify differences
differences = compare_products(original_image, latest_image, original_predictions, latest_predictions, slight_threshold=0.01, high_threshold=0.05)
print("Differences:", differences)

# Draw bounding boxes highlighting differences
draw_bounding_boxes(latest_image_path, latest_predictions, "latest_with_differences.jpg", differences=differences)
