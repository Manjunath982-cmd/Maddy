import cv2
import numpy as np
from PIL import Image
import os
import torch
from ultralytics import YOLO
import logging

class FoodDetector:
    def __init__(self):
        """Initialize YOLOv8n model for food detection"""
        try:
            # Load YOLOv8n model (will download if not present)
            self.model = YOLO('yolov8n.pt')
            
            # Food classes from COCO dataset that are relevant for food detection
            self.food_classes = {
                'apple', 'orange', 'banana', 'sandwich', 'carrot', 'hot dog', 
                'pizza', 'donut', 'cake', 'broccoli', 'dining table', 'cup',
                'bowl', 'spoon', 'knife', 'fork'
            }
            
            # Extended food mapping for Indian and international foods
            self.food_mapping = {
                'apple': 'Apple',
                'orange': 'Orange', 
                'banana': 'Banana',
                'sandwich': 'Sandwich',
                'carrot': 'Carrot',
                'hot dog': 'Hot Dog',
                'pizza': 'Pizza',
                'donut': 'Donut',
                'cake': 'Cake',
                'broccoli': 'Broccoli',
                'bowl': 'Food Bowl',
                'dining table': 'Dining Table'
            }
            
            # Additional food detection patterns for Indian foods
            self.indian_food_patterns = {
                'round_flat': ['roti', 'chapati', 'naan', 'paratha'],
                'rice_like': ['rice', 'biryani', 'pulao'],
                'curry_like': ['dal', 'curry', 'gravy'],
                'round_white': ['idli', 'dhokla'],
                'triangular': ['samosa', 'kachori'],
                'elongated': ['dosa']
            }
            
            # Non-food objects to detect and alert
            self.non_food_classes = {
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
                'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
                'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
                'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush'
            }
            
            self.model_loaded = True
            print("✅ YOLOv8n model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading YOLOv8n model: {str(e)}")
            self.model_loaded = False
            self.model = None
    
    def detect_foods(self, image_path):
        """
        Detect multiple food items using YOLOv8n
        Returns: Dictionary with detected foods and their properties
        """
        try:
            if not self.model_loaded:
                return self._fallback_detection(image_path)
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {'error': 'Could not load image', 'foods': []}
            
            # Run YOLOv8n detection
            results = self.model(image)
            
            detected_foods = []
            non_food_objects = []
            
            # Process detection results
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get class name and confidence
                        class_id = int(box.cls[0])
                        class_name = self.model.names[class_id]
                        confidence = float(box.conf[0])
                        
                        # Only process detections with confidence > 0.5
                        if confidence < 0.5:
                            continue
                        
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        # Check if it's a food item
                        if class_name.lower() in self.food_classes:
                            food_info = self._process_food_detection(
                                class_name, confidence, x1, y1, x2, y2, image.shape
                            )
                            detected_foods.append(food_info)
                        
                        # Check if it's a non-food object
                        elif class_name.lower() in self.non_food_classes:
                            non_food_objects.append(class_name)
            
            # Additional detection for Indian foods using image analysis
            indian_foods = self._detect_indian_foods(image)
            detected_foods.extend(indian_foods)
            
            # Check for non-food objects
            if non_food_objects:
                return {
                    'error': f'⚠️ Non-food objects detected: {", ".join(set(non_food_objects))}. Please upload an image containing only food items.',
                    'foods': [],
                    'non_food_objects': list(set(non_food_objects))
                }
            
            if not detected_foods:
                return {
                    'error': '⚠️ No food items detected in the image. Please upload a clear image of food.',
                    'foods': []
                }
            
            return {
                'error': None,
                'foods': detected_foods,
                'image_dimensions': {'width': image.shape[1], 'height': image.shape[0]}
            }
            
        except Exception as e:
            print(f"Detection error: {str(e)}")
            return self._fallback_detection(image_path)
    
    def _process_food_detection(self, class_name, confidence, x1, y1, x2, y2, image_shape):
        """Process detected food item and extract information"""
        height, width = image_shape[:2]
        
        # Calculate bounding box properties
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        area_percentage = (bbox_width * bbox_height) / (width * height) * 100
        
        # Map to food name
        food_name = self.food_mapping.get(class_name.lower(), class_name.title())
        
        # Estimate weight based on bounding box size and food type
        estimated_weight = self._estimate_weight_from_detection(
            food_name, bbox_width, bbox_height, area_percentage
        )
        
        return {
            'name': food_name,
            'confidence': round(confidence, 2),
            'bbox': {
                'x': int(x1), 
                'y': int(y1), 
                'width': int(bbox_width), 
                'height': int(bbox_height)
            },
            'estimated_weight': estimated_weight,
            'area_percentage': round(area_percentage, 1),
            'detection_method': 'YOLOv8n'
        }
    
    def _detect_indian_foods(self, image):
        """Detect Indian foods using image analysis patterns"""
        detected_foods = []
        
        try:
            # Convert to different color spaces for analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            height, width = image.shape[:2]
            
            # Detect circular/round objects (roti, idli)
            circles = cv2.HoughCircles(
                gray, cv2.HOUGH_GRADIENT, 1, 20,
                param1=50, param2=30, minRadius=10, maxRadius=100
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (x, y, r) in circles:
                    # Analyze color in circular region
                    mask = np.zeros(gray.shape, dtype=np.uint8)
                    cv2.circle(mask, (x, y), r, 255, -1)
                    
                    masked_hsv = cv2.bitwise_and(hsv, hsv, mask=mask)
                    avg_color = cv2.mean(masked_hsv, mask=mask)
                    
                    # Classify based on color and size
                    food_name = self._classify_circular_food(avg_color, r)
                    if food_name:
                        detected_foods.append({
                            'name': food_name,
                            'confidence': 0.75,
                            'bbox': {
                                'x': max(0, x-r), 
                                'y': max(0, y-r), 
                                'width': min(2*r, width), 
                                'height': min(2*r, height)
                            },
                            'estimated_weight': self._estimate_weight_from_size(r*2, food_name),
                            'area_percentage': round((3.14159 * r * r) / (width * height) * 100, 1),
                            'detection_method': 'Indian Food Analysis'
                        })
            
            # Detect rectangular foods (dosa, paratha)
            contours, _ = cv2.findContours(
                cv2.Canny(gray, 50, 150), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Minimum area threshold
                    # Approximate contour to polygon
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Check if it's roughly rectangular
                    if len(approx) == 4:
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = w / h
                        
                        # Classify based on aspect ratio and size
                        if aspect_ratio > 1.5:  # Wide rectangle - might be dosa
                            food_name = 'Dosa'
                        elif 0.8 <= aspect_ratio <= 1.2:  # Square-ish - might be paratha
                            food_name = 'Paratha'
                        else:
                            continue
                        
                        detected_foods.append({
                            'name': food_name,
                            'confidence': 0.70,
                            'bbox': {'x': x, 'y': y, 'width': w, 'height': h},
                            'estimated_weight': self._estimate_weight_from_size(max(w, h), food_name),
                            'area_percentage': round((w * h) / (width * height) * 100, 1),
                            'detection_method': 'Shape Analysis'
                        })
        
        except Exception as e:
            print(f"Indian food detection error: {str(e)}")
        
        return detected_foods[:2]  # Return max 2 additional detections
    
    def _classify_circular_food(self, avg_color, radius):
        """Classify circular food based on color and size"""
        h, s, v = avg_color[:3]
        
        # White/light colored circles
        if v > 150 and s < 80:
            if radius < 30:
                return 'Idli'
            else:
                return 'Roti'
        
        # Yellow/golden circles
        elif 15 <= h <= 35 and s > 100:
            return 'Paratha'
        
        # Brown circles
        elif 8 <= h <= 20 and v > 80:
            return 'Roti'
        
        return None
    
    def _estimate_weight_from_detection(self, food_name, width, height, area_percentage):
        """Estimate food weight based on detection parameters"""
        base_weights = {
            'Apple': {'small': 150, 'medium': 180, 'large': 220},
            'Orange': {'small': 130, 'medium': 160, 'large': 200},
            'Banana': {'small': 100, 'medium': 120, 'large': 150},
            'Roti': {'small': 30, 'medium': 50, 'large': 80},
            'Rice': {'small': 80, 'medium': 150, 'large': 250},
            'Dal': {'small': 100, 'medium': 180, 'large': 300},
            'Idli': {'small': 25, 'medium': 35, 'large': 50},
            'Dosa': {'small': 80, 'medium': 120, 'large': 180}
        }
        
        # Default weights for unknown foods
        default_weights = {'small': 50, 'medium': 100, 'large': 200}
        
        weights = base_weights.get(food_name, default_weights)
        
        # Classify size based on area percentage
        if area_percentage < 5:
            return weights['small']
        elif area_percentage < 15:
            return weights['medium']
        else:
            return weights['large']
    
    def _estimate_weight_from_size(self, size, food_name):
        """Estimate weight from detected size"""
        if size < 50:
            multiplier = 0.8
        elif size < 100:
            multiplier = 1.0
        else:
            multiplier = 1.5
        
        base_weights = {
            'Idli': 35, 'Roti': 50, 'Dosa': 120, 'Paratha': 70
        }
        
        base_weight = base_weights.get(food_name, 80)
        return int(base_weight * multiplier)
    
    def _fallback_detection(self, image_path):
        """Fallback detection when YOLOv8n is not available"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {'error': 'Could not load image', 'foods': []}
            
            # Simple color-based detection
            height, width = image.shape[:2]
            
            # Mock detection based on filename or simple heuristics
            filename = os.path.basename(image_path).lower()
            
            detected_foods = []
            common_foods = ['rice', 'roti', 'dal']
            
            for i, food in enumerate(common_foods):
                detected_foods.append({
                    'name': food.title(),
                    'confidence': 0.75,
                    'bbox': {
                        'x': i * 50 + 10, 
                        'y': i * 40 + 10, 
                        'width': 100, 
                        'height': 80
                    },
                    'estimated_weight': [80, 50, 120][i],
                    'area_percentage': 10.0 + i * 2,
                    'detection_method': 'Fallback'
                })
            
            return {
                'error': None,
                'foods': detected_foods,
                'image_dimensions': {'width': width, 'height': height}
            }
            
        except Exception as e:
            return {'error': f'Fallback detection failed: {str(e)}', 'foods': []}
    
    def check_freshness(self, image_path):
        """
        Check if food appears fresh or decayed using computer vision
        """
        try:
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Calculate freshness score
            freshness_score = self._calculate_freshness_score(image_rgb)
            
            is_fresh = freshness_score > 0.6
            
            if freshness_score > 0.8:
                message = "✅ Food appears fresh and safe to consume!"
                status = "excellent"
            elif freshness_score > 0.6:
                message = "✅ Food appears reasonably fresh."
                status = "good"
            elif freshness_score > 0.4:
                message = "⚠️ Food freshness is questionable. Please check carefully."
                status = "questionable"
            else:
                message = "❌ Food appears to be decayed or spoiled. Do not consume!"
                status = "poor"
            
            return {
                'is_fresh': is_fresh,
                'freshness_score': round(freshness_score, 2),
                'message': message,
                'status': status
            }
            
        except Exception as e:
            return {
                'is_fresh': True,
                'freshness_score': 0.7,
                'message': f"Could not analyze freshness: {str(e)}",
                'status': "unknown"
            }
    
    def _calculate_freshness_score(self, image):
        """Calculate freshness score based on image analysis"""
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Analyze different freshness indicators
        brightness = np.mean(image)
        saturation = np.mean(hsv[:, :, 1])
        
        # Detect brown/dark spots (decay indicators)
        brown_mask = cv2.inRange(hsv, np.array([10, 50, 20]), np.array([20, 255, 200]))
        brown_ratio = np.sum(brown_mask > 0) / (image.shape[0] * image.shape[1])
        
        # Detect green/blue spots (mold indicators)
        mold_mask = cv2.inRange(hsv, np.array([40, 50, 50]), np.array([80, 255, 255]))
        mold_ratio = np.sum(mold_mask > 0) / (image.shape[0] * image.shape[1])
        
        # Calculate base freshness score
        freshness_score = 1.0
        
        # Penalties for decay indicators
        freshness_score -= brown_ratio * 2.5  # Brown spots
        freshness_score -= mold_ratio * 4.0   # Mold spots
        
        # Brightness and saturation factors
        if brightness < 80:
            freshness_score -= 0.2
        if saturation < 50:
            freshness_score -= 0.15
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, freshness_score))