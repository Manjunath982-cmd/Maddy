THIS SHOULD BE A LINTER ERRORimport cv2
import numpy as np
from PIL import Image
import os
import random

class FoodDetector:
    def __init__(self):
        """Initialize the food detection model"""
        # Food categories with confidence weights
        self.food_categories = {
            'apple': 0.9, 'banana': 0.85, 'orange': 0.88, 'rice': 0.82, 'roti': 0.87,
            'dal': 0.83, 'chicken': 0.89, 'paneer': 0.86, 'samosa': 0.91, 'biryani': 0.84,
            'dosa': 0.88, 'idli': 0.90, 'chole': 0.85, 'bread': 0.87, 'egg': 0.92,
            'milk': 0.78, 'pasta': 0.83, 'broccoli': 0.89, 'salmon': 0.86, 'curry': 0.81
        }
        
        # Non-food objects to detect and alert
        self.non_food_objects = [
            'plate', 'bowl', 'spoon', 'fork', 'knife', 'glass', 'cup', 'table', 
            'person', 'hand', 'phone', 'book', 'laptop', 'car', 'building'
        ]
        
        # Load mock model weights (in real implementation, load actual YOLO model)
        self.model_loaded = True
        
    def detect_foods(self, image_path):
        """
        Detect multiple food items in an image
        Returns: Dictionary with detected foods and their properties
        """
        try:
            # Load and process image
            image = cv2.imread(image_path)
            if image is None:
                return {'error': 'Could not load image', 'foods': []}
            
            # Convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image_rgb.shape[:2]
            
            # Mock detection results (in real implementation, use YOLO model)
            detected_foods = self._mock_detection(image_rgb, image_path)
            
            # Check for non-food objects
            non_food_detected = self._detect_non_food_objects(image_rgb)
            
            if non_food_detected:
                return {
                    'error': f'⚠️ Non-food objects detected: {", ".join(non_food_detected)}. Please upload an image containing only food items.',
                    'foods': [],
                    'non_food_objects': non_food_detected
                }
            
            if not detected_foods:
                return {
                    'error': '⚠️ No food items detected in the image. Please upload a clear image of food.',
                    'foods': []
                }
            
            return {
                'error': None,
                'foods': detected_foods,
                'image_dimensions': {'width': width, 'height': height}
            }
            
        except Exception as e:
            return {'error': f'Detection failed: {str(e)}', 'foods': []}
    
    def _mock_detection(self, image, image_path):
        """
        Mock food detection (replace with actual YOLO model in production)
        """
        # Analyze image characteristics to make realistic detections
        height, width = image.shape[:2]
        avg_color = np.mean(image, axis=(0, 1))
        
        detected_foods = []
        
        # Use filename hints or image characteristics for mock detection
        filename = os.path.basename(image_path).lower()
        
        # Check filename for food hints
        possible_foods = []
        for food in self.food_categories.keys():
            if food in filename:
                possible_foods.append(food)
        
        # If no filename hints, use color-based detection
        if not possible_foods:
            possible_foods = self._detect_by_color(avg_color)
        
        # Generate detection results
        for i, food in enumerate(possible_foods[:3]):  # Max 3 foods per image
            # Generate realistic bounding box
            x = random.randint(10, width // 3)
            y = random.randint(10, height // 3)
            w = random.randint(width // 4, width // 2)
            h = random.randint(height // 4, height // 2)
            
            # Ensure box is within image bounds
            x = min(x, width - w - 10)
            y = min(y, height - h - 10)
            
            confidence = self.food_categories.get(food, 0.8) + random.uniform(-0.1, 0.1)
            confidence = max(0.7, min(0.99, confidence))  # Keep confidence between 0.7-0.99
            
            detected_foods.append({
                'name': food,
                'confidence': round(confidence, 2),
                'bbox': {'x': x, 'y': y, 'width': w, 'height': h},
                'estimated_weight': self._estimate_weight_from_bbox(w, h),
                'area_percentage': round((w * h) / (width * height) * 100, 1)
            })
        
        return detected_foods
    
    def _detect_by_color(self, avg_color):
        """Detect likely foods based on average color"""
        r, g, b = avg_color
        
        detected = []
        
        # Green foods
        if g > r and g > b and g > 100:
            detected.extend(['broccoli', 'dal'])
        
        # Orange/Yellow foods
        elif r > 150 and g > 100 and b < 100:
            detected.extend(['orange', 'samosa', 'dosa'])
        
        # Brown foods
        elif r > 100 and g > 80 and b > 60 and r > b:
            detected.extend(['roti', 'bread', 'chicken'])
        
        # White/Light foods
        elif r > 200 and g > 200 and b > 200:
            detected.extend(['rice', 'idli', 'paneer'])
        
        # Default foods if no color match
        if not detected:
            detected = ['rice', 'dal', 'roti']
        
        return detected[:2]  # Return max 2 foods
    
    def _estimate_weight_from_bbox(self, width, height):
        """Estimate food weight based on bounding box dimensions"""
        # Mock weight estimation (in real implementation, use 3D volume estimation)
        area = width * height
        
        # Base weight estimation (very simplified)
        if area < 5000:
            return random.randint(20, 50)  # Small portion
        elif area < 15000:
            return random.randint(50, 120)  # Medium portion
        else:
            return random.randint(120, 250)  # Large portion
    
    def _detect_non_food_objects(self, image):
        """Detect non-food objects and return list of detected items"""
        # Mock detection of non-food objects
        # In real implementation, use object detection model
        
        height, width = image.shape[:2]
        avg_color = np.mean(image)
        
        non_food_detected = []
        
        # Simple heuristics for non-food detection
        # Very bright images might be phones/screens
        if avg_color > 230:
            if random.random() > 0.7:
                non_food_detected.append('phone screen')
        
        # Very dark images might be non-food
        elif avg_color < 50:
            if random.random() > 0.8:
                non_food_detected.append('unclear object')
        
        # Check for very geometric patterns (phones, books, etc.)
        edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 50, 150)
        edge_density = np.sum(edges > 0) / (width * height)
        
        if edge_density > 0.3:  # High edge density might indicate non-food objects
            if random.random() > 0.85:
                non_food_detected.append('geometric object')
        
        return non_food_detected
    
    def check_freshness(self, image_path):
        """
        Check if food appears fresh or decayed
        Returns: Dictionary with freshness status and message
        """
        try:
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Analyze image for freshness indicators
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
                'is_fresh': True,  # Default to fresh if analysis fails
                'freshness_score': 0.7,
                'message': f"Could not analyze freshness: {str(e)}",
                'status': "unknown"
            }
    
    def _calculate_freshness_score(self, image):
        """Calculate freshness score based on image characteristics"""
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Calculate various freshness indicators
        brightness = np.mean(image)
        saturation = np.mean(hsv[:, :, 1])
        
        # Check for brown/dark spots (decay indicators)
        brown_mask = cv2.inRange(hsv, np.array([10, 50, 20]), np.array([20, 255, 200]))
        brown_ratio = np.sum(brown_mask > 0) / (image.shape[0] * image.shape[1])
        
        # Check for green/blue spots (mold indicators)
        green_mask = cv2.inRange(hsv, np.array([40, 50, 50]), np.array([80, 255, 255]))
        green_ratio = np.sum(green_mask > 0) / (image.shape[0] * image.shape[1])
        
        # Calculate freshness score
        freshness_score = 1.0
        
        # Reduce score for decay indicators
        freshness_score -= brown_ratio * 2.0  # Brown spots heavily reduce freshness
        freshness_score -= green_ratio * 3.0  # Green/mold spots severely reduce freshness
        
        # Brightness and saturation factors
        if brightness < 80:  # Too dark might indicate decay
            freshness_score -= 0.2
        if saturation < 50:  # Low saturation might indicate old food
            freshness_score -= 0.1
        
        # Ensure score is between 0 and 1
        freshness_score = max(0.0, min(1.0, freshness_score))
        
        return freshness_score