import cv2
import numpy as np
from PIL import Image
import os
import logging
import random

class FoodDetector:
    def __init__(self):
        """Initialize the food detector with mock capabilities"""
        # Common food items for demonstration
        self.food_items = [
            'rice', 'dal', 'chicken curry', 'roti', 'biryani', 'samosa', 
            'paneer', 'dosa', 'idli', 'chole', 'apple', 'banana', 'orange',
            'bread', 'egg', 'pasta', 'salad', 'pizza', 'sandwich'
        ]
        
        # Weight estimation ranges for different foods (in grams)
        self.weight_ranges = {
            'rice': (150, 300),
            'dal': (100, 200),
            'chicken curry': (120, 250),
            'roti': (30, 50),
            'biryani': (200, 400),
            'samosa': (40, 80),
            'paneer': (80, 150),
            'dosa': (100, 180),
            'idli': (25, 40),
            'chole': (120, 200),
            'apple': (120, 180),
            'banana': (100, 150),
            'orange': (130, 200),
            'bread': (25, 40),
            'egg': (50, 60),
            'pasta': (100, 200),
            'salad': (80, 150),
            'pizza': (150, 300),
            'sandwich': (100, 200)
        }
    
    def detect_foods(self, image_path):
        """
        Mock food detection function that simulates AI detection
        
        Args:
            image_path: Path to the food image
            
        Returns:
            Dictionary with detected foods or error
        """
        try:
            # Check if image file exists
            if not os.path.exists(image_path):
                return {'error': 'Image file not found', 'foods': []}
            
            # Try to open and validate image
            try:
                image = Image.open(image_path)
                image.verify()  # Verify it's a valid image
                image = Image.open(image_path)  # Reopen after verify
            except Exception as e:
                return {'error': f'Invalid image file: {str(e)}', 'foods': []}
            
            # Simulate detection process with random realistic results
            num_foods = random.randint(1, 3)  # Detect 1-3 food items
            detected_foods = []
            
            # Randomly select foods without repetition
            selected_foods = random.sample(self.food_items, min(num_foods, len(self.food_items)))
            
            for food_name in selected_foods:
                # Generate realistic confidence scores
                confidence = random.uniform(0.75, 0.95)
                
                # Estimate weight based on food type
                weight_range = self.weight_ranges.get(food_name, (50, 200))
                estimated_weight = random.randint(weight_range[0], weight_range[1])
                
                # Simulate bounding box coordinates (normalized)
                x = random.uniform(0.1, 0.7)
                y = random.uniform(0.1, 0.7)
                w = random.uniform(0.2, 0.4)
                h = random.uniform(0.2, 0.4)
                
                detected_food = {
                    'name': food_name,
                    'confidence': confidence,
                    'estimated_weight': estimated_weight,
                    'bbox': [x, y, w, h],  # [x, y, width, height] normalized
                    'detection_method': 'mock_ai'
                }
                
                detected_foods.append(detected_food)
            
            return {
                'error': None,
                'foods': detected_foods,
                'image_size': image.size,
                'detection_count': len(detected_foods)
            }
            
        except Exception as e:
            return {
                'error': f'Detection failed: {str(e)}',
                'foods': []
            }
    
    def check_freshness(self, image_path):
        """
        Mock freshness detection
        
        Args:
            image_path: Path to the food image
            
        Returns:
            Dictionary with freshness assessment
        """
        try:
            # Simulate freshness check
            is_fresh = random.choice([True, True, True, False])  # 75% chance of being fresh
            
            if is_fresh:
                freshness_score = random.uniform(0.8, 1.0)
                message = "Food appears fresh and safe to consume."
            else:
                freshness_score = random.uniform(0.3, 0.6)
                message = "Food quality may be compromised. Please check before consuming."
            
            return {
                'is_fresh': is_fresh,
                'freshness_score': freshness_score,
                'message': message,
                'checked_aspects': ['color', 'texture', 'visual_indicators']
            }
            
        except Exception as e:
            return {
                'is_fresh': True,  # Default to fresh if check fails
                'freshness_score': 0.8,
                'message': f"Could not assess freshness: {str(e)}",
                'checked_aspects': []
            }
    
    def analyze_portion_size(self, image_path, food_name):
        """
        Mock portion size analysis
        
        Args:
            image_path: Path to the food image
            food_name: Name of the detected food
            
        Returns:
            Dictionary with portion analysis
        """
        try:
            # Simulate portion size analysis
            portion_sizes = ['small', 'medium', 'large', 'extra_large']
            estimated_portion = random.choice(portion_sizes)
            
            # Confidence in portion estimation
            portion_confidence = random.uniform(0.7, 0.9)
            
            # Size multipliers
            size_multipliers = {
                'small': 0.7,
                'medium': 1.0,
                'large': 1.4,
                'extra_large': 1.8
            }
            
            return {
                'portion_size': estimated_portion,
                'size_multiplier': size_multipliers[estimated_portion],
                'confidence': portion_confidence,
                'analysis_method': 'visual_estimation'
            }
            
        except Exception as e:
            return {
                'portion_size': 'medium',
                'size_multiplier': 1.0,
                'confidence': 0.5,
                'analysis_method': 'default'
            }