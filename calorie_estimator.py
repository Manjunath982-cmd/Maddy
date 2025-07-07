import sqlite3
import numpy as np
from typing import Dict, List, Any

class CalorieEstimator:
    def __init__(self):
        """Initialize the calorie estimator with food database connection"""
        self.density_multipliers = {
            # Density multipliers for volume-to-weight conversion (g/ml)
            'rice': 1.5, 'dal': 1.2, 'curry': 1.1, 'biryani': 1.4,
            'bread': 0.3, 'roti': 0.8, 'naan': 0.7, 'dosa': 0.9,
            'idli': 0.6, 'samosa': 0.8, 'paneer': 1.0, 'chole': 1.3,
            'apple': 0.85, 'banana': 0.9, 'orange': 0.87, 'mango': 0.6,
            'chicken': 1.05, 'fish': 1.0, 'egg': 1.0, 'milk': 1.03,
            'pasta': 1.1, 'pizza': 0.7, 'sandwich': 0.5, 'salad': 0.3
        }
        
        # Portion size corrections based on typical Indian serving sizes
        self.portion_corrections = {
            'rice': 1.2,      # Indians typically eat larger portions of rice
            'dal': 1.0,       # Standard portion
            'roti': 1.0,      # One roti is standard
            'biryani': 1.3,   # Biryani portions are usually larger
            'curry': 1.1,     # Curry portions vary
            'samosa': 0.8,    # Usually 1-2 pieces
            'idli': 0.7,      # 2-3 pieces standard
            'dosa': 1.2,      # One full dosa
            'paneer': 1.0,    # Standard serving
            'chole': 1.1      # Generous portions
        }
    
    def estimate_calories(self, detection_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate total calories and volume from detected foods
        
        Args:
            detection_results: Dictionary containing detected foods from food_detector
            
        Returns:
            Dictionary with calorie estimates, volumes, and nutritional breakdown
        """
        if detection_results.get('error') or not detection_results.get('foods'):
            return {
                'total_calories': 0,
                'total_volume': 0,
                'food_details': [],
                'nutritional_summary': {},
                'error': detection_results.get('error', 'No foods detected')
            }
        
        foods = detection_results['foods']
        total_calories = 0
        total_volume = 0
        food_details = []
        
        # Initialize nutritional totals
        nutritional_totals = {
            'protein': 0, 'carbs': 0, 'fat': 0, 'fiber': 0
        }
        
        for food in foods:
            food_analysis = self._analyze_single_food(food)
            
            if food_analysis:
                total_calories += food_analysis['calories']
                total_volume += food_analysis['volume_ml']
                food_details.append(food_analysis)
                
                # Add to nutritional totals
                for nutrient in nutritional_totals:
                    nutritional_totals[nutrient] += food_analysis.get(nutrient, 0)
        
        # Calculate nutritional percentages and recommendations
        nutritional_summary = self._calculate_nutritional_summary(
            total_calories, nutritional_totals
        )
        
        return {
            'total_calories': round(total_calories, 1),
            'total_volume': round(total_volume, 1),
            'food_details': food_details,
            'nutritional_summary': nutritional_summary,
            'meal_type': self._classify_meal_type(total_calories, foods),
            'health_score': self._calculate_health_score(food_details),
            'error': None
        }
    
    def _analyze_single_food(self, food: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze individual food item and estimate its nutritional content"""
        try:
            food_name = food['name'].lower()
            estimated_weight = food.get('estimated_weight', 100)  # grams
            confidence = food.get('confidence', 0.8)
            
            # Get nutritional data from database
            nutrition_data = self._get_nutrition_from_db(food_name)
            
            if not nutrition_data:
                # Use fallback estimation for unknown foods
                nutrition_data = self._get_fallback_nutrition(food_name)
            
            # Apply portion corrections
            corrected_weight = self._apply_portion_correction(food_name, estimated_weight)
            
            # Calculate volume from weight and density
            volume_ml = self._calculate_volume(food_name, corrected_weight)
            
            # Calculate calories and nutrients per serving
            weight_ratio = corrected_weight / 100.0  # nutrition data is per 100g
            
            calories = nutrition_data['calories_per_100g'] * weight_ratio
            protein = nutrition_data.get('protein', 0) * weight_ratio
            carbs = nutrition_data.get('carbs', 0) * weight_ratio
            fat = nutrition_data.get('fat', 0) * weight_ratio
            fiber = nutrition_data.get('fiber', 0) * weight_ratio
            
            # Apply confidence factor
            confidence_factor = min(confidence, 0.95)  # Cap at 95%
            
            return {
                'name': food['name'],
                'weight_grams': round(corrected_weight, 1),
                'volume_ml': round(volume_ml, 1),
                'calories': round(calories * confidence_factor, 1),
                'protein': round(protein * confidence_factor, 1),
                'carbs': round(carbs * confidence_factor, 1),
                'fat': round(fat * confidence_factor, 1),
                'fiber': round(fiber * confidence_factor, 1),
                'confidence': confidence,
                'cuisine_type': nutrition_data.get('cuisine_type', 'Unknown'),
                'calories_per_100g': nutrition_data['calories_per_100g']
            }
            
        except Exception as e:
            print(f"Error analyzing food {food.get('name', 'unknown')}: {str(e)}")
            return None
    
    def _get_nutrition_from_db(self, food_name: str) -> Dict[str, Any]:
        """Get nutritional information from the food database"""
        try:
            conn = sqlite3.connect('food_app.db')
            c = conn.cursor()
            
            # Try exact match first
            c.execute("""SELECT name, calories_per_100g, protein, carbs, fat, fiber, cuisine_type 
                         FROM food_database 
                         WHERE LOWER(name) = ?""", (food_name,))
            result = c.fetchone()
            
            # If no exact match, try partial match
            if not result:
                c.execute("""SELECT name, calories_per_100g, protein, carbs, fat, fiber, cuisine_type 
                             FROM food_database 
                             WHERE LOWER(name) LIKE ? 
                             ORDER BY 
                                CASE WHEN LOWER(name) LIKE ? THEN 1 ELSE 2 END,
                                LENGTH(name)
                             LIMIT 1""", (f'%{food_name}%', f'{food_name}%'))
                result = c.fetchone()
            
            conn.close()
            
            if result:
                return {
                    'name': result[0],
                    'calories_per_100g': result[1],
                    'protein': result[2] or 0,
                    'carbs': result[3] or 0,
                    'fat': result[4] or 0,
                    'fiber': result[5] or 0,
                    'cuisine_type': result[6] or 'Unknown'
                }
            
            return None
            
        except Exception as e:
            print(f"Database error for {food_name}: {str(e)}")
            return None
    
    def _get_fallback_nutrition(self, food_name: str) -> Dict[str, Any]:
        """Provide fallback nutritional estimates for unknown foods"""
        # Categorize unknown food and provide reasonable estimates
        food_name = food_name.lower()
        
        # Grain/Rice-based foods
        if any(grain in food_name for grain in ['rice', 'grain', 'wheat', 'flour']):
            return {
                'calories_per_100g': 350, 'protein': 8, 'carbs': 70, 'fat': 2, 'fiber': 3,
                'cuisine_type': 'Grain'
            }
        
        # Lentil/Dal-based foods
        elif any(dal in food_name for dal in ['dal', 'lentil', 'bean', 'legume']):
            return {
                'calories_per_100g': 120, 'protein': 9, 'carbs': 20, 'fat': 0.5, 'fiber': 8,
                'cuisine_type': 'Legume'
            }
        
        # Vegetable-based foods
        elif any(veg in food_name for veg in ['vegetable', 'green', 'leaf']):
            return {
                'calories_per_100g': 40, 'protein': 3, 'carbs': 8, 'fat': 0.3, 'fiber': 3,
                'cuisine_type': 'Vegetable'
            }
        
        # Meat/Protein foods
        elif any(meat in food_name for meat in ['chicken', 'meat', 'fish', 'egg']):
            return {
                'calories_per_100g': 180, 'protein': 25, 'carbs': 0, 'fat': 8, 'fiber': 0,
                'cuisine_type': 'Protein'
            }
        
        # Fruit foods
        elif any(fruit in food_name for fruit in ['fruit', 'apple', 'banana', 'orange']):
            return {
                'calories_per_100g': 60, 'protein': 1, 'carbs': 15, 'fat': 0.2, 'fiber': 2,
                'cuisine_type': 'Fruit'
            }
        
        # Fried/Snack foods
        elif any(fried in food_name for fried in ['fried', 'samosa', 'pakoda', 'chips']):
            return {
                'calories_per_100g': 300, 'protein': 6, 'carbs': 30, 'fat': 18, 'fiber': 2,
                'cuisine_type': 'Snack'
            }
        
        # Default unknown food
        else:
            return {
                'calories_per_100g': 150, 'protein': 5, 'carbs': 25, 'fat': 5, 'fiber': 2,
                'cuisine_type': 'Unknown'
            }
    
    def _apply_portion_correction(self, food_name: str, estimated_weight: float) -> float:
        """Apply portion size corrections based on typical serving sizes"""
        food_name = food_name.lower()
        
        # Find the best matching correction factor
        correction_factor = 1.0
        for food_type, factor in self.portion_corrections.items():
            if food_type in food_name:
                correction_factor = factor
                break
        
        return estimated_weight * correction_factor
    
    def _calculate_volume(self, food_name: str, weight_grams: float) -> float:
        """Calculate volume in ml from weight and density"""
        food_name = food_name.lower()
        
        # Find the best matching density multiplier
        density = 1.0  # Default density (water equivalent)
        for food_type, mult in self.density_multipliers.items():
            if food_type in food_name:
                density = mult
                break
        
        # Volume = mass / density
        volume_ml = weight_grams / density
        return max(volume_ml, 1.0)  # Minimum 1ml
    
    def _calculate_nutritional_summary(self, total_calories: float, totals: Dict[str, float]) -> Dict[str, Any]:
        """Calculate nutritional summary and daily value percentages"""
        # Recommended daily values (approximate for an average adult)
        daily_values = {
            'calories': 2000,
            'protein': 50,    # grams
            'carbs': 300,     # grams
            'fat': 65,        # grams
            'fiber': 25       # grams
        }
        
        summary = {
            'calories': {
                'amount': round(total_calories, 1),
                'daily_percentage': round((total_calories / daily_values['calories']) * 100, 1)
            }
        }
        
        for nutrient, amount in totals.items():
            summary[nutrient] = {
                'amount': round(amount, 1),
                'daily_percentage': round((amount / daily_values[nutrient]) * 100, 1)
            }
        
        # Calculate calorie distribution
        protein_calories = totals['protein'] * 4  # 4 cal/g
        carb_calories = totals['carbs'] * 4       # 4 cal/g
        fat_calories = totals['fat'] * 9          # 9 cal/g
        
        total_macro_calories = protein_calories + carb_calories + fat_calories
        
        if total_macro_calories > 0:
            summary['calorie_distribution'] = {
                'protein_percent': round((protein_calories / total_macro_calories) * 100, 1),
                'carbs_percent': round((carb_calories / total_macro_calories) * 100, 1),
                'fat_percent': round((fat_calories / total_macro_calories) * 100, 1)
            }
        else:
            summary['calorie_distribution'] = {
                'protein_percent': 0, 'carbs_percent': 0, 'fat_percent': 0
            }
        
        return summary
    
    def _classify_meal_type(self, total_calories: float, foods: List[Dict]) -> str:
        """Classify the detected foods as breakfast, lunch, dinner, or snack"""
        # Simple classification based on calories and food types
        if total_calories < 200:
            return "Light Snack"
        elif total_calories < 400:
            return "Snack/Light Meal"
        elif total_calories < 600:
            return "Small Meal"
        elif total_calories < 800:
            return "Regular Meal"
        else:
            return "Large Meal"
    
    def _calculate_health_score(self, food_details: List[Dict]) -> Dict[str, Any]:
        """Calculate a health score based on nutritional balance"""
        if not food_details:
            return {'score': 0, 'rating': 'No Data', 'recommendations': []}
        
        total_calories = sum(food['calories'] for food in food_details)
        total_fiber = sum(food.get('fiber', 0) for food in food_details)
        total_protein = sum(food.get('protein', 0) for food in food_details)
        
        # Calculate health metrics
        fiber_score = min(total_fiber / 5.0, 1.0) * 30  # Max 30 points for fiber
        protein_score = min(total_protein / 15.0, 1.0) * 30  # Max 30 points for protein
        
        # Variety score (different cuisines/types)
        cuisine_types = set(food.get('cuisine_type', 'Unknown') for food in food_details)
        variety_score = min(len(cuisine_types) / 3.0, 1.0) * 20  # Max 20 points for variety
        
        # Calorie density score (lower is better for health)
        avg_calorie_density = total_calories / len(food_details)
        density_score = max(0, (300 - avg_calorie_density) / 300) * 20  # Max 20 points
        
        total_score = fiber_score + protein_score + variety_score + density_score
        
        # Rating classification
        if total_score >= 80:
            rating = "Excellent"
        elif total_score >= 60:
            rating = "Good"
        elif total_score >= 40:
            rating = "Fair"
        else:
            rating = "Needs Improvement"
        
        # Generate recommendations
        recommendations = []
        if fiber_score < 15:
            recommendations.append("Add more vegetables and fruits for fiber")
        if protein_score < 15:
            recommendations.append("Include more protein sources")
        if variety_score < 10:
            recommendations.append("Try to include different food groups")
        if density_score < 10:
            recommendations.append("Choose lower calorie density foods")
        
        return {
            'score': round(total_score, 1),
            'rating': rating,
            'recommendations': recommendations
        }