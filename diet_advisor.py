import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any
import random

class DietAdvisor:
    def __init__(self):
        """Initialize the diet advisor with recommendation algorithms"""
        self.bmr_multipliers = {
            'sedentary': 1.2,
            'lightly_active': 1.375,
            'moderately_active': 1.55,
            'very_active': 1.725,
            'extremely_active': 1.9
        }
        
        # Food recommendations based on goals
        self.goal_foods = {
            'weight_loss': {
                'recommended': ['broccoli', 'apple', 'chicken breast', 'dal', 'green vegetables'],
                'avoid': ['samosa', 'fried foods', 'sweets', 'high calorie snacks'],
                'calorie_deficit': 500  # calories per day
            },
            'weight_gain': {
                'recommended': ['nuts', 'paneer', 'biryani', 'milk', 'protein rich foods'],
                'avoid': ['excessive low calorie foods'],
                'calorie_surplus': 500  # calories per day
            },
            'maintain': {
                'recommended': ['balanced meals', 'variety of foods', 'fruits', 'vegetables'],
                'avoid': ['excessive anything'],
                'calorie_adjustment': 0
            },
            'muscle_gain': {
                'recommended': ['chicken', 'eggs', 'dal', 'paneer', 'fish', 'protein sources'],
                'avoid': ['empty calories', 'too much sugar'],
                'protein_focus': True
            }
        }
        
        # Indian meal suggestions
        self.indian_meal_suggestions = {
            'breakfast': [
                {'name': 'Idli with Sambar', 'calories': 200, 'type': 'light'},
                {'name': 'Poha with vegetables', 'calories': 250, 'type': 'medium'},
                {'name': 'Dosa with chutney', 'calories': 300, 'type': 'medium'},
                {'name': 'Upma with nuts', 'calories': 280, 'type': 'medium'},
                {'name': 'Paratha with curd', 'calories': 350, 'type': 'heavy'}
            ],
            'lunch': [
                {'name': 'Rice, Dal, Vegetable curry', 'calories': 450, 'type': 'balanced'},
                {'name': 'Roti, Paneer curry, Salad', 'calories': 400, 'type': 'balanced'},
                {'name': 'Biryani with raita', 'calories': 600, 'type': 'heavy'},
                {'name': 'Chole with rice', 'calories': 500, 'type': 'medium'},
                {'name': 'Mixed dal with vegetables', 'calories': 350, 'type': 'light'}
            ],
            'dinner': [
                {'name': 'Light dal with roti', 'calories': 300, 'type': 'light'},
                {'name': 'Vegetable soup with bread', 'calories': 200, 'type': 'light'},
                {'name': 'Grilled chicken with salad', 'calories': 350, 'type': 'protein'},
                {'name': 'Khichdi with ghee', 'calories': 280, 'type': 'comfort'},
                {'name': 'Fish curry with rice', 'calories': 450, 'type': 'balanced'}
            ],
            'snacks': [
                {'name': 'Mixed nuts', 'calories': 150, 'type': 'healthy'},
                {'name': 'Fruits (apple/banana)', 'calories': 80, 'type': 'healthy'},
                {'name': 'Yogurt with fruits', 'calories': 120, 'type': 'healthy'},
                {'name': 'Green tea with biscuits', 'calories': 100, 'type': 'light'}
            ]
        }
    
    def get_suggestions(self, user_id: int, calorie_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate personalized diet suggestions based on user profile and current intake
        
        Args:
            user_id: User ID to get profile information
            calorie_results: Results from current meal analysis
            
        Returns:
            List of personalized suggestions
        """
        try:
            # Get user profile
            user_profile = self._get_user_profile(user_id)
            if not user_profile:
                return self._get_generic_suggestions()
            
            # Calculate daily calorie needs
            daily_needs = self._calculate_daily_needs(user_profile)
            
            # Get today's total intake
            today_intake = self._get_today_intake(user_id)
            
            # Get recent eating patterns
            eating_patterns = self._analyze_eating_patterns(user_id)
            
            # Generate suggestions
            suggestions = []
            
            # Calorie-based suggestions
            suggestions.extend(self._get_calorie_suggestions(
                daily_needs, today_intake, calorie_results['total_calories']
            ))
            
            # Goal-based suggestions
            suggestions.extend(self._get_goal_suggestions(
                user_profile['diet_goal'], calorie_results
            ))
            
            # Health-based suggestions
            suggestions.extend(self._get_health_suggestions(calorie_results))
            
            # Meal timing suggestions
            suggestions.extend(self._get_timing_suggestions())
            
            # Pattern-based suggestions
            suggestions.extend(self._get_pattern_suggestions(eating_patterns))
            
            # Save suggestions to database
            self._save_suggestions(user_id, suggestions)
            
            return suggestions[:5]  # Return top 5 suggestions
            
        except Exception as e:
            print(f"Error generating suggestions: {str(e)}")
            return self._get_generic_suggestions()
    
    def _get_user_profile(self, user_id: int) -> Dict[str, Any]:
        """Get user profile information from database"""
        try:
            conn = sqlite3.connect('food_app.db')
            c = conn.cursor()
            c.execute("""SELECT age, weight, height, gender, activity_level, diet_goal 
                         FROM users WHERE id = ?""", (user_id,))
            result = c.fetchone()
            conn.close()
            
            if result:
                return {
                    'age': result[0],
                    'weight': result[1],
                    'height': result[2],
                    'gender': result[3],
                    'activity_level': result[4],
                    'diet_goal': result[5]
                }
            return None
            
        except Exception as e:
            print(f"Error getting user profile: {str(e)}")
            return None
    
    def _calculate_daily_needs(self, profile: Dict[str, Any]) -> Dict[str, float]:
        """Calculate daily calorie and nutritional needs using Mifflin-St Jeor equation"""
        age = profile.get('age', 30)
        weight = profile.get('weight', 70)  # kg
        height = profile.get('height', 170)  # cm
        gender = profile.get('gender', 'male')
        activity = profile.get('activity_level', 'moderately_active')
        goal = profile.get('diet_goal', 'maintain')
        
        # Calculate BMR (Basal Metabolic Rate)
        if gender == 'male':
            bmr = 10 * weight + 6.25 * height - 5 * age + 5
        else:
            bmr = 10 * weight + 6.25 * height - 5 * age - 161
        
        # Apply activity multiplier
        multiplier = self.bmr_multipliers.get(activity, 1.55)
        daily_calories = bmr * multiplier
        
        # Adjust for goals
        goal_adjustment = self.goal_foods.get(goal, {}).get('calorie_deficit', 0)
        if goal == 'weight_loss':
            daily_calories -= goal_adjustment
        elif goal == 'weight_gain':
            daily_calories += self.goal_foods[goal]['calorie_surplus']
        
        # Calculate macronutrient needs
        protein_grams = weight * 1.2  # 1.2g per kg body weight
        fat_grams = daily_calories * 0.25 / 9  # 25% of calories from fat
        carb_grams = (daily_calories - (protein_grams * 4) - (fat_grams * 9)) / 4
        
        return {
            'calories': round(daily_calories),
            'protein': round(protein_grams, 1),
            'carbs': round(carb_grams, 1),
            'fat': round(fat_grams, 1),
            'fiber': 25  # Standard recommendation
        }
    
    def _get_today_intake(self, user_id: int) -> Dict[str, float]:
        """Get today's total nutritional intake"""
        try:
            today = datetime.now().date()
            conn = sqlite3.connect('food_app.db')
            c = conn.cursor()
            c.execute("""SELECT SUM(total_calories), SUM(total_volume) 
                         FROM food_estimates 
                         WHERE user_id = ? AND DATE(created_at) = ?""", 
                     (user_id, today))
            result = c.fetchone()
            conn.close()
            
            return {
                'calories': result[0] or 0,
                'volume': result[1] or 0
            }
            
        except Exception as e:
            print(f"Error getting today's intake: {str(e)}")
            return {'calories': 0, 'volume': 0}
    
    def _analyze_eating_patterns(self, user_id: int) -> Dict[str, Any]:
        """Analyze user's eating patterns over the last week"""
        try:
            week_ago = datetime.now() - timedelta(days=7)
            conn = sqlite3.connect('food_app.db')
            c = conn.cursor()
            c.execute("""SELECT detected_foods, total_calories, created_at 
                         FROM food_estimates 
                         WHERE user_id = ? AND created_at >= ?
                         ORDER BY created_at DESC""", 
                     (user_id, week_ago))
            results = c.fetchall()
            conn.close()
            
            if not results:
                return {'average_calories': 0, 'frequent_foods': [], 'meal_frequency': 0}
            
            total_calories = sum(row[1] for row in results)
            avg_calories = total_calories / max(len(results), 1)
            
            # Extract frequent foods
            all_foods = []
            for row in results:
                try:
                    foods = eval(row[0]) if row[0] else []
                    all_foods.extend([food.get('name', '') for food in foods])
                except:
                    continue
            
            # Count food frequency
            food_counts = {}
            for food in all_foods:
                food_counts[food] = food_counts.get(food, 0) + 1
            
            frequent_foods = sorted(food_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            
            return {
                'average_calories': round(avg_calories),
                'frequent_foods': [food[0] for food in frequent_foods],
                'meal_frequency': len(results),
                'total_days': 7
            }
            
        except Exception as e:
            print(f"Error analyzing patterns: {str(e)}")
            return {'average_calories': 0, 'frequent_foods': [], 'meal_frequency': 0}
    
    def _get_calorie_suggestions(self, daily_needs: Dict, today_intake: Dict, current_meal: float) -> List[Dict]:
        """Generate calorie-based suggestions"""
        suggestions = []
        
        remaining_calories = daily_needs['calories'] - today_intake['calories'] - current_meal
        
        if remaining_calories > 800:
            suggestions.append({
                'type': 'calorie_target',
                'message': f"You have {int(remaining_calories)} calories remaining today. Consider having 2-3 more balanced meals.",
                'priority': 'medium'
            })
        elif remaining_calories > 400:
            suggestions.append({
                'type': 'calorie_target',
                'message': f"You have {int(remaining_calories)} calories left. A light dinner would be perfect!",
                'priority': 'medium'
            })
        elif remaining_calories > 0:
            suggestions.append({
                'type': 'calorie_target',
                'message': f"You have {int(remaining_calories)} calories remaining. Consider a healthy snack.",
                'priority': 'low'
            })
        else:
            suggestions.append({
                'type': 'calorie_warning',
                'message': "You've reached your daily calorie target. Consider lighter meals for the rest of the day.",
                'priority': 'high'
            })
        
        return suggestions
    
    def _get_goal_suggestions(self, goal: str, calorie_results: Dict) -> List[Dict]:
        """Generate goal-based suggestions"""
        suggestions = []
        
        if not goal or goal not in self.goal_foods:
            return suggestions
        
        goal_info = self.goal_foods[goal]
        
        if goal == 'weight_loss':
            suggestions.append({
                'type': 'goal_advice',
                'message': "Focus on high-fiber, low-calorie foods. Add more vegetables and lean proteins.",
                'priority': 'medium'
            })
            
            # Suggest specific foods
            recommended = random.choice(goal_info['recommended'])
            suggestions.append({
                'type': 'food_suggestion',
                'message': f"Try adding {recommended} to your next meal for better nutrition.",
                'priority': 'low'
            })
        
        elif goal == 'weight_gain':
            suggestions.append({
                'type': 'goal_advice',
                'message': "Include more calorie-dense, nutritious foods. Don't skip meals!",
                'priority': 'medium'
            })
        
        elif goal == 'muscle_gain':
            protein_in_meal = sum(food.get('protein', 0) for food in calorie_results.get('food_details', []))
            if protein_in_meal < 20:
                suggestions.append({
                    'type': 'protein_advice',
                    'message': "This meal is low in protein. Add eggs, chicken, dal, or paneer for muscle building.",
                    'priority': 'high'
                })
        
        return suggestions
    
    def _get_health_suggestions(self, calorie_results: Dict) -> List[Dict]:
        """Generate health-based suggestions"""
        suggestions = []
        
        health_score = calorie_results.get('health_score', {})
        recommendations = health_score.get('recommendations', [])
        
        for rec in recommendations:
            suggestions.append({
                'type': 'health_improvement',
                'message': rec,
                'priority': 'medium'
            })
        
        # Check for freshness issues
        if not calorie_results.get('is_fresh', True):
            suggestions.append({
                'type': 'food_safety',
                'message': "⚠️ Some food items appear questionable. Always ensure food freshness for safety!",
                'priority': 'high'
            })
        
        return suggestions
    
    def _get_timing_suggestions(self) -> List[Dict]:
        """Generate meal timing suggestions"""
        current_hour = datetime.now().hour
        suggestions = []
        
        if 6 <= current_hour <= 10:
            suggestions.append({
                'type': 'timing',
                'message': "Perfect time for breakfast! Start your day with a nutritious meal.",
                'priority': 'low'
            })
        elif 12 <= current_hour <= 14:
            suggestions.append({
                'type': 'timing',
                'message': "Lunch time! Make sure to include vegetables and proteins.",
                'priority': 'low'
            })
        elif 19 <= current_hour <= 21:
            suggestions.append({
                'type': 'timing',
                'message': "Dinner time! Keep it light and easy to digest.",
                'priority': 'low'
            })
        elif current_hour >= 22:
            suggestions.append({
                'type': 'timing',
                'message': "Late night eating! Consider lighter options to avoid sleep disruption.",
                'priority': 'medium'
            })
        
        return suggestions
    
    def _get_pattern_suggestions(self, patterns: Dict) -> List[Dict]:
        """Generate suggestions based on eating patterns"""
        suggestions = []
        
        avg_calories = patterns.get('average_calories', 0)
        frequent_foods = patterns.get('frequent_foods', [])
        meal_frequency = patterns.get('meal_frequency', 0)
        
        if meal_frequency < 14:  # Less than 2 meals per day average
            suggestions.append({
                'type': 'pattern_advice',
                'message': "You're eating irregularly. Try to maintain consistent meal times for better metabolism.",
                'priority': 'medium'
            })
        
        if len(frequent_foods) <= 2:
            suggestions.append({
                'type': 'variety_advice',
                'message': "Add more variety to your diet! Try different vegetables and protein sources.",
                'priority': 'low'
            })
        
        return suggestions
    
    def _get_generic_suggestions(self) -> List[Dict]:
        """Generate generic suggestions when user profile is not available"""
        generic_suggestions = [
            {
                'type': 'general',
                'message': "Include a variety of colorful vegetables in your meals for better nutrition.",
                'priority': 'medium'
            },
            {
                'type': 'general',
                'message': "Drink plenty of water throughout the day to stay hydrated.",
                'priority': 'low'
            },
            {
                'type': 'general',
                'message': "Try to eat at regular intervals to maintain stable energy levels.",
                'priority': 'medium'
            }
        ]
        
        return random.sample(generic_suggestions, min(3, len(generic_suggestions)))
    
    def _save_suggestions(self, user_id: int, suggestions: List[Dict]):
        """Save generated suggestions to database"""
        try:
            conn = sqlite3.connect('food_app.db')
            c = conn.cursor()
            
            for suggestion in suggestions:
                c.execute("""INSERT INTO diet_suggestions (user_id, suggestion_type, suggestion_text) 
                             VALUES (?, ?, ?)""",
                         (user_id, suggestion['type'], suggestion['message']))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error saving suggestions: {str(e)}")
    
    def get_meal_plan(self, user_id: int, meal_type: str) -> Dict[str, Any]:
        """Generate a meal plan suggestion for specific meal type"""
        try:
            user_profile = self._get_user_profile(user_id)
            daily_needs = self._calculate_daily_needs(user_profile) if user_profile else None
            
            meal_suggestions = self.indian_meal_suggestions.get(meal_type, [])
            
            if not meal_suggestions:
                return {'error': 'Invalid meal type'}
            
            # Filter suggestions based on user goals and needs
            if user_profile and daily_needs:
                goal = user_profile.get('diet_goal', 'maintain')
                
                if goal == 'weight_loss':
                    # Prefer lighter options
                    filtered = [meal for meal in meal_suggestions if meal['calories'] < 400]
                elif goal == 'weight_gain':
                    # Prefer heavier options
                    filtered = [meal for meal in meal_suggestions if meal['calories'] > 300]
                else:
                    filtered = meal_suggestions
                
                meal_suggestions = filtered if filtered else meal_suggestions
            
            # Select a random suggestion
            selected_meal = random.choice(meal_suggestions)
            
            return {
                'meal_name': selected_meal['name'],
                'estimated_calories': selected_meal['calories'],
                'meal_type': selected_meal['type'],
                'timing': meal_type,
                'nutritional_advice': self._get_meal_advice(selected_meal, user_profile)
            }
            
        except Exception as e:
            return {'error': f'Error generating meal plan: {str(e)}'}
    
    def _get_meal_advice(self, meal: Dict, profile: Dict) -> str:
        """Get specific advice for a meal based on user profile"""
        if not profile:
            return "Enjoy your meal and eat mindfully!"
        
        goal = profile.get('diet_goal', 'maintain')
        
        if goal == 'weight_loss' and meal['calories'] > 400:
            return "Consider smaller portions or add more vegetables to reduce calories."
        elif goal == 'weight_gain' and meal['calories'] < 300:
            return "Add some nuts, ghee, or extra protein to increase the calorie content."
        elif goal == 'muscle_gain':
            return "Great choice! Make sure to include adequate protein in this meal."
        else:
            return "This looks like a balanced meal. Enjoy it!"