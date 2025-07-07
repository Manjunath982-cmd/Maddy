import sqlite3
import random
from datetime import datetime, timedelta
import json

class DietAdvisor:
    def __init__(self):
        """Initialize the diet advisor with predefined suggestions and knowledge base"""
        
        # Health-focused diet suggestions
        self.health_suggestions = {
            'weight_loss': [
                "Consider reducing portion sizes by 20% to create a calorie deficit.",
                "Include more protein-rich foods to maintain muscle mass during weight loss.",
                "Add more fiber-rich vegetables to feel full with fewer calories.",
                "Try intermittent fasting - eating within an 8-hour window.",
                "Replace refined carbs with whole grain alternatives.",
                "Drink water before meals to help control portion sizes."
            ],
            'weight_gain': [
                "Add healthy fats like nuts, avocados, and olive oil to increase calories.",
                "Include protein shakes between meals for extra nutrition.",
                "Eat more frequent, smaller meals throughout the day.",
                "Focus on calorie-dense foods like dried fruits and nut butters.",
                "Add strength training to build muscle mass while gaining weight."
            ],
            'maintain': [
                "Continue your current eating patterns - they're working well!",
                "Focus on maintaining variety in your diet for optimal nutrition.",
                "Listen to your body's hunger and fullness cues.",
                "Stay hydrated and maintain regular meal timing."
            ]
        }
        
        # Activity level recommendations
        self.activity_suggestions = {
            'sedentary': [
                "Try to add a 30-minute walk to your daily routine.",
                "Consider standing desk work or frequent movement breaks.",
                "Start with light exercises like stretching or yoga."
            ],
            'lightly_active': [
                "Great job staying active! Try to add one more workout per week.",
                "Include some strength training to complement your cardio.",
                "Consider joining a sports club or fitness class."
            ],
            'moderately_active': [
                "Excellent activity level! Maintain your current routine.",
                "Mix up your workouts to prevent boredom.",
                "Focus on recovery and proper nutrition to fuel your activities."
            ],
            'very_active': [
                "Outstanding fitness commitment! Ensure you're eating enough to fuel your workouts.",
                "Pay attention to recovery and rest days.",
                "Consider working with a nutritionist to optimize your diet."
            ]
        }
        
        # Nutrition-based suggestions
        self.nutrition_suggestions = {
            'high_calorie': [
                "Your calorie intake seems high. Consider smaller portions.",
                "Focus on nutrient-dense foods rather than empty calories.",
                "Try eating more slowly to help your body recognize fullness.",
                "Include more vegetables to balance your meal."
            ],
            'low_calorie': [
                "Your calorie intake might be too low. Consider adding healthy snacks.",
                "Include more protein to maintain energy levels.",
                "Don't skip meals - consistent eating helps metabolism."
            ],
            'balanced': [
                "Great job maintaining a balanced calorie intake!",
                "Your eating patterns look healthy and sustainable.",
                "Continue focusing on variety and moderation."
            ],
            'high_carb': [
                "Consider balancing carbs with more protein and healthy fats.",
                "Choose complex carbohydrates over simple sugars.",
                "Add more vegetables to your carb-heavy meals."
            ],
            'high_protein': [
                "Good protein intake! Make sure to include variety in protein sources.",
                "Balance with adequate carbs for energy and fiber for digestion.",
                "Stay hydrated as protein metabolism requires more water."
            ],
            'high_fat': [
                "Focus on healthy fats like avocados, nuts, and olive oil.",
                "Balance with lean proteins and complex carbohydrates.",
                "Monitor portion sizes as fats are calorie-dense."
            ]
        }
        
        # Meal timing suggestions
        self.timing_suggestions = [
            "Try to eat regular meals at consistent times each day.",
            "Don't skip breakfast - it helps kickstart your metabolism.",
            "Avoid eating large meals close to bedtime.",
            "Consider a light snack if more than 4 hours between meals.",
            "Stay hydrated throughout the day with water."
        ]
        
        # Indian diet specific suggestions
        self.indian_diet_suggestions = [
            "Include a variety of dals and legumes for complete proteins.",
            "Use turmeric and other spices - they have anti-inflammatory properties.",
            "Balance rice with vegetable curries and protein sources.",
            "Try millets as alternatives to rice for better nutrition.",
            "Include curd/yogurt for probiotics and calcium.",
            "Use ghee in moderation for healthy fats.",
            "Include seasonal fruits and vegetables for variety."
        ]
    
    def get_suggestions(self, user_id, calorie_results):
        """
        Generate personalized diet suggestions based on user data and current intake
        
        Args:
            user_id: ID of the user
            calorie_results: Results from calorie estimation
            
        Returns:
            List of personalized suggestions
        """
        try:
            # Get user profile
            user_profile = self._get_user_profile(user_id)
            
            # Get recent eating patterns
            recent_intake = self._get_recent_intake(user_id, days=7)
            
            # Generate suggestions based on current meal
            meal_suggestions = self._analyze_current_meal(calorie_results)
            
            # Generate suggestions based on user goals
            goal_suggestions = self._get_goal_based_suggestions(user_profile)
            
            # Generate nutrition balance suggestions
            nutrition_suggestions = self._get_nutrition_suggestions(calorie_results, recent_intake)
            
            # Generate timing suggestions
            timing_suggestions = self._get_timing_suggestions(user_id)
            
            # Combine and prioritize suggestions
            all_suggestions = []
            all_suggestions.extend(meal_suggestions[:2])
            all_suggestions.extend(goal_suggestions[:2])
            all_suggestions.extend(nutrition_suggestions[:1])
            all_suggestions.extend(timing_suggestions[:1])
            
            # Add some general health tips
            if len(all_suggestions) < 5:
                general_tips = random.sample(self.indian_diet_suggestions, 2)
                all_suggestions.extend(general_tips)
            
            # Save suggestions to database
            self._save_suggestions(user_id, all_suggestions)
            
            return all_suggestions[:5]  # Return top 5 suggestions
            
        except Exception as e:
            print(f"Error generating suggestions: {str(e)}")
            return self._get_default_suggestions()
    
    def _get_user_profile(self, user_id):
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
                    'age': result[0] or 25,
                    'weight': result[1] or 70,
                    'height': result[2] or 170,
                    'gender': result[3] or 'other',
                    'activity_level': result[4] or 'lightly_active',
                    'diet_goal': result[5] or 'maintain'
                }
            else:
                return self._get_default_profile()
                
        except Exception as e:
            print(f"Error getting user profile: {str(e)}")
            return self._get_default_profile()
    
    def _get_default_profile(self):
        """Return default user profile"""
        return {
            'age': 25,
            'weight': 70,
            'height': 170,
            'gender': 'other',
            'activity_level': 'lightly_active',
            'diet_goal': 'maintain'
        }
    
    def _get_recent_intake(self, user_id, days=7):
        """Get recent food intake data"""
        try:
            conn = sqlite3.connect('food_app.db')
            c = conn.cursor()
            
            # Get intake from last N days
            cutoff_date = datetime.now() - timedelta(days=days)
            c.execute("""SELECT total_calories, detected_foods, created_at 
                         FROM food_estimates 
                         WHERE user_id = ? AND created_at >= ?
                         ORDER BY created_at DESC""", 
                     (user_id, cutoff_date))
            
            results = c.fetchall()
            conn.close()
            
            total_calories = sum(row[0] or 0 for row in results)
            avg_daily_calories = total_calories / max(days, 1)
            
            return {
                'total_calories': total_calories,
                'avg_daily_calories': avg_daily_calories,
                'meal_count': len(results),
                'days': days
            }
            
        except Exception as e:
            print(f"Error getting recent intake: {str(e)}")
            return {
                'total_calories': 0,
                'avg_daily_calories': 0,
                'meal_count': 0,
                'days': days
            }
    
    def _analyze_current_meal(self, calorie_results):
        """Analyze current meal and provide specific suggestions"""
        suggestions = []
        
        total_calories = calorie_results.get('total_calories', 0)
        food_details = calorie_results.get('food_details', [])
        
        # Analyze calorie content
        if total_calories > 800:
            suggestions.append("This meal is quite calorie-dense. Consider reducing portion sizes or adding more vegetables.")
        elif total_calories < 200:
            suggestions.append("This meal seems light. Consider adding some protein or healthy fats for better satiety.")
        else:
            suggestions.append("Good calorie content for this meal!")
        
        # Analyze food variety
        if len(food_details) == 1:
            suggestions.append("Try to include multiple food groups in your meal for better nutrition balance.")
        elif len(food_details) >= 3:
            suggestions.append("Great variety in your meal! This helps ensure you get diverse nutrients.")
        
        # Analyze specific foods
        food_names = [food.get('name', '').lower() for food in food_details]
        
        if any('rice' in name or 'bread' in name for name in food_names):
            if not any('vegetable' in name or 'salad' in name for name in food_names):
                suggestions.append("Consider adding some vegetables to balance the carbohydrates in your meal.")
        
        if any('fried' in name or 'samosa' in name for name in food_names):
            suggestions.append("You have some fried food. Try to balance with fresh vegetables or fruits.")
        
        return suggestions
    
    def _get_goal_based_suggestions(self, user_profile):
        """Get suggestions based on user's diet goals"""
        diet_goal = user_profile.get('diet_goal', 'maintain')
        activity_level = user_profile.get('activity_level', 'lightly_active')
        
        suggestions = []
        
        # Add goal-specific suggestions
        if diet_goal in self.health_suggestions:
            suggestions.extend(random.sample(self.health_suggestions[diet_goal], 2))
        
        # Add activity-specific suggestions
        if activity_level in self.activity_suggestions:
            suggestions.extend(random.sample(self.activity_suggestions[activity_level], 1))
        
        return suggestions
    
    def _get_nutrition_suggestions(self, calorie_results, recent_intake):
        """Generate nutrition-based suggestions"""
        suggestions = []
        
        # Analyze calorie patterns
        avg_calories = recent_intake.get('avg_daily_calories', 0)
        
        if avg_calories > 2500:
            suggestions.extend(random.sample(self.nutrition_suggestions['high_calorie'], 1))
        elif avg_calories < 1500:
            suggestions.extend(random.sample(self.nutrition_suggestions['low_calorie'], 1))
        else:
            suggestions.extend(random.sample(self.nutrition_suggestions['balanced'], 1))
        
        return suggestions
    
    def _get_timing_suggestions(self, user_id):
        """Generate meal timing suggestions"""
        suggestions = []
        
        # Add general timing suggestions
        suggestions.extend(random.sample(self.timing_suggestions, 1))
        
        return suggestions
    
    def _save_suggestions(self, user_id, suggestions):
        """Save suggestions to database"""
        try:
            conn = sqlite3.connect('food_app.db')
            c = conn.cursor()
            
            for suggestion in suggestions:
                c.execute("""INSERT INTO diet_suggestions (user_id, suggestion_type, suggestion_text) 
                             VALUES (?, ?, ?)""",
                         (user_id, 'ai_generated', suggestion))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error saving suggestions: {str(e)}")
    
    def _get_default_suggestions(self):
        """Return default suggestions when there's an error"""
        return [
            "Maintain a balanced diet with vegetables, proteins, and whole grains.",
            "Stay hydrated by drinking plenty of water throughout the day.",
            "Try to eat regular meals at consistent times.",
            "Include a variety of colorful fruits and vegetables in your diet.",
            "Practice portion control to maintain a healthy weight."
        ]
    
    def get_weekly_report(self, user_id):
        """Generate a weekly nutrition report"""
        try:
            recent_intake = self._get_recent_intake(user_id, days=7)
            user_profile = self._get_user_profile(user_id)
            
            # Calculate recommended daily calories
            bmr = self._calculate_bmr(user_profile)
            recommended_calories = self._calculate_daily_calories(bmr, user_profile['activity_level'])
            
            report = {
                'period': '7 days',
                'total_calories': recent_intake['total_calories'],
                'avg_daily_calories': recent_intake['avg_daily_calories'],
                'recommended_daily_calories': recommended_calories,
                'meal_count': recent_intake['meal_count'],
                'compliance': self._calculate_compliance(recent_intake['avg_daily_calories'], recommended_calories),
                'suggestions': self.get_suggestions(user_id, {'total_calories': recent_intake['avg_daily_calories']}),
                'generated_at': datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            print(f"Error generating weekly report: {str(e)}")
            return None
    
    def _calculate_bmr(self, user_profile):
        """Calculate Basal Metabolic Rate using Harris-Benedict equation"""
        age = user_profile['age']
        weight = user_profile['weight']
        height = user_profile['height']
        gender = user_profile['gender']
        
        if gender.lower() == 'male':
            bmr = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
        else:  # female or other
            bmr = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)
        
        return bmr
    
    def _calculate_daily_calories(self, bmr, activity_level):
        """Calculate daily calorie needs based on activity level"""
        activity_multipliers = {
            'sedentary': 1.2,
            'lightly_active': 1.375,
            'moderately_active': 1.55,
            'very_active': 1.725,
            'extremely_active': 1.9
        }
        
        multiplier = activity_multipliers.get(activity_level, 1.375)
        return int(bmr * multiplier)
    
    def _calculate_compliance(self, actual_calories, recommended_calories):
        """Calculate how well user is meeting their calorie goals"""
        if recommended_calories == 0:
            return 0
        
        ratio = actual_calories / recommended_calories
        
        if 0.9 <= ratio <= 1.1:
            return 100  # Excellent compliance
        elif 0.8 <= ratio <= 1.2:
            return 80   # Good compliance
        elif 0.7 <= ratio <= 1.3:
            return 60   # Fair compliance
        else:
            return 40   # Poor compliance