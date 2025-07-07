import pandas as pd
import numpy as np
import random
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define realistic car data
car_brands_models = {
    'Maruti Suzuki': ['Swift', 'Alto', 'Baleno', 'Vitara Brezza', 'Dzire', 'Wagon R', 'Ertiga', 'Ciaz', 'S-Cross'],
    'Hyundai': ['i20', 'Grand i10', 'Creta', 'Verna', 'Elite i20', 'Xcent', 'Elantra', 'Tucson'],
    'Honda': ['City', 'Amaze', 'Jazz', 'WR-V', 'CR-V', 'Civic', 'Accord', 'BR-V'],
    'Toyota': ['Innova', 'Fortuner', 'Etios', 'Corolla', 'Camry', 'Yaris', 'Glanza'],
    'Tata': ['Nexon', 'Harrier', 'Safari', 'Tiago', 'Tigor', 'Hexa', 'Zest', 'Indica'],
    'Mahindra': ['XUV500', 'Scorpio', 'Bolero', 'KUV100', 'TUV300', 'XUV300', 'Thar'],
    'Ford': ['EcoSport', 'Endeavour', 'Figo', 'Aspire', 'Freestyle', 'Mustang'],
    'Volkswagen': ['Polo', 'Vento', 'Ameo', 'Tiguan', 'Passat'],
    'Skoda': ['Rapid', 'Octavia', 'Superb', 'Kodiaq', 'Fabia'],
    'Renault': ['Duster', 'Kwid', 'Captur', 'Lodgy', 'Fluence'],
    'Nissan': ['Terrano', 'Micra', 'Sunny', 'X-Trail', 'GT-R'],
    'BMW': ['3 Series', '5 Series', 'X1', 'X3', 'X5', '7 Series'],
    'Mercedes-Benz': ['C-Class', 'E-Class', 'S-Class', 'GLA', 'GLC', 'GLS'],
    'Audi': ['A3', 'A4', 'A6', 'Q3', 'Q5', 'Q7'],
    'Kia': ['Seltos', 'Sonet', 'Carnival'],
    'MG': ['Hector', 'ZS EV', 'Gloster']
}

# Brand pricing tiers (base multiplier for present price)
brand_tiers = {
    'Maruti Suzuki': 1.0, 'Hyundai': 1.1, 'Honda': 1.2, 'Toyota': 1.3, 'Tata': 0.9,
    'Mahindra': 1.0, 'Ford': 1.1, 'Volkswagen': 1.2, 'Skoda': 1.2, 'Renault': 0.95,
    'Nissan': 1.0, 'BMW': 2.5, 'Mercedes-Benz': 2.8, 'Audi': 2.6, 'Kia': 1.1, 'MG': 1.2
}

def generate_car_dataset(n_samples=2000):
    """Generate a realistic car dataset"""
    
    data = []
    
    for _ in range(n_samples):
        # Select random brand and model
        brand = random.choice(list(car_brands_models.keys()))
        model = random.choice(car_brands_models[brand])
        car_name = f"{brand} {model}"
        
        # Generate year (2003-2020)
        year = random.randint(2003, 2020)
        age = 2024 - year
        
        # Base present price based on car segment and brand
        if brand in ['BMW', 'Mercedes-Benz', 'Audi']:
            base_price = random.uniform(25, 80)  # Luxury cars
        elif brand in ['Honda', 'Toyota', 'Volkswagen', 'Skoda']:
            base_price = random.uniform(8, 25)   # Premium cars
        elif model in ['Innova', 'Fortuner', 'XUV500', 'Scorpio', 'Creta', 'Harrier']:
            base_price = random.uniform(12, 30)  # SUVs/MUVs
        else:
            base_price = random.uniform(4, 15)   # Regular cars
        
        # Apply brand tier multiplier
        present_price = base_price * brand_tiers.get(brand, 1.0)
        present_price = round(present_price, 1)
        
        # Generate fuel type based on brand and model
        if brand in ['BMW', 'Mercedes-Benz', 'Audi']:
            fuel_type = random.choice(['Petrol', 'Diesel'], p=[0.6, 0.4])
        elif model in ['Innova', 'Fortuner', 'XUV500', 'Scorpio', 'Duster']:
            fuel_type = random.choice(['Petrol', 'Diesel'], p=[0.3, 0.7])
        else:
            fuel_type = random.choice(['Petrol', 'Diesel', 'CNG'], p=[0.65, 0.3, 0.05])
        
        # Generate transmission type
        if brand in ['BMW', 'Mercedes-Benz', 'Audi']:
            transmission = random.choice(['Manual', 'Automatic'], p=[0.2, 0.8])
        elif year >= 2015:
            transmission = random.choice(['Manual', 'Automatic'], p=[0.7, 0.3])
        else:
            transmission = random.choice(['Manual', 'Automatic'], p=[0.85, 0.15])
        
        # Generate seller type
        seller_type = random.choice(['Individual', 'Dealer'], p=[0.4, 0.6])
        
        # Generate owner count
        if age <= 3:
            owner = random.choice([0, 1], p=[0.6, 0.4])
        elif age <= 7:
            owner = random.choice([0, 1, 2], p=[0.2, 0.6, 0.2])
        else:
            owner = random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
        
        # Generate kilometers driven based on age and owner
        base_km_per_year = random.uniform(8000, 15000)
        km_multiplier = 1 + (owner * 0.2)  # More owners = more km
        kms_driven = int(age * base_km_per_year * km_multiplier)
        kms_driven = max(5000, min(kms_driven, 300000))  # Realistic bounds
        
        # Calculate selling price with realistic depreciation
        # Base depreciation rate
        annual_depreciation = 0.12 if brand not in ['BMW', 'Mercedes-Benz', 'Audi'] else 0.15
        
        # Calculate depreciated value
        selling_price = present_price * ((1 - annual_depreciation) ** age)
        
        # Apply adjustments
        # Fuel type adjustment
        if fuel_type == 'Diesel':
            selling_price *= 1.05
        elif fuel_type == 'CNG':
            selling_price *= 0.95
        
        # Transmission adjustment
        if transmission == 'Automatic':
            selling_price *= 1.03
        
        # Seller type adjustment
        if seller_type == 'Dealer':
            selling_price *= 1.02
        
        # Owner adjustment
        selling_price *= (1 - owner * 0.03)
        
        # Kilometers adjustment
        km_factor = 1 - (kms_driven / 500000)  # Gradual reduction
        selling_price *= km_factor
        
        # Add some random variation
        selling_price *= random.uniform(0.9, 1.1)
        
        # Ensure minimum price
        selling_price = max(0.5, selling_price)
        selling_price = round(selling_price, 2)
        
        data.append({
            'Car_Name': car_name,
            'Year': year,
            'Selling_Price': selling_price,
            'Present_Price': present_price,
            'Kms_Driven': kms_driven,
            'Fuel_Type': fuel_type,
            'Seller_Type': seller_type,
            'Transmission': transmission,
            'Owner': owner
        })
    
    return pd.DataFrame(data)

def add_specific_popular_cars(df):
    """Add some specific popular car entries with known patterns"""
    
    popular_cars = [
        # Maruti Suzuki Swift entries
        {'Car_Name': 'Maruti Suzuki Swift', 'Year': 2018, 'Present_Price': 7.5, 'Kms_Driven': 45000, 
         'Fuel_Type': 'Petrol', 'Seller_Type': 'Individual', 'Transmission': 'Manual', 'Owner': 1},
        {'Car_Name': 'Maruti Suzuki Swift', 'Year': 2015, 'Present_Price': 7.5, 'Kms_Driven': 75000, 
         'Fuel_Type': 'Diesel', 'Seller_Type': 'Dealer', 'Transmission': 'Manual', 'Owner': 1},
        
        # Honda City entries
        {'Car_Name': 'Honda City', 'Year': 2017, 'Present_Price': 12.0, 'Kms_Driven': 50000, 
         'Fuel_Type': 'Petrol', 'Seller_Type': 'Individual', 'Transmission': 'Manual', 'Owner': 1},
        {'Car_Name': 'Honda City', 'Year': 2019, 'Present_Price': 12.0, 'Kms_Driven': 30000, 
         'Fuel_Type': 'Petrol', 'Seller_Type': 'Dealer', 'Transmission': 'Automatic', 'Owner': 0},
        
        # Toyota Innova entries
        {'Car_Name': 'Toyota Innova', 'Year': 2016, 'Present_Price': 18.0, 'Kms_Driven': 80000, 
         'Fuel_Type': 'Diesel', 'Seller_Type': 'Individual', 'Transmission': 'Manual', 'Owner': 1},
        {'Car_Name': 'Toyota Innova', 'Year': 2014, 'Present_Price': 18.0, 'Kms_Driven': 120000, 
         'Fuel_Type': 'Diesel', 'Seller_Type': 'Dealer', 'Transmission': 'Manual', 'Owner': 2},
        
        # Hyundai Creta entries
        {'Car_Name': 'Hyundai Creta', 'Year': 2018, 'Present_Price': 14.0, 'Kms_Driven': 40000, 
         'Fuel_Type': 'Petrol', 'Seller_Type': 'Dealer', 'Transmission': 'Manual', 'Owner': 1},
        {'Car_Name': 'Hyundai Creta', 'Year': 2019, 'Present_Price': 14.0, 'Kms_Driven': 25000, 
         'Fuel_Type': 'Diesel', 'Seller_Type': 'Individual', 'Transmission': 'Automatic', 'Owner': 0},
    ]
    
    # Calculate selling prices for these cars
    for car in popular_cars:
        age = 2024 - car['Year']
        annual_depreciation = 0.12
        selling_price = car['Present_Price'] * ((1 - annual_depreciation) ** age)
        
        # Apply same adjustments as main function
        if car['Fuel_Type'] == 'Diesel':
            selling_price *= 1.05
        elif car['Fuel_Type'] == 'CNG':
            selling_price *= 0.95
        
        if car['Transmission'] == 'Automatic':
            selling_price *= 1.03
        
        if car['Seller_Type'] == 'Dealer':
            selling_price *= 1.02
        
        selling_price *= (1 - car['Owner'] * 0.03)
        km_factor = 1 - (car['Kms_Driven'] / 500000)
        selling_price *= km_factor
        
        car['Selling_Price'] = round(max(0.5, selling_price), 2)
    
    # Add to dataframe
    popular_df = pd.DataFrame(popular_cars)
    return pd.concat([df, popular_df], ignore_index=True)

if __name__ == "__main__":
    print("Generating realistic used car dataset...")
    
    # Generate main dataset
    df = generate_car_dataset(2000)
    
    # Add specific popular car entries
    df = add_specific_popular_cars(df)
    
    # Shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Save to CSV
    df.to_csv('car_data.csv', index=False)
    
    print(f"Dataset created with {len(df)} entries")
    print("\nDataset Summary:")
    print(f"Years: {df['Year'].min()} - {df['Year'].max()}")
    print(f"Price range: ₹{df['Selling_Price'].min()} - ₹{df['Selling_Price'].max()} Lakhs")
    print(f"Fuel types: {df['Fuel_Type'].value_counts().to_dict()}")
    print(f"Transmission: {df['Transmission'].value_counts().to_dict()}")
    print(f"Seller types: {df['Seller_Type'].value_counts().to_dict()}")
    
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nDataset saved as 'car_data.csv'")