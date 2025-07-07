import csv
import random

# Set random seed for reproducibility
random.seed(42)

def create_car_dataset():
    """Create a simple car dataset without external dependencies"""
    
    # Car brands and models
    cars = [
        "Maruti Suzuki Swift", "Maruti Suzuki Alto", "Maruti Suzuki Baleno",
        "Hyundai i20", "Hyundai Creta", "Hyundai Verna",
        "Honda City", "Honda Amaze", "Honda Jazz",
        "Toyota Innova", "Toyota Fortuner", "Toyota Etios",
        "Tata Nexon", "Tata Harrier", "Tata Tiago",
        "Mahindra XUV500", "Mahindra Scorpio", "Mahindra Bolero",
        "Ford EcoSport", "BMW 3 Series", "Mercedes-Benz C-Class"
    ]
    
    fuel_types = ["Petrol", "Diesel", "CNG"]
    seller_types = ["Individual", "Dealer"]
    transmissions = ["Manual", "Automatic"]
    
    data = []
    
    # Create header
    header = ["Car_Name", "Year", "Selling_Price", "Present_Price", "Kms_Driven", 
              "Fuel_Type", "Seller_Type", "Transmission", "Owner"]
    data.append(header)
    
    # Generate 1000 records
    for i in range(1000):
        car_name = random.choice(cars)
        year = random.randint(2003, 2020)
        age = 2024 - year
        
        # Base present price based on car type
        if "BMW" in car_name or "Mercedes" in car_name:
            present_price = round(random.uniform(25, 80), 1)
        elif "Toyota" in car_name or "Honda" in car_name:
            present_price = round(random.uniform(8, 25), 1)
        else:
            present_price = round(random.uniform(4, 15), 1)
        
        # Generate other attributes
        kms_driven = random.randint(10000, 200000)
        fuel_type = random.choice(fuel_types)
        seller_type = random.choice(seller_types)
        transmission = random.choice(transmissions)
        owner = random.randint(0, 3)
        
        # Calculate selling price with depreciation
        annual_depreciation = 0.12
        selling_price = present_price * ((1 - annual_depreciation) ** age)
        
        # Apply adjustments
        if fuel_type == "Diesel":
            selling_price *= 1.05
        elif fuel_type == "CNG":
            selling_price *= 0.95
            
        if transmission == "Automatic":
            selling_price *= 1.03
            
        if seller_type == "Dealer":
            selling_price *= 1.02
            
        selling_price *= (1 - owner * 0.03)
        km_factor = 1 - (kms_driven / 500000)
        selling_price *= km_factor
        
        # Add random variation
        selling_price *= random.uniform(0.9, 1.1)
        selling_price = max(0.5, round(selling_price, 2))
        
        row = [car_name, year, selling_price, present_price, kms_driven,
               fuel_type, seller_type, transmission, owner]
        data.append(row)
    
    return data

def save_to_csv(data, filename):
    """Save data to CSV file"""
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(data)

if __name__ == "__main__":
    print("Generating car dataset...")
    dataset = create_car_dataset()
    save_to_csv(dataset, 'car_data.csv')
    print(f"Dataset created with {len(dataset)-1} records")
    print("Dataset saved as 'car_data.csv'")
    
    # Show first few rows
    print("\nFirst 5 rows:")
    for i, row in enumerate(dataset[:6]):
        if i == 0:
            print("Headers:", ", ".join(row))
        else:
            print(f"Row {i}: {row}")