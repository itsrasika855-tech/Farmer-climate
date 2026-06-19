import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv("Indian_Climate_Dataset_2024_2025.csv")


df = df[['Temperature_Avg (°C)', 'Temperature_Max (°C)', 'Temperature_Min (°C)',
         'Humidity (%)', 'Rainfall (mm)', 'Wind_Speed (km/h)', 'Cloud_Cover (%)']].copy()
df.columns = ['temperature', 'temp_max', 'temp_min', 'humidity', 'rainfall', 'wind_speed', 'cloud_cover']
df.dropna(inplace=True)


df['soil_moisture'] = (df['rainfall'] * 0.4 + df['humidity'] * 0.3 + np.random.normal(0, 5, len(df))).clip(0, 100)
df['temperature']  += np.random.normal(0, 1.2, len(df))
df['humidity']     += np.random.normal(0, 2.5, len(df))
df['rainfall']     += np.random.normal(0, 1.5, len(df)).clip(0)


def assign_risk(row):
    if row['rainfall'] > 55:
        return 'Flood'
    elif row['temp_max'] >= 40 and row['humidity'] < 35:
        return 'Heatwave'
    elif row['temp_max'] >= 36 and row['rainfall'] < 5:
        return 'Drought'
    elif row['temp_min'] <= 8 and row['cloud_cover'] < 20:
        return 'Frost'
    elif row['humidity'] > 85 and row['rainfall'] > 10:
        return 'CropDisease'
    elif row['rainfall'] < 3 and row['humidity'] < 45: 
        return 'LowYield'
    else:
        return 'Normal'

df['risk'] = df.apply(assign_risk, axis=1)


features = ['temperature', 'humidity', 'rainfall', 'soil_moisture', 'wind_speed', 'cloud_cover']
x = df[features]
y=df['risk']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=150, random_state=42)
rf.fit(x_train, y_train)
print(f"Climate Model Accuracy: {accuracy_score(y_test, rf.predict(x_test))*100:.2f}%")
print(classification_report(y_test, rf.predict(x_test)))


print("\n=== Climate Risk Prediction System ===")

temp = float(input("Enter Temperature: "))
humidity = float(input("Enter Humidity: "))
rainfall = float(input("Enter Rainfall: "))
soil = float(input("Enter Soil Moisture: "))
wind = float(input("Enter Wind Speed: "))
cloud = float(input("Enter Cloud Cover: "))

sample = [[temp, humidity, rainfall, soil, wind, cloud]]

risk = rf.predict(sample)[0]


print("\n===== RESULT =====")
print("Predicted Risk:", risk)


if risk == "Drought":
    print("Advice: Use drip irrigation and conserve water.")
elif risk == "Flood":
    print("Advice: Improve drainage and protect crops.")
elif risk == "Heatwave":
    print("Advice: Increase irrigation and use mulching.")
elif risk == "Frost":
    print("Advice: Cover crops and protect seedlings.")
elif risk == "CropDisease":
    print("Advice: Monitor crops and apply disease control.")
elif risk == "LowYield":
    print("Advice: Improve soil moisture and farming practices.")
else:
    print("Advice: No major climate risk detected.")
