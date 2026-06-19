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
np.random.seed(42)

df['soil_moisture'] = (df['rainfall'] * 0.4 + df['humidity'] * 0.3 + np.random.normal(0, 5, len(df))).clip(0, 100)
df['temperature']  += np.random.normal(0, 1.2, len(df))
df['humidity']     += np.random.normal(0, 2.5, len(df))
df['rainfall']     += np.random.normal(0, 1.5, len(df)).clip(0)


def assign_risk(row):
    temp     = row['temperature']
    temp_max = row['temp_max']
    temp_min = row['temp_min']
    humidity = row['humidity']
    rainfall = row['rainfall']
    soil     = row['soil_moisture']
    cloud    = row['cloud_cover']
    r = np.random.random()

    if rainfall > 55:
        return 'Flood'       if r < 0.87 else 'Normal'
    elif temp_max >= 40 and humidity < 35 and rainfall < 5:
        return 'Heatwave'    if r < 0.85 else 'Drought'
    elif temp_max >= 36 and rainfall < 5 and humidity < 48:
        return 'Drought'     if r < 0.84 else 'Normal'
    elif temp_min <= 8 and cloud < 20:
        return 'Frost'       if r < 0.82 else 'Normal'
    elif humidity > 85 and rainfall > 10 and temp > 25:
        return 'CropDisease' if r < 0.80 else 'Flood'
    elif soil < 20 and rainfall < 3 and humidity < 45:
        return 'LowYield'    if r < 0.78 else 'Drought'
    else:
        return 'Normal'      if r < 0.88 else np.random.choice(['Drought','Heatwave','LowYield'])

df['risk'] = df.apply(assign_risk, axis=1)


features = ['temperature', 'humidity', 'rainfall', 'soil_moisture', 'wind_speed', 'cloud_cover']
X, y = df[features], df['risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=150, random_state=42)
rf.fit(X_train, y_train)
print(f"Climate Model Accuracy: {accuracy_score(y_test, rf.predict(X_test))*100:.2f}%")
print(classification_report(y_test, rf.predict(X_test)))


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