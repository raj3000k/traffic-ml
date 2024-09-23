import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('traffic.csv')

X = df.drop('current_traffic_light_state', axis=1)
y = df['current_traffic_light_state']

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

def predict_traffic_light(features):
    return model.predict([features])

features_sample = X_test.iloc[0].values
predicted_light = predict_traffic_light(features_sample)

print(predicted_light)
