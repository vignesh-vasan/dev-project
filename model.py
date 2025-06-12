import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv('Housing.csv')

# Select relevant features
X = df[['area', 'bedrooms', 'bathrooms']]
y = df['price']

# Fill missing values if any
X.fillna(X.mean(), inplace=True)
y.fillna(y.mean(), inplace=True)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))

# Example prediction
new_house = [[2500, 3, 2]]  # area=2500, 3 bedrooms, 2 bathrooms
predicted_price = model.predict(new_house)
print("Predicted Price for new house:", predicted_price)
