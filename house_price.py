import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample dataset
data = {
    'Area': [800, 1000, 1200, 1500, 1800],
    'Bedrooms': [2, 2, 3, 3, 4],
    'Price': [4000000, 5000000, 6000000, 7500000, 9000000]
}

df = pd.DataFrame(data)

X = df[['Area', 'Bedrooms']]
y = df['Price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

print("Predicted Prices:", pred)
print("MSE:", mean_squared_error(y_test, pred))

# Custom prediction
area = int(input("Enter area: "))
bed = int(input("Enter bedrooms: "))

result = model.predict([[area, bed]])
print("Estimated House Price:", result[0])
