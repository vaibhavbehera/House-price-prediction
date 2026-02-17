import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

df = pd.read_csv("House Price Prediction Dataset.csv")
dataset = pd.DataFrame(df)

X = dataset[['Area', 'Bedrooms']] #input
y = dataset['Price']  #output

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state= 42)

# Standardizing the dataset
sc = StandardScaler()

#Fitting the data
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

# #Cross validation
# reg = LinearRegression()
# reg.fit(X_train, y_train)
# mse = cross_val_score(reg, X_train, y_train, 
#                       scoring = "mean_squared_error", cv = 10)
# np.mean(mse)
# print("Predicted Prices:", pred)
print("MSE:", mean_squared_error(y_test, pred))

# Custom prediction
area = int(input("Enter area: "))
bed = int(input("Enter bedrooms: "))

result = model.predict([[area, bed]])
print("Estimated House Price:", result[0])
