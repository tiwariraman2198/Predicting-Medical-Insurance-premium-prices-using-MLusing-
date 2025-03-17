import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv(r"/content/Customer Data.csv")

df.shape
df,info

df.head()
df.tail()

df.duplicated().sum()

#descriptive statistics

df.describe().T

#exploratory data analysis

sns.displot(x='Age',data=df,aspect=10/6,kde=True)

fig,ax=plt.subplots(figsize=(12,6))
sns.lineplot(x=df.Age,y=df.PremiumPrice).set_title('Insurance Premium Price by Age')

pr_lab=['Low','Basic','Average','High','SuperHigh']
df['PremiumLabel']=pr_bins=pd.cut(df['PremiumPrice'],bins=5,labels=pr_lab,precision=0)

fig,ax=plt.subplots(figsize=(12,6))
sns.countplot(x='PremiumLabel', data=df,ax=ax).set_title('Distribution of the Insurance Premium Price')

plot = sns.barplot(data=df, x="Diabetes", y= "PremiumPrice" ).set_title('Insurance Premium Price for Diabetic vs Non-Diabetic Patients')


plot = sns.barplot(data=df, x="BloodPressureProblems", y= "PremiumPrice" ).set_title('Insurance Premium Price for Patients with/without Blood Pressure Problems')

plot = sns.barplot(data=df, x="AnyTransplants", y= "PremiumPrice" ).set_title('Insurance Premium Price for Patients with/without Any Transplants')

plot = sns.barplot(data=df, x="AnyChronicDiseases", y= "PremiumPrice" ).set_title('Insurance Premium Price for Patients with/without Any Chronic Diseases')

plot = sns.barplot(data=df, x="KnownAllergies", y= "PremiumPrice" ).set_title('Insurance Premium Price for Patients with/without Any Known Allergies')

plot = sns.barplot(data=df, x="NumberOfMajorSurgeries", y= "PremiumPrice" ).set_title('Insurance Premium Price for Number Of Major Surgeries by Patients')

#calculating BMI

# Calculating BMI
w = df['Weight'];
h = df['Height'];

#bmi = 10000*(weight/(height*height));

df['BMI'] = 10000*(w/(h*h))

df.head()

fig,ax=plt.subplots(figsize=(12,6))
sns.boxplot(data=df, x="PremiumPrice", y="BMI_Status", hue="BMI_Status", dodge=False).set_title('Insurance Premium Price for Various BMI Status')



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


#linear regression model

scaler = StandardScaler()

df[['Height', 'Weight']] = scaler.fit_transform(df[['Height', 'Weight']])

X = df.drop(columns=['PremiumPrice', 'PremiumLabel','BMI_Status'])
y = df['PremiumPrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Training Mean Squared Error: {train_mse}")
print(f"Training R-squared: {train_r2}")
print(f"Testing Mean Squared Error: {test_mse}")
print(f"Testing R-squared: {test_r2}")


# support vector regressor

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Assuming X and y are already defined datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Support Vector Regression
svr = SVR(kernel='linear')
svr.fit(X_train, y_train)

# Predictions
y_train_pred_svr = svr.predict(X_train)
y_test_pred_svr = svr.predict(X_test)

# Metrics for SVR
train_mse_svr = mean_squared_error(y_train, y_train_pred_svr)
train_r2_svr = r2_score(y_train, y_train_pred_svr)
train_mae_svr = mean_absolute_error(y_train, y_train_pred_svr)

test_mse_svr = mean_squared_error(y_test, y_test_pred_svr)
test_r2_svr = r2_score(y_test, y_test_pred_svr)
test_mae_svr = mean_absolute_error(y_test, y_test_pred_svr)

# Print results
print("\nSupport Vector Regression:")
print(f"Training MSE: {train_mse_svr}, R²: {train_r2_svr}, MAE: {train_mae_svr}")
print(f"Testing MSE: {test_mse_svr}, R²: {test_r2_svr}, MAE: {test_mae_svr}")

# Plot MSE
plt.figure(figsize=(10, 6))

metrics_mse = ['Training MSE', 'Testing MSE']
values_mse = [train_mse_svr, test_mse_svr]
plt.subplot(1, 2, 1)
plt.bar(metrics_mse, values_mse, color=['skyblue', 'orange'])
plt.title('MSE Comparison')
plt.ylabel('MSE')

# Plot R²
metrics_r2 = ['Training R²', 'Testing R²']
values_r2 = [train_r2_svr, test_r2_svr]
plt.subplot(1, 2, 2)
plt.bar(metrics_r2, values_r2, color=['lightgreen', 'salmon'])
plt.title('R² Comparison')
plt.ylabel('R²')

plt.tight_layout()
plt.show()


# gradeint boosting regressor

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Define the features (X) and the target (y)
X = df.drop(columns=['PremiumPrice', 'PremiumLabel','BMI_Status'])
y = df['PremiumPrice']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Gradient Boosting model
gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train)

# Predict the target for the training and test sets
y_train_pred = gbr.predict(X_train)
y_test_pred = gbr.predict(X_test)

# Evaluate the model on the training data
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

# Evaluate the model on the test data
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Training Mean Squared Error: {train_mse}")
print(f"Training R-squared: {train_r2}")
print(f"Testing Mean Squared Error: {test_mse}")
print(f"Testing R-squared: {test_r2}")


#xgboost

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb

X = df.drop(columns=['PremiumPrice', 'PremiumLabel','BMI_Status'])
y = df['PremiumPrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgbr = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=3)
xgbr.fit(X_train, y_train)

y_train_pred = xgbr.predict(X_train)
y_test_pred = xgbr.predict(X_test)

train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Training Mean Squared Error: {train_mse}")
print(f"Training R-squared: {train_r2}")
print(f"Testing Mean Squared Error: {test_mse}")
print(f"Testing R-squared: {test_r2}")


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Creating the DataFrame with the provided data
data = {
    'Model': ['LR', 'SVR', 'GBR', 'XGB'],
    'Training MSE': [14423889.84, 39379659.77, 4348893.89, 4853789.04],
    'Training R2': [0.62, -0.0189, 0.89, 0.87],
    'Testing MSE': [12624515.46, 42649408.64, 6260052.65, 5774758.56],
    'Testing R2': [0.69, -0.0555, 0.84, 0.86]
}

results_df = pd.DataFrame(data)

# Melt the DataFrame to long format for easier plotting
mse_df = results_df.melt(id_vars='Model', value_vars=['Training MSE', 'Testing MSE'],
                         var_name='Metric', value_name='MSE')
r2_df = results_df.melt(id_vars='Model', value_vars=['Training R2', 'Testing R2'],
                        var_name='Metric', value_name='R2')

# Plotting MSE
plt.figure(figsize=(12, 6))
sns.barplot(x='Model', y='MSE', hue='Metric', data=mse_df, palette="Blues_d")
plt.title('MSE Comparison Across Models')
plt.ylabel('Mean Squared Error')
plt.xlabel('Model')
plt.show()

# Plotting R²
plt.figure(figsize=(12, 6))
sns.barplot(x='Model', y='R2', hue='Metric', data=r2_df, palette="Greens_d")
plt.title('R² Comparison Across Models')
plt.ylabel('R² Score')
plt.xlabel('Model')
plt.ylim(-0.1, 1)  # Adjusting the limit to account for negative R² in SVR
plt.show()
