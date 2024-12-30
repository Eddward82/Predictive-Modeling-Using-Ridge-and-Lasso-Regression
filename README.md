
# Predictive-Modeling-Using-Ridge-and-Lasso-Regression-Techniques
In this analysis, I demonstrated the data regression skills I  have learned in the course of completing the machine learning course. I leveraged a wide variety of tools, but also this report focuses on present findings, insights, and next steps. I included some visuals from your code output, but this report is intended as a summary of my findings, not as a code review.
## Importing the required libraries

To suppress warnings
```
def warn (*args, **kwargs):
    pass
import warnings
warnings.warn = warn
```
To import necessary libraries
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
```

## Importing the dataset
The data is technical specs of cars. The dataset is downloaded from UCI Machine Learning Repository at http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg. The attributes of the data include cylinders, displacement, horsepower, weight, acceleration, model year, origin, and mpg. All the attribute types are continous except for model year and origin, which are multi-valued discretes. All the variables are independent except for the mpg, which is a dependent/target variable.

```
data = pd.read_csv("auto-mpg.csv")
data.head()
```
## Data Exploration
```
data.info
```
![image](https://github.com/user-attachments/assets/b756b8bc-d4ac-4605-b30b-f77c70121f7a)

```
data.describe()
```
![image](https://github.com/user-attachments/assets/b1bf2f33-d31c-4980-8851-f6b0381deed9)

## Checking for missing data
```
data.isnull().sum()
```
![image](https://github.com/user-attachments/assets/9b4dab6c-94b8-450e-83d9-bf3bd1a1ef6a)

## To visualise the relationships amongst the data
The relationships between the independent and dependent variable are visualised usng sns.pairplot as shown below
```
sns.pairplot(data, x_vars=['displacement', 'horsepower', 'weight', 'acceleration'], y_vars='mpg')
```

![image](https://github.com/user-attachments/assets/ef7179a8-bdc6-4272-8860-7d002925da00)

## Objective
The objective of this project is to predict a car's fuel efficiency (mpg) using features such as engine displacement, horsepower, weight, and acceleration. The analysis aims to identify the most effective linear regression model to achieve this goal.
### To view the first few rows of the data
```
data.head()
```
![image](https://github.com/user-attachments/assets/d6a27719-9725-4b8a-b684-e0f83dc45441)

### Transforming categorical data to numerical value
One-hot encoding was used to transform categorical data into numerical values so that machine learning models can use it.
- Encoding categorical variables (origin). Origin values represent regions:
- 1 = USA, 2 = Europe, and 3 = Asia
```
data = pd.get_dummies(data, columns=['origin'], drop_first=True)
```
### To view the data to check the columns
As shown in the result below the numerical variables have been created from the origin categorical variable.
```
print(data.columns)
```
![image](https://github.com/user-attachments/assets/14395bdf-26f8-4643-bbc0-4e7f6d355264)

### Feature-Target Split
The syntax below creates the feature-target split for the data.
```
X = data [['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration','model year', 'origin_2', 'origin_3']]
y = data['mpg']
```
## Linear Regression Models
### Data Spliting
The data is split into test and train sections. While the test_size is 20%, and the train size is 80%
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
```
X_test
```

![image](https://github.com/user-attachments/assets/424d4c6d-0167-4793-9861-6ce90cd3694e)

### To check the data types and the data uniqueness
As shown, the horsepower variable has an object datatype but must be changed to float, and also carries a non-numerical attribute '?'. This should be removed.
```
print(data.dtypes)
print(data['horsepower'].unique())
```
![image](https://github.com/user-attachments/assets/1429721a-315e-4ac7-b75c-d6614ac7ac14)

### Data types changed from object to numeric
The non-numeric attribute '?' was removed and replaced with 'NAN', and the datatype changed to numeric.
```
data.replace('?', np.nan, inplace=True)
data = data.astype(float, errors='ignore')
print(data.dtypes)
```
![image](https://github.com/user-attachments/assets/75bb2ad6-dbd0-4979-bc79-aa05ea70f974)

### Missing values checked again
Since the nonnumeric object '?' was changed to NAN, this made the horsepower variable to have missing values as shown
```
print(data.isnull().sum())
print(data.describe())
```
![image](https://github.com/user-attachments/assets/76dcbfd9-c162-4d22-bdbe-a489f8258d8f)

### The missing values are replaced
The missing values were filled with the mean of the horsepower column using the syntax below:
```
data['horsepower'].fillna(data['horsepower'].mean(), inplace=True)
print(data.isnull().sum())
```

![image](https://github.com/user-attachments/assets/e101bb6e-338b-4a7c-97d6-4e1d751afc1e)

### The data is checked again
```
data.info()
```
![image](https://github.com/user-attachments/assets/6bd2a1a7-62db-4899-8ec0-89955991e1ba)

### Data Split
The data is split again 
```
X = data [['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration','model year', 'origin_2', 'origin_3']]
y = data['mpg']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Data Training and Evaluation

```
Ir = LinearRegression()
Ir.fit(X_train, y_train)
y_pred = Ir.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
```

![image](https://github.com/user-attachments/assets/90ef097b-f39d-4fdd-8ed9-d7060c202742)

### Ridge and Lasso Regression
#### Train Ridge

```
ridge = Ridge(alpha=1.0)
ridge.fit(X_train,y_train)
```
![image](https://github.com/user-attachments/assets/e377ac65-dfe1-465f-83b4-6c7d3ce417fa)

```
y_pred_ridge = ridge.predict(X_test)
print("MSE (Ridge):", mean_squared_error(y_test, y_pred_ridge))
print("R2 Score (Ridge):", r2_score(y_test, y_pred_ridge))
```

![image](https://github.com/user-attachments/assets/5529db19-0266-4903-a55a-b51265893d11)

#### Train Lasso
```
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
```

![image](https://github.com/user-attachments/assets/43e6eb98-86ae-4940-97bc-f789d11acb8b)

```
y_pred_lasso = lasso.predict(X_test)
print("MSE (Lasso):", mean_squared_error(y_test, y_pred_lasso))
print("R2 Score (Lasso):", r2_score(y_test, y_pred_lasso))
```

![image](https://github.com/user-attachments/assets/77107db5-de97-4260-a9ec-ccac17e3bb0e)


#### Visualising Results

```
plt.scatter(y_test, y_pred, label="Linear Regression", alpha=0.5)
plt.xlabel("Actual MPG")
plt.ylabel("Predicted MPG")
plt.title("Linear Regression: Actual vs. Predicted MPG")
plt.legend ()
plt.show()
```

![image](https://github.com/user-attachments/assets/0106bf44-1356-452a-942e-9de82ceee7ba)

### Insights and key findings

#### Linear Regression
- Mean Squared Error (MSE): 8.339142508255916
- R2 Score = 0.8449006123776615
- Analysis: Linear regression provides good performance with a relatively low MSE and a high R2 score (~0.844), indicating the model explains 84.4% of the variance in the target variable.

#### Ridge Regression (alpha=0)
- Mean Squared Error (MSE): 8.34606804606358
- R2 Score: 0.8447729434475317
- Analysis: Ridge regression performs very similarly to linear regression, with only a slight increase in MSE and a negligible decrease in R2 score. Regularization does not significantly improve the model here, suggesting minimal multicollinearity among features.

#### Lasso Regression (alpha = 0.1)
- Mean Squared Error (MSE): 8.784105610924883
- R2 Score: 0.8366247607565661
- Analysis: Lasso regression has a slightly higher MSE and a slightly lower R2 score (~0.837). The feature selection performed by Lasso might have excluded some variables important for prediction, leading to a small drop in performance.

### Key Observations
- Linear Regression has the best performance, with the lowest MSE and the highest R2 score
- Ridge Regression produces almost identical results to linear regression, indicating the regularisation was not critical for this dataset.
- Lasso Regression shows slightly reduced performance, potentially due to its feature selection mechanism, which might have excluded some relevant variables.

### Recommendations
- If simplicity and interpretability are important, linear regression is sufficient given its superior performance
- If overfitting or multicollinearity were concerns, ridge regression would be a good choice with comparable performance.
- Lasso regression could be useful in scenarios where reducing the number of features for a simpler model is a priority, despite its slightly reduced accuracy.
