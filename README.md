# Covid19-India-Report
import pandas as pd
import io
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression

#reading data from csv
from google.colab import files
uploaded= files.upload()
df= pd.read_csv(io.BytesIO(uploaded['covid_india.csv']))

#plot for cured and confirmed patients in india
plt.style.use("Solarize_Light2")
df.plot(x='Cured', y='Confirmed', style='o')  
plt.title("Cured vs Confirmed")  
plt.xlabel('Cured')  
plt.ylabel('Confirmed')  
plt.show()

#train the data with linear regression algorithm
X = df['Cured'].values.reshape(-1,1)
y = df['Confirmed'].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=0)
regressor = LinearRegression()  
regressor.fit(X_train, y_train) #training the algorithm

#predicting dat after training it
y_pred = regressor.predict(X_test)

#printing the actual and predicted data 
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(df)

#comparing the acutal data and the predictable data in one plot
df1 = df.tail(50)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
