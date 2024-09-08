import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("E:\projects\Ground-Water-Level-Analysis-and-Prediction-master\Ground-Water-Level-Analysis-and-Prediction-master\data.csv")

# Check for null values
sb.heatmap(data.isnull())
plt.show()

# Count plot for 'Situation' column
sb.countplot(x='Situation', data=data)
plt.show()

# Print info about the data
data.info()

# Encode the 'Situation' column and drop it from the dataset
Availabilty = pd.get_dummies(data['Situation'], drop_first=True)
data.drop(['Situation'], axis=1, inplace=True)
data1 = pd.concat([data, Availabilty], axis=1)

# Correlation heatmap
sb.heatmap(data1.corr(), annot=True, cmap="coolwarm")
plt.show()

# Plot time series data
data1["Total_Rainfall"].plot(kind='line')
data1["Net annual groundwater availability"].plot()
data1["Total_Usage"].plot()
plt.legend(['Total_Rainfall', 'Net annual groundwater availability', 'Total_Usage'])
plt.show()

# Pie chart
labels = ['Total Rainfall', 'Net Annual GroundWater', 'Total Use', 'Future Available', 'Projected demand for domestic and industrial uses upto 2025', 'Natural discharge during non-monsoon season']
sizes = [14.84, 13.64, 8.39, 5.29, 1.063483, 1.210483]
cols = ['c', 'm', 'r', 'b', 'g', 'y']
plt.pie(sizes, labels=labels, colors=cols, startangle=90, shadow=True, explode=(0,0.01,0.01,0.01,0.1,0.2), autopct='%1.1f%%')
plt.show()

# Pair plot
sb.pairplot(data1, x_vars=['Groundwater availability for future irrigation use'], y_vars=['Net annual groundwater availability'], kind='scatter', diag_kind='hist', height=6.0)
plt.show()
