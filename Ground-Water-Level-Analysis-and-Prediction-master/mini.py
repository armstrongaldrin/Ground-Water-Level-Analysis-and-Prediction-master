import pandas as pd
import seaborn as sb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# Load the dataset
data = pd.read_csv("E:\projects\Ground-Water-Level-Analysis-and-Prediction-master\Ground-Water-Level-Analysis-and-Prediction-master\data.csv")

# Visualize the target variable distribution
sb.countplot(x='Situation', data=data, palette='bright')

# Encode the 'Situation' column
Availabilty = pd.get_dummies(data['Situation'], drop_first=True)
data.drop(['Situation'], axis=1, inplace=True)
data1 = pd.concat([data, Availabilty], axis=1)

# Extract features and target variables
X = data1.iloc[:, [4, 5, 6, 9, 10, 11]]
Y = data1.iloc[:, 12]

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.55, random_state=0)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train the Logistic Regression model
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, Y_train)

# Predict the test set results
y_pred = classifier.predict(X_test)

# Evaluate the model
cm = confusion_matrix(Y_test, y_pred)
new_score = accuracy_score(Y_test, y_pred)

# Output the confusion matrix and accuracy score
print("Confusion Matrix:\n", cm)
print("Accuracy Score:", new_score)
