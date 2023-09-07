import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Loading the Iris dataset from Kaggle using pandas
iris_df = pd.read_csv("C:\\Users\\Rama Krishna\\Desktop\\iris flower\\archive\\IRIS.csv")

# Exploring the dataset and visualize it
print(iris_df.head())
print(iris_df.describe())
sns.pairplot(iris_df, hue='species')
plt.show()

# Preparing the data for training the model
label_encoder = LabelEncoder()
iris_df['species_encoded'] = label_encoder.fit_transform(iris_df['species'])
X = iris_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = iris_df['species_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training a logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Visualizing the results using pie charts, bar charts, and histograms
species_counts = iris_df['species'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(species_counts, labels=species_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Species Distribution')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x='species', y='sepal_length', data=iris_df)
plt.title('Sepal Length by Species')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data=iris_df, x='petal_width', hue='species', element='step', common_norm=False)
plt.title('Petal Width Distribution by Species')
plt.legend(title='Species')
plt.show()
