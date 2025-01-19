# Credit Card Fraud Detection
This project analyzes a credit card fraud dataset using Python

# Steps

# Step 1: Import Libraries

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```
# Step 2: Load the Dataset
```
data = pd.read_csv('Creditcard_data.csv')
```
# Step 3: Initial Exploration of the Dataset
```
data.head()
data.info()
data.describe()
```
data.head(): To display the first 5 rows of the dataset.

data.info(): Provides metadata like column names, data types, and non-null counts.

data.describe(): Provides descriptive analysis of the columns like mean,median etc .

# Step 4: Class Distribution Analysis
```
data["Class"].value_counts()
```
Count the number of instances having target variable (class) as class == 1 or class == 0.

# Step 5: Missing Values Check
```
missing_values = data.isnull().sum()
print("Missing Values in Each Column:")
print(missing_values)
```
To check whether there is any missing value or not in any of the columns.

# Step 6: Separate Classes
```
data_0 = data[data['Class'] == 0]
data_1 = data[data['Class'] == 1]
print('class 0:', data_0.shape)
print('class 1:', data_1.shape)
```
Divides the dataset into two subsets:

data_0: Transactions labeled as class 0 (non-fraud).

data_1: Transactions labeled as class 1 (fraud).

# Step 7: Visualize Target Variable
```
data['Class'].value_counts().plot(kind='bar', color='skyblue', title="Target Variable Distribution")
```
Creates a bar graph to visualize the imbalance between class 0 and class 1.

# Step 8: Handle Imbalanced Dataset with SMOTE
```
from imblearn.over_sampling import SMOTE
from collections import Counter

y = data['Class']
x = data.drop('Class', axis=1)

smote = SMOTE()
x_smote, y_smote = smote.fit_resample(x, y)
```
SMOTE (Synthetic Minority Oversampling Technique): Balances the dataset by creating synthetic samples for the minority class.

x: Independent features.

y: Target variable.

fit_resample: Applies SMOTE to create x_smote and y_smote, balanced datasets.

# Step 9: Recombining to single dataset
```
import pandas as pd

balanced_data = pd.concat([pd.DataFrame(x_smote), pd.DataFrame(y_smote, columns=['Class'])], axis=1)

print(balanced_data.head())
print(balanced_data['Class'].value_counts())
```

# Step 10: Create Samples

 ```
from sklearn.model_selection import train_test_split
import numpy as np

# 1. Simple Random Sampling
sample1 = balanced_data.iloc[np.random.choice(len(balanced_data), size=int(0.2 * len(balanced_data)), replace=False)]

# 2. Stratified Sampling
strata = balanced_data.groupby('Class')
sample2 = strata.apply(lambda x: x.sample(int(0.2 * len(x)), random_state=2)).reset_index(drop=True)

# 3. Systematic Sampling
k = len(balanced_data) // int(0.2 * len(balanced_data))
start = np.random.randint(0, k)
sample3 = balanced_data.iloc[start::k]

# 4. Cluster Sampling
num_clusters = 5
cluster_labels = np.arange(len(balanced_data)) % num_clusters
balanced_data['Cluster'] = cluster_labels
selected_cluster = np.random.choice(num_clusters)
sample4 = balanced_data[balanced_data['Cluster'] == selected_cluster].drop('Cluster', axis=1)

# 5. Bootstrapping
sample5 = balanced_data.iloc[np.random.choice(len(balanced_data), size=int(0.2 * len(balanced_data)), replace=True)]

# Print the lengths of the samples
print(len(sample1), len(sample2), len(sample3), len(sample4), len(sample5))
```
Multiple sampling techniques are applied to the balanced dataset to extract samples:
1. Simple Random Sampling: Selects a random subset of the data without replacement.
2. Stratified Sampling: Ensures that the class distribution is maintained in the sample.
3. Systematic Sampling: Selects samples based on a fixed interval k.
4. Cluster Sampling: Divides the dataset into clusters and selects one cluster.
5. Bootstrapping: Samples data points with replacement.

# Step 11: Importing Libraries
```
# importing libraries
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
```
# Step 12: Defining the Models to be used 
```
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
    "SVM": SVC(),
    "k-NN": KNeighborsClassifier()
}
```
# Step 13: Creating a result matrix
```
results = {}
samples = [sample1, sample2, sample3, sample4, sample5]
```
# Step 14: Train and evaluate various models
```

from sklearn.metrics import accuracy_score

for model_name, model in models.items():
    results[model_name] = []
    for i, sample in enumerate(samples):

        X_sample = sample.drop('Class', axis=1)
        y_sample = sample['Class']

        X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        results[model_name].append(accuracy)


results_df = pd.DataFrame(results, index=["Sample1", "Sample2", "Sample3", "Sample4", "Sample5"])
print(results_df)

results_df.to_csv("model_accuracies.csv")
```
# Step 15: Storing the data in .csv file 
```
results_df = pd.DataFrame(results, index=["Sample1", "Sample2", "Sample3", "Sample4", "Sample5"])
print(results_df)

results_df.to_csv('Submission_Abhiroop.csv')
```
# Best Model for each sample 
# Sample 1: Gradient Boosting 
# Sample 2: Gradient Boosting 
# Sample 3: Logistic Regression
# Sample 4: Decision Tree / Gradient Boosting 
# Sample 5: Gradient Boosting (overfits) / both logistic Regression and Decision Tree
