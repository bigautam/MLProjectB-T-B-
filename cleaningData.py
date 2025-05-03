import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


df = pd.read_csv("/mnt/c/Users/Bethl/MLProjectB-T-B-/PS_20174392719_1491204439457_log.csv")
# print(df.head(5))

#cleaning up the step so that they are in a 24hr interval instead of up to 744
df["step"] = df["step"] % 24
# Convert any 0s (which come from multiples of 24) to 24
df.loc[df["step"] == 0, "step"] = 24

costomer_df = df[df["nameDest"].str.startswith("C")]

max_size = costomer_df['isFraud'].value_counts().max()
print(max_size)
# Balance the target label by upsampling minority class
lst = [costomer_df]
for class_index, group in costomer_df.groupby('isFraud'):
    lst.append(group.sample(max_size, replace=True))

# Combine into a balanced DataFrame
costomer_df = pd.concat(lst).reset_index(drop=True)

# Now sample 10,000 rows for modeling
sample_df = costomer_df.sample(n=10000, random_state=42)


sample_df1 =sample_df.copy()
X=sample_df1.drop('isFraud',axis=1)
y=sample_df1['isFraud']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=111)

#Standardizing the numerical columns
col_names=['amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest']
features_train = X_train[col_names]
features_test = X_test[col_names]
scaler = StandardScaler().fit(features_train.values)
features_train = scaler.transform(features_train.values)
features_test = scaler.transform(features_test.values)
X_train[col_names] = features_train
X_test[col_names] =features_test

X_train=X_train.drop(['nameOrig','nameDest','isFlaggedFraud', 'type'],axis=1)
X_train = X_train.reset_index(drop=True)

X_test=X_test.drop(['nameOrig','nameDest','isFlaggedFraud',"type"],axis=1)
X_test = X_test.reset_index(drop=True)

clf = DecisionTreeClassifier(max_depth=4, random_state=42)

clf.fit(X_train, y_train)

# plot for but maybe itll look good if its vs code opened on jupter have no idea tho :>
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print(sample_df1["isFraud"].value_counts(normalize=True))


plt.figure(figsize=(20, 10)) 
plot_tree(clf, 
          feature_names=X_train.columns, 
          class_names=["Not Fraud", "Fraud"], 
          filled=True, 
          rounded=True)
plt.title("Decision Tree for Fraud Detection")
plt.show()