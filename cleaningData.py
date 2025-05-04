import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# gotta change path per person 

df = pd.read_csv("/mnt/c/Users/sanig/ML/MLProjectB-T-B-/PS_20174392719_1491204439457_log.csv/PS_20174392719_1491204439457_log.csv")

# Step adjustment to 24hr format
df["step"] = df["step"] % 24
df.loc[df["step"] == 0, "step"] = 24

# Filter for customers only
df = df[df["nameDest"].str.startswith("C")]

# Sample 5000 rows FIRST to reduce memory use, then balance
df = df.sample(n=5000, random_state=42)

# Balance dataset by upsampling minority class
fraud = df[df["isFraud"] == 1]
non_fraud = df[df["isFraud"] == 0].sample(n=len(fraud) * 2, random_state=42)  # Downsample non-fraud to avoid massive duplication

balanced_df = pd.concat([fraud, non_fraud], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

# Train/test split
X = balanced_df.drop(columns=['isFraud'])
y = balanced_df['isFraud']

# Drop unused features
X = X.drop(columns=['nameOrig', 'nameDest', 'isFlaggedFraud', 'type'])

# Scale numeric columns
numeric_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=111)

# -------------------------------------- Decision Tree -----------------------------------------------------------

# clf = DecisionTreeClassifier(max_depth=4, random_state=42)

# clf.fit(X_train, y_train)

# # plot for but maybe itll look good if its vs code opened on jupter have no idea tho :>
# y_pred = clf.predict(X_test)
# print(classification_report(y_test, y_pred))
# print(sample_df1["isFraud"].value_counts(normalize=True))


# plt.figure(figsize=(20, 10)) 
# plot_tree(clf, 
#           feature_names=X_train.columns, 
#           class_names=["Not Fraud", "Fraud"], 
#           filled=True, 
#           rounded=True)
# plt.title("Decision Tree for Fraud Detection")
# plt.show()

# -------------------------------------- Decision Tree -----------------------------------------------------------

# -------------------------------------- KNN  -----------------------------------------------------------

# Split again for validation set
X_train_small, X_valid, y_train_small, y_valid = train_test_split(
    X_train, y_train, test_size=0.3, random_state=42
)

ks = list(range(1, 16))  # Reduce range to speed up
accuracies_train = []
accuracies_valid = []

for k in ks:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_small, y_train_small)

    accuracies_train.append(accuracy_score(y_train_small, model.predict(X_train_small)))
    accuracies_valid.append(accuracy_score(y_valid, model.predict(X_valid)))

# Plot K results
plt.figure(figsize=(8, 4))
plt.plot(ks, accuracies_valid, label="Validation Accuracy", marker='o')
plt.plot(ks, accuracies_train, label="Training Accuracy", linestyle='--', marker='x')
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy for Different k Values")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Best k
best_k = ks[np.argmax(accuracies_valid)]
print(f"Best k: {best_k}")

# Final KNN model
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Evaluation
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# -------------------------------------- KNN -----------------------------------------------------------