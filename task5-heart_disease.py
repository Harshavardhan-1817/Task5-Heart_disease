# Decision Trees & Random Forests

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#Load dataset
df = pd.read_csv("heart.csv")  # Use heart.csv or your dataset
print("✅ Dataset Loaded")
print(df.head())

# Split features and target
X = df.drop('target', axis=1)
y = df['target']

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree Classifier
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)

# Visualize tree
plt.figure(figsize=(20,10))
plot_tree(tree_clf, feature_names=X.columns, class_names=['No Disease', 'Disease'], filled=True)
plt.title("Decision Tree")
plt.savefig("decision_tree.png")
print("📊 Decision tree saved as 'decision_tree.png'")

#3. Control tree depth to avoid overfitting
tree_clf_pruned = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_clf_pruned.fit(X_train, y_train)

# Accuracy
y_pred = tree_clf.predict(X_test)
y_pred_pruned = tree_clf_pruned.predict(X_test)

print("\n🌳 Decision Tree Accuracy (no depth limit):", accuracy_score(y_test, y_pred))
print("🌳 Pruned Decision Tree Accuracy (max_depth=3):", accuracy_score(y_test, y_pred_pruned))

#  Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)

print("🌲 Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

# ✅ 5. Feature Importances
importances = pd.Series(rf_clf.feature_importances_, index=X.columns)
print("\n🔑 Feature Importances:")
print(importances.sort_values(ascending=False))

# Plot feature importances
importances.sort_values(ascending=True).plot(kind='barh')
plt.title("Feature Importances - Random Forest")
plt.tight_layout()
plt.savefig("feature_importances.png")
print("📊 Feature importances plot saved as 'feature_importances.png'")

# Cross-validation
cv_scores_tree = cross_val_score(tree_clf_pruned, X, y, cv=5)
cv_scores_rf = cross_val_score(rf_clf, X, y, cv=5)

print("\n📈 Cross-validation Accuracy (Decision Tree):", np.mean(cv_scores_tree))
print("📈 Cross-validation Accuracy (Random Forest):", np.mean(cv_scores_rf))

input("\n✅ Task finished. Press Enter to exit...")
