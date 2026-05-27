import pickle

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

# LOAD DATASET

iris = load_iris()

X = iris.data

y = iris.target

# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MODEL TRAINING

model = RandomForestClassifier(
    # hyperparameters
    n_estimators=100,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
)

model.fit(X_train, y_train)

# PREDICTIONS

y_pred = model.predict(X_test)

# ACCURACY

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# SAVE MODEL


with open("models/random_forest_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully.")
