"""RBF feature mapping followed by a classifier."""
from sklearn.datasets import make_classification
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=1000, n_features=8, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
model = make_pipeline(StandardScaler(), RBFSampler(gamma=1.0, n_components=500, random_state=42), LogisticRegression(max_iter=2000))
model.fit(X_train, y_train)
print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))
