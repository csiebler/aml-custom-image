import mlflow
import mlflow.sklearn

from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Enable mlflow autologging
mlflow.sklearn.autolog()

# Load and split data
iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


with mlflow.start_run():
    # Train model
    svm_model_linear = SVC(kernel='linear', C=1.0).fit(X_train, y_train)

    # Get testing metrics and log
    svm_predictions = svm_model_linear.predict(X_test)
    score = svm_model_linear.score(X_test, y_test)
    mlflow.log_metric("testing_accuracy", score)