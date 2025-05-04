from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from sklearn.datasets import load_svmlight_file

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_file',
                        type=str,
                        required=True,
                        help='Dataset (as SVMLight)')
    return parser.parse_args()

class Evaluator:
    def __init__(self, model_parameterizer, parameter_domain, X, y, metrics):
        self.model_parameterizer = model_parameterizer
        self.parameter_domain = parameter_domain
        self.models = {parameter: self.model_parameterizer(parameter) for parameter in self.parameter_domain}
        self.X = X
        self.y = y
        self.metrics = metrics # Should be dict[str, metric]
        self.X_train, self.y_train, self.X_test, self.y_test = None, None, None, None
        self.scores = {}

    def train_test_split(self, train_size=0.7, random_state=111):
        if self.X is None or self.y is None:
            raise ValueError('Either X or y are unspecified so far')
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, train_size=train_size, random_state=random_state)
    
    def fit(self):
        if not all(v is not None for v in [self.X_train, self.y_train, self.X_test, self.y_test]):
            raise ValueError('Splits have not been assigned')
        for parameter in self.models.keys():
            self.models[parameter].fit(self.X_train, self.y_train)

    def predict_with_parameter(self, X, parameter):
        return self.models[parameter].predict(X)
    
    def predict_all(self, X):
        return {parameter: model.predict(X) for parameter, model in self.models.items()}
    
    def collect_metrics(self, y_true, y_pred):
        return {metric_name: metric(y_true, y_pred) for metric_name, metric in self.metrics.items()}
    
    def evaluate(self, X, y_true):
        self.scores = {parameter: self.collect_metrics(y_true, y_pred) for parameter, y_pred in self.predict_all(X).items()}
        return self.scores

    def best_parameter_by_metric(self, metric):
        if all(score is not metric for score in self.scores.values()):
            raise ValueError(f'Some score does not contain the metric {metric}')
        return max(self.scores, key=lambda parameter: self.scores[parameter][metric])

    def confusion_matrix(self, y_true, y_pred, axes, show_labels=True):
        conf_matrix = confusion_matrix(y_true, y_pred)
        axes.matshow(conf_matrix)
        for i, j in np.ndindex(conf_matrix.shape):
            axes.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center')
        axes.set_xlabel('Predicted')
        axes.set_ylabel('Actual')

def main():
    args = get_args()

    X, y = load_svmlight_file(args.dataset_file)

    knn_parameterizer = lambda k: KNeighborsClassifier(n_neighbors=k)
    knn_parameter_domain = range(1,16)
    knn_metrics = {'accuracy': accuracy_score}

    knn_evaluator = Evaluator(model_parameterizer=knn_parameterizer, parameter_domain=knn_parameter_domain, X=X, y=y, metrics=knn_metrics)
    knn_evaluator.train_test_split()
    knn_evaluator.fit()
    evaluations = knn_evaluator.evaluate(knn_evaluator.X_test, knn_evaluator.y_test)
    best_parameter = knn_evaluator.best_parameter_by_metric(metric='accuracy')
    print(f'Best parameter is {best_parameter} with metrics {evaluations[best_parameter]}')

if __name__ == '__main__':
    main()