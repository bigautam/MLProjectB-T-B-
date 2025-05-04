from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np

from sklearn.datasets import load_svmlight_file

import argparse
import pickle
import json
import os

import evaluators

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_file',
                        type=str,
                        required=True,
                        help='Dataset (as SVMLight)')
    parser.add_argument('--models',
                        nargs='+',
                        type=str,
                        required=True,
                        help='Models to evaluate',
                        choices=['knn', 'decision_tree', 'logistic_regression', 'mlp'])
    parser.add_argument('--pickle_out_dir',
                        type=str,
                        help='Output destination for Evaluator pickle')
    parser.add_argument('--scores_out_dir',
                        type=str,
                        help='Output destination for scores (JSON)')
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
        if not all(metric in score for score in self.scores.values()):
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
    def evaluate_and_save(config, X, y):
        model_evaluator = Evaluator(model_parameterizer=config['model_parameterizer'], parameter_domain=config['parameter_domain'], X=X, y=y, metrics=config['metrics'])
        model_evaluator.train_test_split()
        model_evaluator.fit()

        evaluations = model_evaluator.evaluate(model_evaluator.X_test, model_evaluator.y_test)
        print(f'Best parameters for {config["model_name"]}')
        for metric in config['metrics'].keys():
            best_parameter = model_evaluator.best_parameter_by_metric(metric=metric)
            print(f'\tBest parameter for {metric} is {best_parameter} with {evaluations[best_parameter][metric]}')

        if args.pickle_out_dir:
            pickle_filepath = os.path.join(args.pickle_out_dir, 'evaluators', f'{config["model_name"]}.pkl')
            os.makedirs(os.path.dirname(pickle_filepath), exist_ok=True)
            with open(file=pickle_filepath, mode='wb') as pickle_file:
                pickle.dump(obj=model_evaluator, file=pickle_file)
        
        if args.scores_out_dir:
            scores_filepath = os.path.join(args.scores_out_dir, 'results', f'{config["model_name"]}.json')
            os.makedirs(os.path.dirname(scores_filepath), exist_ok=True)
            with open(file=scores_filepath, mode='w') as json_file:
                json.dump(obj=model_evaluator.scores, fp=json_file)

    args = get_args()

    X, y = load_svmlight_file(args.dataset_file)

    print(f'=== BEGIN EVALUATING FOR {args.models} ===')
    for config in evaluators.get_configs_for_names(args.models):
        print(f'EVALUATION FOR {config["model_name"]}')
        evaluate_and_save(config=config, X=X, y=y)

if __name__ == '__main__':
    main()