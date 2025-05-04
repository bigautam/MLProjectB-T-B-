from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import numpy as np

def knn_parameterizer(k):
        return KNeighborsClassifier(n_neighbors=k)

def decision_tree_parameterizer(depth):
        return DecisionTreeClassifier(max_depth=depth)

def logistic_regression_parameterizer(C):
        return LogisticRegression(C=C)

def mlp_parameterizer(iters):
        return MLPClassifier(hidden_layer_sizes=(64, 32),
                    max_iter=iters,
                    activation='relu',
                    solver='adam',
                    random_state=42,
                    verbose=True)

evaluator_configs = {
    'knn': {
        'model_name': 'KNN',
        'model_parameterizer': knn_parameterizer,
        'parameter_domain': range(1,16),
        'metrics': {'accuracy': accuracy_score, 'f1': f1_score, 'precision': precision_score, 'recall': recall_score}
    },
    'decision_tree': {
        'model_name': 'Decision Tree',
        'model_parameterizer': decision_tree_parameterizer,
        'parameter_domain': range(1,10),
        'metrics': {'accuracy': accuracy_score, 'f1': f1_score, 'precision': precision_score, 'recall': recall_score}
    },
    'logistic_regression': {
        'model_name': 'Logistic Regression',
        'model_parameterizer': logistic_regression_parameterizer,
        'parameter_domain': [x/10 for x in range(1,11)],
        'metrics': {'accuracy': accuracy_score, 'f1': f1_score, 'precision': precision_score, 'recall': recall_score}
    },
    'mlp': {
        'model_name': 'MLP',
        'model_parameterizer': mlp_parameterizer,
        'parameter_domain': range(30,330,30),
        'metrics': {'accuracy': accuracy_score, 'f1': f1_score, 'precision': precision_score, 'recall': recall_score}
    }
}

def get_configs_for_names(names):
    return [evaluator_configs[name] for name in names]