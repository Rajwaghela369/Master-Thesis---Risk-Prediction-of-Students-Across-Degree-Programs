
#### Libraries

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import shap
from sklearn import tree
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, 
    balanced_accuracy_score,
    f1_score,
    accuracy_score,
    RocCurveDisplay
)
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
from sklearn.inspection import PartialDependenceDisplay
import seaborn as sns
from hyperopt import *

class ModelTrainer:
    """Class for training and evaluating machine learning models"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model_info = {}

    @staticmethod
    def split_data(df, features):
        """
        Split dataframes into features (X) and target (y)
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            
        Returns:
            x_train: Training features
            y_train: Training target
            x_test: Test features 
            y_test: Test target
        """
        # Split into features and target
        x = df[features]
        y = df['status']
        return x, y
        
    def cross_validation_hpo(self, x, y, search_space, folds=5, algo='lr'):
        """
        Perform cross-validation with hyperparameter optimization.
        
        Args:
            x: Features DataFrame
            y: Target Series 
            search_space: Hyperparameter search space
            folds: Number of CV folds (default 5)
            algo: Algorithm to use - 'lr', 'dt', 'rf' or 'xgb' (default 'lr')
            
        Returns:
            Dictionary containing model info and metrics for each fold
        """
        balanced_accuracies = []
        accuracies = []
        specificity_track = []
        sensitivity_track = []
        false_predictions = []
        outer_cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.random_state)
        
        for fold_num, (train_idx, test_idx) in enumerate(outer_cv.split(x, y), 1):
            print(f'------------------------------------- Fold {fold_num} --------------------------------------------------')

            x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            best_model = self._optimize_hyperparameters(x_train, y_train, search_space, algo, folds)
            metrics, y_pred = self._evaluate_fold(best_model, x_train, y_train, x_test, y_test)
            
            self.model_info[fold_num] = {
                'model': best_model,
                'x_train': x_train,
                'x_test': x_test,
                'y_train': y_train,
                'y_test': y_test,
                'y_pred': y_pred,
                **metrics
            }
            
            self._print_fold_results(metrics, best_model, algo)
            
            balanced_accuracies.append(metrics['balanced_accuracy'])
            accuracies.append(metrics['accuracy_score'])
            specificity_track.append(metrics['specificity'])
            sensitivity_track.append(metrics['sensitivity'])
            false_predictions.append(metrics['false_predictions'])
        self._print_summary_stats(balanced_accuracies, accuracies, specificity_track, sensitivity_track, false_predictions)
        return self.filter_best_model(feature_cols=x.columns, algo=algo)

    def _optimize_hyperparameters(self, x_train, y_train, search_space, algo, folds):
        """Optimize model hyperparameters using hyperopt"""
        def objective(space):
            cv_sk = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            
            if algo == 'lr':
                clf = LogisticRegression(**space)
                space['random_state'] = self.random_state
                bal_acc_mean = cross_val_score(
                estimator=clf,
                X=x_train, 
                y=y_train,
                scoring='balanced_accuracy',
                cv=cv_sk
                ).mean()
                
                return {'loss': -bal_acc_mean, 'status': STATUS_OK, 'model': clf}
            
            elif algo in ['dt', 'rf']:
                space['random_state'] = self.random_state
                clf = (DecisionTreeClassifier(**space) if algo == 'dt' 
                      else RandomForestClassifier(**space))
                
                bal_acc_mean = cross_val_score(
                estimator=clf,
                X=x_train, 
                y=y_train,
                scoring='balanced_accuracy',
                cv=cv_sk
                ).mean()
                
                return {'loss': -bal_acc_mean, 'status': STATUS_OK, 'model': clf}
            
            elif algo == 'xgb':
                space['random_state'] = self.random_state
                return self._xgb_objective(space, x_train, y_train, folds)
                

        trials = Trials()
        fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=10,
            trials=trials
        )
        
        best_trial = min(trials.trials, key=lambda x: x['result']['loss'])
        return best_trial['result']['model']

    def _xgb_objective(self, space, x_train, y_train, folds):
        """XGBoost specific objective function"""
        space['random_state'] = self.random_state
        inner_cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.random_state)
        
        fold_scores = []
        for train_idx, val_idx in inner_cv.split(x_train, y_train):
            x_train_fold = x_train.iloc[train_idx]
            x_val_fold = x_train.iloc[val_idx]
            y_train_fold = y_train.iloc[train_idx]
            y_val_fold = y_train.iloc[val_idx]
            
            clf = XGBClassifier(**space)
            clf.fit(x_train_fold, y_train_fold)
            y_pred_val = clf.predict(x_val_fold)
            fold_scores.append(balanced_accuracy_score(y_val_fold, y_pred_val))

        return {'loss': -np.mean(fold_scores), 'status': STATUS_OK, 'model': clf}

    def filter_best_model(self, feature_cols, algo='lr'):
        """Select Model with highest Balanced Accuracy from the folds"""
        fold_no, best_model = max(self.model_info.items(), key=lambda x: x[1]['balanced_accuracy'])
        if algo == 'dt':
            plt.figure(figsize=(20,10))
            tree.plot_tree(best_model['model'], feature_names=feature_cols, filled=True, class_names=True)
            plt.show()
        print(f"best model is from {fold_no} fold: {best_model['model']}")
        self._plot_roc_curve(best_model)
        return best_model, self._get_feature_importance(best_model, feature_cols, algo)

    @staticmethod
    def plot_partial_dependence(model, n_jobs=5, 
                              response_method='auto', kind='average', n_cols=6, size=1):
        """Generate partial dependence plots"""
        pdp = PartialDependenceDisplay.from_estimator(
            estimator=model['model'],
            X=model['x_test'],
            features=model['x_test'].columns,
            n_jobs=n_jobs,
            response_method=response_method,
            kind=kind,
            n_cols=n_cols
        )
        
        sizes = {1: (22, 4), 2: (30, 12), 3: (35, 22)}
        plt.gcf().set_size_inches(*sizes[size])
        plt.show()

    @staticmethod
    def plot_shap_force(model, shap_values, 
                       student_index, link='logit', model_name='lr'):
        """Generate SHAP force plot for a single prediction"""
        clf = model['model']
        x_test = model['x_test']
        y_test = model['y_test']
        student = x_test.iloc[student_index].values.reshape(1, -1)
        status = y_test.iloc[student_index]
        pred = clf.predict(student)
        prob = clf.predict_proba(student)
        
        df = pd.DataFrame({
            'grades': x_test.iloc[student_index],        })
        print(f"Student index {student_index} grades:\n{df}")
        print(f"Status: {status}, Prediction: {pred}, Probability [0]: {prob[0][0]:.3f}, Probability [1]: {prob[0][1]:.3f}")

        if model_name == 'lr':
            shap_values = shap_values
        elif model_name == 'xgb': 
            shap_values = shap_values
        elif model_name == 'dt':
            shap_values = shap_values[...,1]
        elif model_name == 'rf':
            shap_values = shap_values[...,1]

        return shap.force_plot(shap_values[student_index], link=link)

    @staticmethod
    def _calculate_metrics(y_true, y_pred):
        """Calculate classification metrics"""
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        return {
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'accuracy_score': accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'confusion_matrix': cm,
            'sensitivity': tp / (tp + fn) if (tp + fn) != 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) != 0 else 0,
            'false_predictions': (fp + fn) / (tn + fp + fn + tp)
        }

    def _evaluate_fold(self, model, x_train, y_train, x_test, y_test):
        """Evaluate model on validation fold"""
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        return self._calculate_metrics(y_test, y_pred), y_pred

    @staticmethod
    def _print_fold_results(metrics, model, algo):
        """Print results for a single fold"""
        cm = metrics['confusion_matrix']
        tn, fp, fn, tp = cm.ravel()
        
        print('\nConfusion Matrix')
        print('\t                 predicted 0     predicted 1')
        print(f'\t actual 0            {tn:02d}               {fp:02d}')
        print(f'\t actual 1            {fn:02d}               {tp:02d}\n')
        
        print('Model parameters:')
        if algo == 'xgb':
            print(model.get_xgb_params())
        else:
            print(model.get_params())
            
        print(f"\nBalanced acc.: {metrics['balanced_accuracy']*100:.5f}%")
        print(f"Validation accuracy.: {metrics['accuracy_score']*100:.5f}%")
        print(f"Sensitivity score (for 1): {metrics['sensitivity']*100:.5f}%")
        print(f"Specificity score (for 0): {metrics['specificity']*100:.5f}%")
        print(f"F1 score.: {metrics['f1']*100:.5f}%")
        print(f"False prediction: {metrics['false_predictions']*100:.5f}%\n")

    @staticmethod
    def _print_summary_stats(balanced_accuracies, accuracies, specificity_track, sensitivity_track, false_predictions):
        """Print summary statistics across all folds"""
        print('====================== Summary ======================')
        
        metrics = {
            'Balanced Accuracy': balanced_accuracies,
            'Accuracy': accuracies,
            'Sensitivity': sensitivity_track,
            'Specificity': specificity_track,
            'False Predictions': false_predictions
        }
        
        for metric_name, values in metrics.items():
            mean = np.mean(values)
            std = np.std(values)
            print(f'Mean {metric_name}: {mean*100:.5f}%')
            print(f'Standard Deviation of {metric_name}: +/- {std*100:.3f}\n')

    @staticmethod
    def _plot_roc_curve(model):
        """Plot ROC curve"""
        RocCurveDisplay.from_predictions(
            model['y_test'],
            model['y_pred'],
            name="Student Drop out model",
            color="darkorange"
        ).ax_.set(
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title="Student Drop out: model"
        )

    @staticmethod
    def _get_feature_importance(model, feature_cols, algo):
        """Get feature importance DataFrame"""
        if algo == 'lr':
            return pd.DataFrame({
                'variable': feature_cols,
                'coefficient': model['model'].coef_[0]
            }).round(5).sort_values('coefficient', ascending=False).style.bar(
                color=['red', 'green'], 
                align='zero'
            )
        else:
            importances = model['model'].feature_importances_
            features = model['model'].feature_names_in_
            return pd.DataFrame({
                'Feature': features,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False).style.bar(color='lightblue')