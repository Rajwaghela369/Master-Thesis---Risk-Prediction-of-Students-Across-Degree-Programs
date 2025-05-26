
#### libraries

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import shap
from sklearn import tree
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, f1_score, accuracy_score, RocCurveDisplay
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
from sklearn.inspection import PartialDependenceDisplay
import seaborn as sns
from hyperopt import *

#### Cross Validation Hyperparameter Optimization pipeline

def cross_validation_hpo(x, y, search_space, folds=5, rnd_st=42, algo='lr'):

    model_info = {}          # store models and training / validation results of each folds
    i = 0 
    balanced_accuracies = [] # track balanced accuracy of outer folds
    accuracies = []          # tracl accuracy of outer folds
    specificity_track = []
    sensitivity_track= []
    # define outer folds. 
    outer_cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=rnd_st) # define outer k-folds
    
    for train_idx, val_idx in outer_cv.split(x, y):
        i += 1
        print(f'------------------------------------- Fold {i} --------------------------------------------------')

        ## Splitting X_train into train and validation sets
        x_train, x_val, y_train, y_val = x.iloc[train_idx], x.iloc[val_idx], y.iloc[train_idx], y.iloc[val_idx]
        
        ## define objective (inner loop) optimize the hyperparameters of the model.
        def objective(space):

            cv_sk = StratifiedKFold(n_splits=5, shuffle=True, random_state=64)
            if algo == 'lr': 
                clf = LogisticRegression(**space)
                bal_acc_mean = cross_val_score(estimator=clf, X=x_train, y=y_train, scoring='balanced_accuracy', cv=cv_sk).mean()
                return {'loss': -bal_acc_mean, 'status': STATUS_OK, 'model': clf}
            
            elif algo == 'dt':
                space['random_state'] = rnd_st
                clf = DecisionTreeClassifier(**space)
                bal_acc_mean = cross_val_score(estimator=clf, X=x_train, y=y_train, scoring='balanced_accuracy', cv=cv_sk).mean()
                return {'loss': -bal_acc_mean, 'status': STATUS_OK, 'model': clf}
            
            elif algo == 'rf':
                space['random_state'] = rnd_st
                clf = RandomForestClassifier(**space)
                bal_acc_mean = cross_val_score(estimator=clf, X=x_train, y=y_train, scoring='balanced_accuracy', cv=cv_sk).mean()
                return {'loss': -bal_acc_mean, 'status': STATUS_OK, 'model': clf}
            
            elif algo == 'xgb':
                space['random_state'] = rnd_st
                inner_cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=rnd_st)

                # List to store balanced accuracy for each fold
                fold_balanced_accuracies = []
                
                for train_idx, val_idx in inner_cv.split(x_train, y_train):
                    # Split data into train and validation for current fold
                    x_train_fold, x_val_fold = x_train.iloc[train_idx], x_train.iloc[val_idx]
                    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    clf = XGBClassifier(**space)
                    clf.fit(x_train_fold, y_train_fold)
                    y_pred_val = clf.predict(x_val_fold)
                    # Calculate balanced accuracy for the current fold
                    bal_acc = balanced_accuracy_score(y_val_fold, y_pred_val)
                    fold_balanced_accuracies.append(bal_acc)

                bal_acc_mean = np.mean(fold_balanced_accuracies)
                return {'loss': -bal_acc_mean, 'status': STATUS_OK, 'model': clf}


        # Initialize an empty Trials object to store results of each iteration
        trials = Trials()  

        # Use Hyperopt's fmin function to find the best hyperparameters
        best = fmin(
            fn=objective,        # Function to optimize
            space=search_space,  # Hyperparameter search space
            algo=tpe.suggest,    # Tree-structured Parzen Estimator (TPE) optimization algorithm
            max_evals=10,       # Maximum number of evaluations
            trials=trials        # Store results in the trials object
        )
        print(' ')

        best_trial = min(trials.trials, key=lambda x:x['result']['loss'])     # Retrieve the best trial based on the minimum loss (negative balanced accuracy)
        model = best_trial['result']['model']    # Extract the best model from the best trial
        model.fit(x_train, y_train)              # Train model on training data of the nth fold
        y_pred_val = model.predict(x_val)        # Predict valdiation set of the nth fold
        

        bal_acc = balanced_accuracy_score(y_val, y_pred_val)    # Balanced accuracy (nth fold)
        f1 = f1_score(y_val, y_pred_val)    # f1 score (nth fold)

        # confusion matrix (nth fold)
        cm = confusion_matrix(y_val, y_pred_val)   
        TN, FP, FN, TP = cm.ravel() 
        print('')
        print('\t Confusion Matrix')
        print('\t                 predicted 0     predicted 1')
        print('\t actual 0            ' + "{0:0=2d}".format(TN) + '               ' + "{0:0=2d}".format(FP))
        print('\t actual 1            ' + "{0:0=2d}".format(FN) + '               ' + "{0:0=2d}".format(TP))
        print('')

        acc_score = accuracy_score(y_val, y_pred_val)   # accuracy (nth fold)
        false_pred_rate = (FP + FN) / (TN + FP + FN + TP)   # False predictions (nth fold)
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0    # recall and sensitvity are same
        specificity = TN / (TN + FP) if (TN + FP) != 0 else 0

        # store model and validations results
        model_info[i] = {
            'model': model,
            'balanced_accuracy':bal_acc,
            'accuracy_score': acc_score,
            'f1':f1,
            'confusion_matrix':cm,
            'sensitivity': recall,
            'specificity': specificity,
            'false_predictions': false_pred_rate
        }
        if algo=='lr':
            param = model.get_params()
            print(param)
            
        elif algo=='dt':
            param = model.get_params()
            print(param)
        
        elif algo=='rf':
            param = model.get_params()
            print(param)

        elif algo=='xgb':
            print(model.get_xgb_params())
        print('')
        print(
            f'Balanced acc.: {np.round(bal_acc, 5)*100}%', 
            '\n'
            f'Validation accuracy.: {np.round(acc_score, 5)*100}%',
            '\n'
            f'sensitivity score (for 1): {np.round(recall, 5)*100}%',
            '\n'
            f'specificty score (for 0): {np.round(specificity, 5)*100}%',
            '\n'
            f'f1 score.: {np.round(f1, 5)*100}%',
            '\n'
            f'false prediction: {np.round(false_pred_rate, 5)*100}%',
            '\n')
        print(' ')

        balanced_accuracies.append(bal_acc)  # append balanced accuray of nth fold
        accuracies.append(acc_score)         # append accuray of nth fold
        specificity_track.append(specificity)
        sensitivity_track.append(recall)
    
    # Calculate the mean accuracy and mean balanced accuray with respective standard deviations
    mean_accuracy = np.mean(accuracies)   
    std_accuracy = np.std(accuracies)
    mean_balanced_accuracy = np.mean(balanced_accuracies)
    std_balanced_accuracy = np.std(balanced_accuracies)
    mean_specificity = np.mean(specificity_track)
    std_specificity = np.std(specificity_track)
    mean_sensitivity = np.mean(sensitivity_track)
    std_sensitivity = np.mean(sensitivity_track)
    
    print(f'====================== Summary ======================')
    print(f'Mean Balanced Accuracy: {np.round(mean_balanced_accuracy, 5) * 100}%')
    print(f'Standard Deviation of Balanced Accuracy: +/- {np.round(std_balanced_accuracy, 3)*100}')
    print(f'Mean Accuracy: {np.round(mean_accuracy, 5) * 100}%')
    print(f'Standard Deviation of Accuracy: +/- {np.round(std_accuracy, 3)*100}')
    print(f'mean specificity: {np.round(mean_specificity, 5) * 100}%')
    print(f'Standard Deviation of specificity: +/- {np.round(std_specificity, 3)*100}')
    print(f'Mean sensitivity: {np.round(mean_sensitivity, 5) * 100}%')
    print(f'Standard Deviation of sensitivity: +/- {np.round(std_sensitivity, 3)*100}')

    return model_info


#### Evaluating models based on test set with model selection.

def evaluation(models, X_train, Y_train, x_test, y_test):
    model_info = {} # store the models with test results

    ## iterate to tuned models from folds.
    for i in range(len(models)):
        model = models[i + 1]['model']
        model.fit(X_train, Y_train)
        y_pred = model.predict(x_test)
        
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        acc_score = accuracy_score(y_test, y_pred)

        cm = confusion_matrix(y_test, y_pred)
        TN, FP, FN, TP = cm.ravel()  # Extract values

        false_pred_rate = (FP + FN) / (TN + FP + FN + TP)
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) != 0 else 0

        # store models and test results
        model_info[i] = {
            'model': model,
            'balanced_accuracy':bal_acc,
            'accuracy_score': acc_score,
            'f1':f1,
            'confusion_matrix':cm,
            'sensitivity': recall,
            'specificity': specificity,
            'false_predictions': false_pred_rate,
            'y_test':y_test,
            'y_pred':y_pred
        }

    ## selection of best model based on balanced accuray.
    best_model_idx = max(model_info, key=lambda x: model_info[x]['balanced_accuracy'])
    best_model = model_info[best_model_idx]
    return best_model

#### Display best model results and metrices
def results(model, cols, algo='lr'):

    print('model parameters:')
    if algo=='lr':
        param = model['model'].get_params()
        print(f"solver: {param['solver']}, C: {param['C']}, penalty: {param['penalty']}, "
            f"fit_intercept: {param['fit_intercept']}, class_weight: {param['class_weight']}, "
            f"multi_class: {param['multi_class']}")
        
    elif algo=='dt':
        param = model['model'].get_params()
        print(param)
    
    elif algo=='rf':
        param = model['model'].get_params()
        print(param)

    elif algo=='xgb':
        param = model['model'].get_xgb_params()
        print(param)

    # confusion matrix
    cm = model['confusion_matrix']
    TN, FP, FN, TP = cm.ravel() 
    print('')
    print('\t Confusion Matrix')
    print('\t                 predicted 0     predicted 1')
    print('\t actual 0            ' + "{0:0=2d}".format(TN) + '               ' + "{0:0=2d}".format(FP))
    print('\t actual 1            ' + "{0:0=2d}".format(FN) + '               ' + "{0:0=2d}".format(TP))
    print('')


    # scores / metrices
    print(' ')
    print(f"Balanced acc.: {np.round(model['balanced_accuracy'], 5)*100}%", 
        '\n'
        f"Test accuracy.: {np.round(model['accuracy_score'], 5)*100}%",
        '\n'
        f"sensitivity (for 1): {np.round(model['sensitivity'], 5)*100}%",
        '\n'
        f"specificty (for 0): {np.round(model['specificity'], 5)*100}%",
        '\n'
        f"f1 score.: {np.round(model['f1'], 5)*100}%",
        '\n')

    print('---------------------------------------------')
    
    ## ROC curve
    curve = RocCurveDisplay.from_predictions(
        model['y_test'],
        model['y_pred'],
        name=f"Student Drop out model",
        color="darkorange",
    )
    curve.ax_.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        
        title=f"Student Drop out: model",
    )   

    if algo=='lr':      # - logistic regression
        feature_importance = pd.DataFrame({
            'variable': cols,
            'coefficient': model['model'].coef_[0]
        }).round(decimals = 5).sort_values('coefficient', ascending = False).style.bar(color = ['red', 'green'], align = 'zero')
        return feature_importance
    
    elif algo=='dt':    # - decision tree
        importances = model['model'].feature_importances_
        features = model['model'].feature_names_in_
        feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance':importances
        }).sort_values(by='Importance', ascending=False)

        fig = plt.figure(figsize=(20, 12))
        tree_structure = tree.plot_tree( 
        model['model'],
        feature_names=cols,
        class_names=None,
        filled=True,
        )   

        return feature_importance.style.bar(color='lightblue')
    
    elif algo=='rf':   # xgb boost
        importances = model['model'].feature_importances_
        features = model['model'].feature_names_in_
        feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance':importances
        }).sort_values(by='Importance', ascending=False)

        return feature_importance.style.bar(color='lightblue')

    elif algo=='xgb':   # xgb boost
        importances = model['model'].feature_importances_
        features = model['model'].feature_names_in_
        feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance':importances
        }).sort_values(by='Importance', ascending=False)

        return feature_importance.style.bar(color='lightblue')

    


#### Plot Partial Depencedence with respect to Classes.
def plot_partial_dependence(model, x_test, features, n_jobs=5, response_method='auto', kind='average', n_cols=6, sem=1):
    """
    Function to generate and display Partial Dependence Plots (PDPs) for a given model and feature set.

    Parameters:
    - model: Trained estimator (e.g., RandomForest, XGBoost, etc.) to be used for PDP.
    - x_test: The dataset on which the partial dependence is to be calculated (test data).
    - features: List of features for which PDPs are to be created.
    - n_jobs: Number of CPU cores to use for parallel computation. Default is 5.
    - response_method: Method for calculating the response, default is 'auto'.
    - kind: Type of plot, either 'both', 'individual', or 'average'. Default is 'both'.
    - n_cols: Number of columns in the plot grid. Default is 6.
    - plot_size: Tuple specifying the size of the plot. Default is (22, 4).
    
    Returns:
    - Displays the PDP plot.
    """
    # Generate the Partial Dependence Plot (PDP) using the given estimator and data
    pdp_plot = PartialDependenceDisplay.from_estimator(
        estimator=model,         # Trained model
        X=x_test,                # Test data (features)
        features=features,       # List of feature indices or names to generate PDPs for
        n_jobs=n_jobs,           # Parallelization for faster computation
        response_method=response_method,  # Response method for calculating the prediction
        kind=kind,               # 'both' for showing individual and averaged PDPs
        n_cols=n_cols            # Number of columns in the grid of plots
    )
    if sem==1:
        # Adjust the plot size for better visibility
        plt.gcf().set_size_inches(22, 4)
        plt.show()
    elif sem==2:
        # Adjust the plot size for better visibility
        plt.gcf().set_size_inches(30, 12)
        plt.show()
    elif sem==3:
        # Adjust the plot size for better visibility
        plt.gcf().set_size_inches(35, 22)
        plt.show()


#### Plot force plot
def force_plot(model, x_test, y_test, shap_values, student_index, link='logit'):

    student_status = y_test.iloc[student_index]
    y_pred = model.predict(x_test.iloc[student_index].values.reshape(1, -1))
    y_prob = model.predict_proba(x_test.iloc[student_index].values.reshape(1, -1))

    # Display student details and predicitons
    print(f"Student index {student_index} grades: \n {x_test.iloc[student_index]} \n")
    print(f"Status: {student_status}, Prediction: {y_pred}, Probability: {y_prob}")


    # Generate SHAP force plot
    return shap.force_plot(shap_values[student_index], link=link)