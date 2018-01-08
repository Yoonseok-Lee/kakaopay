###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
#
# Display inline matplotlib plots with IPython
#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')
###########################################

import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import seaborn as sns

from xgboost.sklearn import XGBRegressor
from sklearn import linear_model
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.metrics import r2_score as performance_metric
import sklearn.learning_curve as curves
from functools import partial

def PLSModelLearning(X, y, components=5):
    """ Calculates the performance of several models with varying sizes of training data.
        The learning and testing scores for each model are then plotted. """
    
    # Create 10 cross-validation sets for training and testing
    spiltsObj = ShuffleSplit(n_splits = 10, test_size = 0.2, random_state = 0)
    cv = spiltsObj.get_n_splits(X)
    
    # Generate the training set sizes increasing by 50
    train_sizes = [150, 200, 250, 300, 350]
    
    regressor = PLSRegression(n_components=components)
    
    # Calculate the training and testing scores
    sizes, train_scores, test_scores = curves.learning_curve(regressor, X, y, \
    cv = cv, train_sizes = train_sizes, scoring = 'r2')
        
    # Find the mean and standard deviation for smoothing
    train_std = np.std(train_scores, axis = 1)
    train_mean = np.mean(train_scores, axis = 1)
    test_std = np.std(test_scores, axis = 1)
    test_mean = np.mean(test_scores, axis = 1)
    
    # Subplot the learning curve 
    pl.figure(figsize=(7, 5))
    pl.plot(sizes, train_mean, 'o-', color = 'r', label = 'Training Score')
    pl.plot(sizes, test_mean, 'o-', color = 'g', label = 'Testing Score')
    pl.fill_between(sizes, train_mean - train_std, \
            train_mean + train_std, alpha = 0.15, color = 'r')
    pl.fill_between(sizes, test_mean - test_std, \
            test_mean + test_std, alpha = 0.15, color = 'g')
    
    # Labels
    pl.title('PLSModelLearning' + ' ' + 'Components Size=' + str(components))
    pl.xlabel('Number of Training Points')
    pl.ylabel('Score')
    pl.xlim([140, 360])
    pl.ylim([-0.05, 1.05])
    pl.show()

def LinearModelLearning(X, y):
    """ Calculates the performance of several models with varying sizes of training data.
        The learning and testing scores for each model are then plotted. """
    
    # Create 10 cross-validation sets for training and testing
    spiltsObj = ShuffleSplit(n_splits = 10, test_size = 0.2, random_state = 0)
    cv = spiltsObj.get_n_splits(X)
    
    # Generate the training set sizes increasing by 50
    train_sizes = np.rint(np.linspace(1, X.shape[0]*0.8 - 1, 9)).astype(int)
    
    regressor = linear_model.LinearRegression()
    
    # Calculate the training and testing scores
    sizes, train_scores, test_scores = curves.learning_curve(regressor, X, y, \
    cv = cv, train_sizes = train_sizes, scoring = 'r2')
        
    # Find the mean and standard deviation for smoothing
    train_std = np.std(train_scores, axis = 1)
    train_mean = np.mean(train_scores, axis = 1)
    test_std = np.std(test_scores, axis = 1)
    test_mean = np.mean(test_scores, axis = 1)
    
    # Subplot the learning curve 
    pl.figure(figsize=(7, 5))
    pl.plot(sizes, train_mean, 'o-', color = 'r', label = 'Training Score')
    pl.plot(sizes, test_mean, 'o-', color = 'g', label = 'Testing Score')
    pl.fill_between(sizes, train_mean - train_std, \
            train_mean + train_std, alpha = 0.15, color = 'r')
    pl.fill_between(sizes, test_mean - test_std, \
            test_mean + test_std, alpha = 0.15, color = 'g')
    
    # Labels
    pl.title('General Linear Model')
    pl.xlabel('Number of Training Points')
    pl.ylabel('Score')
    pl.xlim([0, X.shape[0]*0.8])
    pl.ylim([-0.05, 1.05])
    pl.show()


def TRModelLearning(X, y, targetModel, title=None):
    """ Calculates the performance of several models with varying sizes of training data.
        The learning and testing scores for each model are then plotted. """
    
    # Create 10 cross-validation sets for training and testing
    spiltsObj = ShuffleSplit(n_splits = 10, test_size = 0.2, random_state = 0)
    cv = spiltsObj.get_n_splits(X)
    
    # Generate the training set sizes increasing by 50
    train_sizes = np.rint(np.linspace(1, X.shape[0]*0.8 - 1, 9)).astype(int)

    print(train_sizes)

    # Create the figure window
    fig = pl.figure(figsize=(10,7))

    # Create three different models based on max_depth
    for k, depth in enumerate([1,3,6,10]):
        
        # Create a Decision tree regressor at max_depth = depth
        regressor = targetModel(max_depth = depth)
        
        # Calculate the training and testing scores
        sizes, train_scores, test_scores = curves.learning_curve(regressor, X, y, \
            cv = cv, train_sizes = train_sizes, scoring = 'r2')
        
        # Find the mean and standard deviation for smoothing
        train_std = np.std(train_scores, axis = 1)
        train_mean = np.mean(train_scores, axis = 1)
        test_std = np.std(test_scores, axis = 1)
        test_mean = np.mean(test_scores, axis = 1)

        # Subplot the learning curve 
        ax = fig.add_subplot(2, 2, k+1)
        ax.plot(sizes, train_mean, 'o-', color = 'r', label = 'Training Score')
        ax.plot(sizes, test_mean, 'o-', color = 'g', label = 'Testing Score')
        ax.fill_between(sizes, train_mean - train_std, \
            train_mean + train_std, alpha = 0.15, color = 'r')
        ax.fill_between(sizes, test_mean - test_std, \
            test_mean + test_std, alpha = 0.15, color = 'g')
        
        # Labels
        ax.set_title(title+'max_depth = %s'%(depth))
        ax.set_xlabel('Number of Training Points')
        ax.set_ylabel('Score')
        ax.set_xlim([0, X.shape[0]*0.8])
        ax.set_ylim([-0.05, 1.05])
    
    # Visual aesthetics
    ax.legend(bbox_to_anchor=(1.05, 2.05), loc='lower left', borderaxespad = 0.)
    fig.suptitle('Decision Tree Regressor Learning Performances', fontsize = 16, y = 1.03)
    fig.tight_layout()
    fig.show()


def TRModelComplexity(X, y, targetModel, title=None):
    """ Calculates the performance of the model as model complexity increases.
        The learning and testing errors rates are then plotted. """

    # Create 10 cross-validation sets for training and testing
    spiltsObj = ShuffleSplit(n_splits = 10, test_size = 0.2, random_state = 0)
    cv = spiltsObj.get_n_splits(X)
    
    # Vary the max_depth parameter from 1 to 10
    max_depth = np.arange(1,11)

    # Calculate the training and testing scores
    train_scores, test_scores = curves.validation_curve(targetModel(), X, y, \
        param_name = "max_depth", param_range = max_depth, cv = cv, scoring = 'r2')

    # Find the mean and standard deviation for smoothing
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot the validation curve
    pl.figure(figsize=(7, 5))
    
    pl.title(title+'Complexity')
    pl.plot(max_depth, train_mean, 'o-', color = 'r', label = 'Training Score')
    pl.plot(max_depth, test_mean, 'o-', color = 'g', label = 'Validation Score')
    pl.fill_between(max_depth, train_mean - train_std, \
        train_mean + train_std, alpha = 0.15, color = 'r')
    pl.fill_between(max_depth, test_mean - test_std, \
        test_mean + test_std, alpha = 0.15, color = 'g')
    
    # Visual aesthetics
    pl.legend(loc = 'lower right')
    pl.xlabel('Maximum Depth')
    pl.ylabel('Score')
    pl.ylim([-0.05,1.05])
    pl.show()

# 로드 공간에서 최대,최소 사이의 랜덤 값 사이즈 만큼 생성
def LogSampler(min_val, max_val, seed, size):
    log_min_val = np.log(min_val)
    log_max_val = np.log(max_val)
    assert log_max_val > log_min_val
    log_spread = log_max_val - log_min_val
    random_state = np.random.RandomState(seed)
    data = np.ndarray(size)
    for x in np.nditer(np.arange(size)):
        log_value = random_state.rand() * log_spread + log_min_val
        data[x] = np.exp(log_value)
        
    return data

def get_optimal_GBModel(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """
    
    # Create cross-validation sets from the training data
    spiltsObj = ShuffleSplit(n_splits = 10, test_size = 0.2, random_state = 0)
    cv_sets = spiltsObj.get_n_splits(X)

    # Create a Gradient Boosted regressor object
    GBRegressor = partial(XGBRegressor, seed=8329)
    regressor = GBRegressor()

    # Create a dictionary of parameter distributions to try
    params = {'max_depth': range(3, 7),
              'learning_rate': LogSampler(0.01, 1.0, seed=29385, size=10),
              'gamma': LogSampler(0.000001, 1.0, seed=46435, size=10),
              'reg_lambda': LogSampler(0.000001, 1.0, seed=92385, size=10)}

    # Transform 'performance_metric' into a scoring function using 'make_scorer' 
    scoring_fnc = make_scorer(performance_metric)

    # Create the grid search object
    #grid = GridSearchCV(regressor, param_grid=params, 
    #                          scoring=scoring_fnc, cv=cv_sets, n_jobs = 4)

    grid = RandomizedSearchCV(regressor, param_distributions=params, 
                              scoring=scoring_fnc, cv=cv_sets, n_iter=30)
     
    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_

def get_performance_model(model, data_test, target_test):
    client_data = data_test
    
    predictions = pd.DataFrame(dict(Actual=target_test * 1000.,
                                    Predicted=model.predict(client_data) * 1000.,))
    
    r2 = performance_metric(predictions.Predicted, predictions.Actual)
    errors = (predictions.Actual - predictions.Predicted)
    error = errors.std()
    
    predictions.plot.scatter(x='Predicted', 
                             y='Actual', 
                             title='Out of Sample R2 on clients: %.2f. Error Std. Dev +-$%.2f' 
                             % (r2, error,)
                             )
    
    error_data = pd.DataFrame({'|error|': np.abs(errors), 'Price': target_test * 1000})
    sns.lmplot(data=error_data, x='Price', y='|error|', size=10, aspect=1.2)

    


#def check_model_factors(model):
#    scores = model.booster().get_fscore()
#    feature_importances = pd.Series(scores)
#    feature_importances = feature_importances / feature_importances.sum()
#    feature_importances.sort_values().plot.bar(colormap='Paired', title='Feature importances')
    

def PredictTrials(X, y, fitter, data):
    """ Performs trials of fitting and predicting data. """

    # Store the predicted prices
    prices = []

    for k in range(10):
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, \
            test_size = 0.2, random_state = k)
        
        # Fit the data
        reg = fitter(X_train, y_train)
        
        # Make a prediction
        pred = reg.predict([data[0]])[0]
        prices.append(pred)
        
        # Result
        print ("Trial {}: ${:,.2f}".format(k+1, pred))

    # Display price range
    print ("\nRange in prices: ${:,.2f}".format(max(prices) - min(prices)))