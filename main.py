import numpy as np
# Set random seed
seed = 46
np.random.seed(seed)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, confusion_matrix,
    roc_auc_score
)
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.impute import SimpleImputer
from collections import Counter
from prettytable import PrettyTable

def remove_outliers(data, column, threshold=2):
    z_scores = np.abs(stats.zscore(data[column]))
    data_without_outliers = data[(z_scores < threshold)]
    return data_without_outliers

def select_features(X_train, y_train, method, feature, X,
                    response_variable):
    fs = SelectKBest(score_func=method, k=feature)
    selected = fs.fit_transform(X_train, y_train)
    idx = fs.get_support(indices=True)
    columns = X.iloc[:, idx].columns
    
    fig = plt.figure(figsize=(5,3))
    pd.Series(fs.scores_[idx], index=columns).nlargest(
        feature).plot(kind='barh', color='gray').invert_yaxis()
    # Increase the text size
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Score', fontsize=20)
    plt.ylabel('Features', fontsize=20)
    plt.show()
    
    return idx

def select_model():
    lda = LinearDiscriminantAnalysis()
    knn = KNeighborsClassifier(n_neighbors=4)
    mlp = MLPClassifier(hidden_layer_sizes=(80, 80),
                        activation='tanh', max_iter=5000)
    return lda, knn, mlp

def split_train_test(X, y, feature_options, outcome, folds):
    cols = X.columns
    tables = []
    kfold = StratifiedKFold(n_splits=folds, shuffle=True)
    selected_features = None

    for i, (train, test) in enumerate(kfold.split(X, y)):
        X_train, y_train = X.iloc[train], y.iloc[train]
        X_test, y_test = X.iloc[test], y.iloc[test]
        org = X_train
        
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        ros = RandomOverSampler(random_state=seed)
        X_train, y_train = ros.fit_resample(X_train, y_train)
        
        imp = SimpleImputer(strategy="median")
        X_train = pd.DataFrame(imp.fit_transform(X_train))
        X_test = pd.DataFrame(imp.transform(X_test))
        
        if i == 0:
            selected_features = select_features(X_train, y_train, 
                                                mutual_info_classif,
                                                feature, org, outcome)
            
        X_train_selected = X_train.iloc[:, selected_features]
        X_test_selected = X_test.iloc[:, selected_features]
        
        table = main(X_train_selected, X_test_selected, 
                     y_train, y_test, feature, cols, outcome)
        tables.extend(table)

    return pd.DataFrame(tables)

def main(X_train, X_test, y_train, y_test, feature, cols, outcome):
    models = select_model()
    results = []
    threshold = 0.55  # Define the threshold value

    for model in models:
        # Training
        model.fit(X_train, y_train)
        y_pred_train = model.predict_proba(X_train)[:, 1]  # Use probability predictions for positive class
        y_binary = np.where(y_pred_train > threshold, 1, 0)  # Convert to binary form
        train_score = balanced_accuracy_score(y_train, y_binary)

        # Testing
        y_pred_proba = model.predict_proba(X_test)
        if len(np.unique(y_train)) == 2:  # Binary classification case
            y_pred = np.where(y_pred_proba[:, 1] >= threshold, 1, 0)  # Adjust threshold as needed
            accuracy = balanced_accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            f1 = f1_score(y_test, y_pred, average='weighted')
        else:  # Multiclass classification case
            y_pred_proba_normalized = y_pred_proba / np.sum(y_pred_proba,
                                                            axis=1, 
                                                            keepdims=True)
            y_pred = np.argmax(y_pred_proba_normalized, axis=1)  # Convert probabilities to class labels
            accuracy = balanced_accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, 
                                    y_pred_proba_normalized, 
                                    multi_class='ovo')
            f1 = f1_score(y_test, y_pred, average='macro')  # or 'micro' based on preference

        # Saving results
        results.append({
            'Response': outcome,
            'Features': feature,
            'Train_Bal_Accuracy': train_score,
            'Model': model.__class__.__name__,
            'Balanced Accuracy': accuracy,
            'F1 Score': f1,
            'ROC AUC': roc_auc
        })

    return results

def create_response2(y):
    y_new = np.where((y['d100agvhd24'] == 1) & 
                     (y['d100agvhd34'] == 1), 1, 0)
    return y_new

def create_response3(y):
    y_new = np.where((y['d100agvhd24'] == 0) & 
                     (y['d100agvhd34'] == 0), 0,
                     np.where((y['d100agvhd24'] == 1) & 
                              (y['d100agvhd34'] == 1), 2, 1))
    return y_new

results = []

if __name__ == '__main__':
    # Read data
    Path_datafile = '../Data/ck1901_public_datafile.sas7bdat'
    raw_data = pd.read_sas(Path_datafile, format='sas7bdat')

    # Convert specified columns to categorical
    column_to_convert = ['gvhdgpmod', 'karnofcat', 'donorgpa',
                         'atgyndos', 'drcmvpr_2', 'condescrpmod',
                         'olkstaprgp', 'coorgscoremod']
    raw_data[column_to_convert].astype('category')
    data = raw_data.copy()

    # Drop rows with missing response variables
    Y_cols = ['agvhd', 'd100agvhd24', 'd100agvhd34']
    data.replace(99, np.nan, inplace=True)
    data.dropna(how='any', subset=Y_cols, inplace=True)

    # Create response variables
    data['response_0to2_vs_3to4'] = create_response2(data)
    data['response_0and1_vs_2_vs_3and4'] = create_response3(data)

    # Remove outliers from numeric columns
    numeric_columns = ['yeartx', 'age', 'intdxtx']
    for column in numeric_columns:
        data = remove_outliers(data, column)

    X_cols = ['yeartx', 'age', 'gvhdgpmod', 'karnofcat', 
              'intdxtx', 'donorgpa', 'atgyndos', 'drcmvpr_2',
              'condescrpmod', 'olkstaprgp', 'coorgscoremod']
    response_options = ['d100agvhd24', 'response_0to2_vs_3to4',
                        'response_0and1_vs_2_vs_3and4']
    response_feature_options = {
        'd100agvhd24': [5, 6, 8],
        'response_0to2_vs_3to4': [5, 6, 9],
        'response_0and1_vs_2_vs_3and4': [5, 6, 7]
    }
    splits = 4

    for response_option in response_options:
        print("Response Variable:", response_option)
        print("=" * 50)

        feature_options_response = response_feature_options.get(
            response_option, [])
        print(Counter(data[response_option]))

        for feature in feature_options_response:
            result = split_train_test(data[X_cols],
                                      data[response_option],
                                      feature, response_option, splits)
            avg_results = result.groupby(['Model', 'Response']).mean()

            # Print the table
            table = PrettyTable()
            table.field_names = ['Model'] + list(avg_results.columns)

            for index, row in avg_results.iterrows():
                model = index
                table.add_row([model] + 
                              [f'{value:.2f}' for value in row])

            print(table)
            results.append(result)

    results_table = pd.concat(results, ignore_index=True)

    # Create subplots for each response type
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)

    # Plot box plots for agvhd
    sns.boxplot(data=results_table[
                results_table['Response'] == 'd100agvhd24'],
                y='Balanced Accuracy',
                x='Features', hue='Model', ax=axes[0])
    axes[0].set_title('Response 0 to 1 vs 2 to 4')
    axes[0].set_ylim([0, 1])

    # Plot box plots for response_0to2_vs_3to4
    sns.boxplot(data=results_table[
                results_table['Response'] == 'response_0to2_vs_3to4'],
                y='Balanced Accuracy',
                x='Features', hue='Model', ax=axes[1])
    axes[1].set_title('Response 0 to 2 vs 3 to 4')
    axes[1].set_ylim([0, 1])

    # Plot box plots for response_0and1_vs_2_vs_3and4
    sns.boxplot(data=results_table[
                results_table['Response'] == 
                'response_0and1_vs_2_vs_3and4'],
                y='Balanced Accuracy',
                x='Features', hue='Model', ax=axes[2])
    axes[2].set_title('Response 0 and 1 vs 2 vs 3 and 4')
    axes[2].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.show()
