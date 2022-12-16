##########################
# Globals
##########################
# imports needed for functions
import pandas as pd
import numpy as np

# testing
from sklearn import metrics
from sklearn.metrics import f1_score
    
# plotting
import seaborn as sns
from matplotlib import pyplot as plt

print('Imports loaded.')

# separate dataframe into features and labels
def get_features_and_labels(df):
    feature_data = df.iloc[:,:-1]
    label_data = df.iloc[:,-1]

    return feature_data, label_data

# dict for converting data to right class integer code
types_index = {
    0: 1,
    1: 2,
    2: 3,
    3: 5,
    4: 6,
    5: 7 
}

# plotting
def plot_cf_matrix(cf_matrix, title):
    labels = types_index.values()
    cmap = sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)

    ax = sns.heatmap(cf_matrix, annot=True, cmap=cmap)
    ax.set_title(title + '\n')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values')

    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)

    plt.show()

# min max normalisation
def min_max_normalisation(df):
    norm_df = (df-df.min())/(df.max()-df.min())
    return norm_df

##########################
# Decision Tree Functions
##########################

# helper functions
def count_values(label_data, obs):
    '''
    Function to count the number of observations per label in a node
    '''
    labels = label_data.iloc[obs]
    labels_unique = list(np.unique(label_data))
    labels
    count = dict()

    for label in labels_unique:
        count[label] = 0

    for label in labels:
        count[label] = count[label] + 1

    return list(count.values())

# functions to calculate the split
def gini(label_data, obs):
    '''
    Calculates gini for each node.
    '''
    class_count = list(label_data.iloc[obs].value_counts())
    # class_count = count_values(label_data, obs)
    gini = 0
    for i in range(len(class_count)):
        prob_class = class_count[i]/sum(class_count)
        gini += prob_class * (1-prob_class)

    return gini

def split_node(col, obs, split_val):
    '''
    Splits nodes on a specified split value.
    '''
    small = []
    big = []

    for i in obs:
        # print(i, col)
        if col[i] <= split_val:
            small.append(i)
        else:
            big.append(i)

    return small, big

def find_best_split_feature(col, obs, label_data):
    '''Finds the best split for each feature of the dataset.
    Input: col is the feature vector, obs are the indexes for observations in the node, label_data is the vector of true values 
    Output: best_split, best_gini 
    '''
    best_gini_feature = 1
    best_split_feature = 0
    # second_best = 0

    # go through only unique values
    uniq_vals = np.sort(col.unique())

    for val in uniq_vals:
        current_split = val

        small, big = split_node(col, obs, val)
        gini_small = gini(label_data, small)
        gini_big = gini(label_data, big)
        n = len(small) + len(big)

        w_gini = ((len(small) / n) * gini_small) + ((len(big) / n) * gini_big)

        if w_gini < best_gini_feature:
            best_gini_feature = w_gini
            best_split_feature = current_split

    # get avarage for best split
    next_split = list(uniq_vals).index(best_split_feature)+1
    if next_split < len(uniq_vals):
        best_split_feature = (best_split_feature + uniq_vals[next_split])/2

        # print(best_split_feature, best_gini_feature)

    return best_split_feature, best_gini_feature

def find_best_split(feature_data, label_data, obs):
    '''
    Find best split of the full dataset.
    '''
    df = feature_data.iloc[obs,:]

    best_feature = 0
    best_gini = 1
    best_split = 0
    
    for i in range(len(df.columns)):
        col = df.iloc[:,i]
        f_split, f_gini = find_best_split_feature(col, obs, label_data)
        
        if f_gini < best_gini:
            best_gini = f_gini
            best_split = f_split
            best_feature = i
            
    return best_feature, best_gini, best_split

# main functions
# some hardcoding in build_tree -> type_index dict needed to calculate class
def build_tree(max_depth, feature_data, label_data, obs, curr_step=0):
    '''
    Build tree from dataframe
    '''
    features_dict = dict(enumerate(list(feature_data.columns)))
    
    node_dict = dict()

    gini_node = gini(label_data, obs)

    best_feature, best_gini, best_split = find_best_split(feature_data, label_data, obs)
    left, right = split_node(feature_data.iloc[:,best_feature], obs, best_split)

    value_list = count_values(label_data, obs)

    # node_dict['split_num'] = curr_step
    node_dict['feature'] = features_dict[best_feature]
    node_dict['threshold'] = round(best_split, 3)
    node_dict['gini_node'] = round(gini_node, 3)
    node_dict['values'] = str(value_list)
    node_dict['samples'] = sum(value_list)
    node_dict['class'] = types_index[value_list.index(max(value_list))] 

    if curr_step < max_depth and gini_node > 0:
        node_dict['left'] = build_tree(max_depth, feature_data, label_data, left, curr_step+1)
        node_dict['right'] = build_tree(max_depth, feature_data, label_data, right, curr_step+1)

    return node_dict

def predict(decision_node, observation): 
    feature = decision_node["feature"]
    threshold = decision_node["threshold"]
    feature_value = observation[feature]

    direction = "left" if feature_value <= threshold else "right"
    if direction in decision_node:
        return predict(decision_node[direction], observation)
    else:
        return decision_node["class"]

# callable functions for training and testing
def decision_tree_train(feature_data, label_data, max_depth):
    '''Build the decision tree using training data.

    Inputs:
    feature_data -> dataframe with all observations and features
    label_data -> dataframe with true labels
    max_depth -> maximum tree depth

    Outputs:
    tree_dict -> the decision tree as a nested dictionary
    '''
    max_depth = max_depth

    obs_root = feature_data.index.values

    tree_dict = build_tree(max_depth, feature_data, label_data, obs_root)

    return tree_dict

def decision_tree_predict(feature_data, tree_dict):
    '''Uses the decision tree to predict classes for test data.
    
    Inputs:
    feature_data -> dataframe with all observations and features
    tree_dict -> the decision tree as a nested dictionary

    Outputs:
    predictions -> the vector of predictions as a list
    '''
    predictions = []
    for i in range(feature_data.shape[0]):
        row = feature_data.iloc[i,:]
        predictions.append(predict(tree_dict, row))

    return predictions

# printing the tree functions
# print statements
def print_tree(tree_dict, string='', direction=''): 
    feature = tree_dict["feature"]
    threshold = round(tree_dict["threshold"], 2)
    gini = round(tree_dict["gini_node"], 2)
    values = tree_dict["values"]
    class_node = tree_dict["class"]

    len_str = len(string) * " "

    # print(f'{string}Split {int(len(string)/4)}{direction}\n{string}Best split: {feature} <= {threshold}\n{len_str}gini = {gini}\n{len_str}values = {values}\n{len_str}class = {class_node}')
    print(f'{string}{feature} <= {threshold}\n{len_str}gini = {gini}\n{len_str}values = {values}\n{len_str}class = {class_node}')

    if 'left' in tree_dict:
        # tree_dict_print['left'] = tree_dict['left']['feature']
        # print(f'{string}{feature} <= {threshold}\n{len_str}gini = {gini}\n{len_str}values = {values}')
        print_tree(tree_dict['left'], string + '--> ', ' left')
    # else:
    #     print(f'{string}gini = {gini}\n{len_str}values = {values}\n{len_str}class = {class_node}')

    if 'right' in tree_dict:
        # print(f'{string}{feature} <= {threshold}\n{len_str}gini = {gini}\n{len_str}values = {values}')
        print_tree(tree_dict['right'], string + '--> ', ' right')
    # else:
    #     print(f'{string}gini = {gini}\n{len_str}values = {values}\n{len_str}class = {class_node}')

# make dict
test = dict()
def print_tree_dict(tree_dict, string=0): 
    feature = tree_dict["feature"]
    threshold = round(tree_dict["threshold"], 2)
    gini = round(tree_dict["gini_node"], 2)
    values = tree_dict["values"]
    class_node = tree_dict["class"]

    tree_dict_print = dict()

    tree_dict_print['split_num'] = string
    # tree_dict_print['feature'] = feature
    # tree_dict_print['threshold'] = threshold
    tree_dict_print['gini'] = gini
    tree_dict_print['values'] = values
    tree_dict_print['class'] = class_node

    if 'left' in tree_dict:
        # tree_dict_print['left'] = f'{tree_dict["left"]["feature"]} <= {round(tree_dict["left"]["threshold"], 2)}'
        print_tree_dict(tree_dict['left'], string + 1)
    if 'right' in tree_dict:
        # tree_dict_print['right'] = f'{tree_dict["right"]["feature"]} <= {round(tree_dict["right"]["threshold"], 2)}'
        print_tree_dict(tree_dict['right'], string + 1)

    node_name = f'{feature} <= {threshold}'
    test[node_name] = tree_dict_print

print('Decision tree functions loaded.')

##########################
# Neural Network Functions
##########################

# activation functions
def relu(x):
    return max(0.0, x)

def sigmoid(x):
    sig = 1 / (1 + np.exp(-x))
    return sig

# vectorised sigmoid
sigm_vec = np.vectorize(sigmoid)

def softmax(h_inp):
    '''
    Input: the vector of predictions -> x * thetas vector 
    Output: probabilities. sum of all probs = 1
    '''
    return  (np.exp(h_inp.T) / np.sum(np.exp(h_inp), axis=1)).T

# callable functions
# neural net train also hard coded with types_index dict
def neural_net_train(feature_data, label_data, learning_rate, epochs):
    n = feature_data.shape[0]
    n_feat = feature_data.shape[1]
    k = len(np.unique(label_data))

    np.random.seed(42) # why use a see - https://towardsdatascience.com/random-seed-numpy-786cf7876a5f
    # initialise weight and bias for 1st layer
    w_1 = np.random.randn(n_feat, n_feat) # weights for first layer
    b_1 = np.zeros((n_feat, 1))

    # initialise weight and bias for 2nd layer
    w_2 = np.random.randn(n_feat, k)
    b_2 = np.zeros((1,k))

    # one-hot encoding
    y_enc = pd.get_dummies(label_data).to_numpy().astype(float)

    preds = 0

    # list of cost
    loss_list = list()

    for epoch in range(epochs):
        # feed forward 
        # layer 1
        a1 = w_1 @ feature_data.T + b_1
        a_1 = sigm_vec(a1) # output of the first layer - 149x9

        # layer 2
        a2 = a_1.T @ w_2 + b_2
        # get prediction vector with softmax
        preds = softmax(a2)

        # calculate loss with cross-entropy
        loss = 0
        for i in range(n):
            for j in range(k):
                loss += y_enc[i][j] * np.log(preds[i][j])

        cross_ent = -1/n*loss
        loss_list.append(cross_ent)

        # back propagation
        # on softmax layer (output)
        d1_1_2 = preds - y_enc
        d1_3 = a_1 # activation from hidden layer
        d1_final = np.dot(d1_3, d1_1_2)

        w_2 -= learning_rate * d1_final
        b_2 -= learning_rate * d1_1_2.sum(axis=0) 

        # on sigmoid layer (hidden)
        d2_1 = np.dot(preds - y_enc, w_2.T)
        d2_2 = a_1 * (1 - a_1)
        d2_3 = feature_data
        d2_final = np.dot(d2_2 * d2_1.T, d2_3)

        w_1 -= learning_rate * d2_final
        b_1 -= learning_rate * np.dot(d2_2, np.sum(d2_1, axis=1).reshape((149,1)))

    # get max value of predictions to get y hat with classes predicted
    # predictions = pd.DataFrame(preds).rename(columns=types_index).idxmax(axis=1).to_numpy()

    return w_1, b_1, w_2, b_2, loss_list#, predictions

# function to test the data also hard coded with types_index dict
def neural_net_predict(w_1, b_1, w_2, b_2, feature_data):
    '''
    Predict unseen data.
    Inputs:
    weights and biases outputted by the neural_net_train function.
    feature_data -> numpy array with all observations and features
    Outputs:
    A vector of predictions
    '''
    # layer 1
    a1 = w_1 @ feature_data.T + b_1
    a_1 = sigm_vec(a1) # output of the first layer - 149x9

    # layer 2
    a2 = a_1.T @ w_2 + b_2
    
    # get prediction vector with softmax
    predictions = softmax(a2)
    predictions = pd.DataFrame(predictions).rename(columns=types_index).idxmax(axis=1).to_numpy()

    return predictions

print('Neural net functions loaded.')