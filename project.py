#!/usr/bin/env python
# coding: utf-8

# ## Project : Classification
# 
# #### Due Date: Saturday May 8, 2020 at 11:59PM EDT
# 
# We want to develop a model that can classify spam emails from non-spam emails. Note that spam means junk, commercial or bulk emails. We will get practice with
# 
# - Encoding text as numbers 
# - Fitting models to data
# - Evaluating predictions with metrics 
# 
# You will have the chance to test your own model in a [Kaggle competition](https://www.kaggle.com/c/mg-gy-8413-business-analytics-spring-2021).  
# 
# The questions guide you step-by-step through the assignment. Please post to Slack with any questions. 
# 
# #### Collaboration Policy
# 
# Data analysis is a collaborative activity. While you may discuss the homework with classmates, you should answer the questions by yourself. If you discuss the assignments with other students, then please **include their names** below.
# 

# **Name:** *list name here*

# **NetId:** *list netid here*

# **Collaborators:** *list names here*

# 
# 
# ### Rubric
# Question | Points
# --- | ---
# 1.1 | 1
# 1.2 | 1
# 1.3 | 1
# 1.4 | 0
# 2.1 | 3
# 3.1 | 2
# 3.2 | 2
# 4.1 | 1
# 5.1 | 2
# 6.1 | 1
# 6.2 | 1
# 6.3 | 2
# 6.4 | 2
# 6.5 | 1
# 7.1 | 2
# 8.1 | 2
# 8.2 | 3
# Total | 29

# In[296]:


import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

# Set some parameters in the packages 

plt.rcParams['figure.figsize'] = (9,7)
pd.options.display.max_rows = 20  
pd.options.display.max_columns = 15

# specify path to files

import os, sys
from IPython.display import Image

home_path = os.environ["HOME"]
training_set_path = f'{home_path}/shared/project/train.csv' 
testing_set_path = f'{home_path}/shared/project/test.csv'


# In[220]:


# TEST 

assert 'pandas' in sys.modules and "pd" in locals()
assert 'numpy' in sys.modules and "np" in locals()
assert 'matplotlib' in sys.modules and "plt" in locals()
assert 'sklearn' in sys.modules and "LogisticRegression" in locals()


# ### Question 1
# 
# In email classification, our goal is to classify emails as spam or not spam (referred to as "ham") using features generated from the text in the email. 
# 
# The dataset consists of email messages and their labels (0 for ham, 1 for spam). Your labeled training dataset contains 8348 labeled examples, and the test set contains 1000 unlabeled examples.
# 
# Run the following cells to load in the data into DataFrames.
# 
# The `train` DataFrame contains labeled data that you will use to train your model. It contains four columns:
# 
# 1. `id`: An identifier for the training example
# 1. `subject`: The subject of the email
# 1. `email`: The text of the email
# 1. `spam`: 1 if the email is spam, 0 if the email is ham (not spam)
# 
# The `test` DataFrame contains 1000 unlabeled emails. You will predict labels for these emails.

# In[221]:


original_training_data = pd.read_csv(training_set_path)


# In[222]:


pd.read_csv(testing_set_path)


# We should convert the text to lowercase letters.

# In[223]:


original_training_data['email'] = original_training_data['email'].str.lower()
original_training_data


# #### Question 1.1
# 
# First, let's check if our data contains any missing values. Fill in the cell below to print the number of NaN values in each column. If there are NaN values, replace them with appropriate filler values (i.e., NaN values in the `subject` or `email` columns should be replaced with empty strings). Print the number of NaN values in each column after this modification to verify that there are no NaN values left.
# 
# Note that while there are no NaN values in the `spam` column, we should be careful when replacing NaN labels. Doing so without consideration may introduce significant bias into our model when fitting.

# In[224]:


print('Before Filling:')
print(original_training_data.isnull().sum())

original_training_data['subject'] = original_training_data['subject'].fillna('')# fill missing values with empty string




print('------------')
print('After Filling:')
print(original_training_data.isnull().sum())


# In[225]:


# TEST

assert original_training_data.isnull().sum().sum() == 0


# In[226]:


original_training_data


# In[227]:


ham = original_training_data[(original_training_data["spam"] == 0)]
ham


# In[228]:


ham.iloc[0,2]


# In[229]:


Spam = original_training_data[(original_training_data["spam"] == 1)]
Spam


# In[230]:


original_training_data[(original_training_data["spam"] == 1)].iloc[3,2]


# #### Question 1.2
# 
# Print the text of the zeroth ham and the third spam email from the original training set. You need to filter with the conditions 
# 
# - `original_training_data['spam'] == 0`
# - `original_training_data['spam'] == 1`
# 
# You need to access the row with **index 0** or the row with **index 3**.

# In[231]:


zero_ham = original_training_data[(original_training_data["spam"] == 0)].iloc[0,2]
fourth_spam = original_training_data[(original_training_data["spam"] == 1)].iloc[3,2]



print("Ham \n", zero_ham)
print("Spam \n", fourth_spam)


# In[232]:


# TEST
assert len(zero_ham) > 0 and zero_ham[:0] == ''
assert len(fourth_spam) > 0 and fourth_spam[:0] == ''


# #### Question 1.3
# 
# Below we have four features of the emails. Among the choices, select those features indicative of spam.
# 
# 1. HTML tags within brackets < and >
# 1. Dollar sign $ about money
# 1. URL indicating website .com 
# 1. Salutation with the word dear
# 
# Each choice should appear in the spam email but not the ham email.

# In[233]:


q1c = [1,2,4]


# In[234]:


# TEST 

assert set(q1c).issubset({1,2,3,4})


# #### Question 1.4
# 
# The training data is available for both training models and **validating** the models that we train.  We therefore need to split the training data into separate training and validation datsets.  You will need this **validation data** to assess the performance of your classifier once you are finished training. Note that we set the seed (random_state) to 42. This will produce a "random" sequence of numbers that is the same for every student. Do not modify this in the following questions, as our tests depend on this random seed.

# In[235]:


training_set, validation_set = train_test_split(original_training_data, test_size=0.1, random_state=42)


# In[236]:


# TEST 

assert len(training_set) == 7513
assert len(validation_set) == 835


# ### Question 2
# 
# We would like to take the text of an email and predict whether the email is ham or spam. This is a *classification* problem, so we can use logistic regression to train a classifier. Recall that to train an logistic regression model we need a numeric feature matrix $X$ and a vector of corresponding binary labels $y$.  
# 
# However the data is text, not numbers. To address this, we can create numeric features derived from the email text and use those features for logistic regression.
# 
# Each row of $X$ is an email. Each column of $X$ contains one feature for all the emails. We'll guide you through creating a simple feature, and you'll create more interesting ones when you are trying to increase your accuracy.

# #### Question 2.1
# 
# Create a function called `words_in_texts` that takes in a list of `words` and a pandas Series of email `texts`. It should output a 2-dimensional NumPy array containing one row for each email text. The row should contain either a 0 or a 1 for each word in the list: 0 if the word doesn't appear in the text and 1 if the word does. For example:
# 
# ```
# >>> words_in_texts(['hello', 'bye', 'world'], 
#                    pd.Series(['hello', 'hello worldhello']))
# 
# array([[1, 0, 0],
#        [1, 0, 1]])
# ```

# In[237]:


def words_in_texts(words, texts):
    '''
    Inputs:
        words (list-like): words to find
        texts (Series): strings to search in
    
    Output:
        NumPy array of 0s and 1s with shape (n, p) where n is the
        number of texts and p is the number of words.
    '''
    indicator_ary = np.array([[word in text for word in words] for text in texts]).astype(int)
    return indicator_ary
 
                
words_in_texts(['hello', 'bye', 'world'], 
                           pd.Series(['hello', 'hello worldhello']))


# In[238]:


# TEST
assert np.allclose(words_in_texts(['hello', 'bye', 'world'], 
                           pd.Series(['hello', 'hello worldhello'])),
            np.array([[1, 0, 0], 
                      [1, 0, 1]])) == True


# ### Question 3
# We need to identify some features that allow us to distinguish spam emails from ham emails. One idea is to compare the distribution of a single feature in spam emails to the distribution of the same feature in ham emails. If the feature is whether a certain word occurs in the text, this amounts to comparing the proportion of spam emails with the word to the proportion of ham emails with the word.
# 
# The following plot (which was created using `sns.barplot`) compares the proportion of emails in each class containing a particular set of words. 

# In[239]:


Image(f"{home_path}/shared/project/training_conditional_proportions.png")


# To generate the bar-chart we will use the `melt` function in `pandas`. For example, suppose we had the following data. 

# In[240]:


df = pd.DataFrame({
    'word_1': [1, 0, 1, 0],
    'word_2': [0, 1, 0, 1],
    'type': ['spam', 'ham', 'ham', 'ham']
})
display(df)


# With melt we switch from a "wide" format to a "long" format in the table.

# In[241]:


df.melt("type")


# Note that `melt` will turn columns into variable. Here `word_1` and `word_2` become `variable`. Their values are stored in the value column.
# 
# #### Question 3.1
# 
# We want to create a bar chart like the one above comparing the proportion of spam and ham emails containing certain words. We will take the words to be 
# 
# - `opportunity`
# - `bank`
# - `receive`
# - `dear`
# - `best`
# - `deal`
# 
# We need three steps. For Step 1, we encode the words as 0 and 1 with the function from Question 2.

# In[242]:


# Step 1 encode the words as 1 and 0

training_set = training_set.reset_index(drop=True) 

vocabulary = ['body', 'html', 'please', 'money', 'business', 'offer']
encoded_words = words_in_texts(vocabulary, training_set['email'])

encoded_table = pd.DataFrame(data = encoded_words, columns = vocabulary)
encoded_table['label'] = training_set['spam'].replace({0:"ham", 1: "spam"})

encoded_table


# Moreover we changed 0 and 1 to "ham" and "spam". For Step 2, we use melt to generate the data for the bar-chart

# In[243]:


encoded_table_melted = encoded_table.melt('label')
display(encoded_table_melted)

encoded_table_melted_groups = encoded_table_melted.groupby(['label', 'variable']).mean()
encoded_table_melted_groups = encoded_table_melted_groups.reset_index()
encoded_table_melted_groups


# For Step 3, use the table `encoded_table_melted_groups` to generate a bar-chart with `sns.barplot`.

# In[244]:


# YOUR CODE HERE
#raise NotImplementedError()
sns.barplot(x = 'variable', y = 'value', hue = 'label', data = encoded_table_melted_groups )

plt.ylim([0, 1])
plt.xlabel('Words')
plt.ylabel('Proportion of Emails')
plt.legend(title = "")
plt.title("Frequency of Words in Spam/Ham Emails")

q3a_gca = plt.gca();


# In[245]:


# TEST 

heights = [patch.get_height() for patch in q3a_gca.get_children() if isinstance(patch, matplotlib.patches.Rectangle)]

assert set(encoded_table_melted_groups["value"].values).issubset(set(heights))


# When the feature takes two values, it makes sense to compare its proportions across classes (as in the previous question). Otherwise, if the feature can take on numeric values, we can compare the distributions of these values for different classes. 
# 

# #### Question 3.2
# 
# The length of emails might indicate spam or ham. Below we have a chart comparing the distribution of the length of spam emails to the distribution of the length of ham emails in the training set. 

# In[246]:


Image(f"{home_path}/shared/project/training_conditional_densities2.png")


# In[247]:


training_set


# Make a copy of the training set. Add a column to the copy called `length` that has the number of characters in the email.

# In[248]:


training_set_copy = training_set.copy()

rest = []
for i in training_set_copy['email']:
    rest.append(len(i))
training_set_copy['length'] = rest 
training_set_copy['length']


# In[249]:


# TEST 

assert 'length' in training_set_copy.columns


# Now we can generate the chart.

# In[250]:


sns.distplot(training_set_copy.loc[training_set_copy['spam'] == 0, 'length'],hist=False, label='Ham')
sns.distplot(training_set_copy.loc[training_set_copy['spam'] == 1, 'length'],hist=False, label='Spam')
plt.xlabel('Length of email body')
plt.ylabel('Distribution')
plt.xlim((0,50000))
plt.grid();


# ### Question 4
# 
# Notice that the output of `words_in_texts(words, train['email'])` is a numeric matrix containing features for each email. This means we can use it directly to train a model.

# #### Question 4.1
# 
# We've given you 5 words that might be useful as features to distinguish spam/ham emails. Use these words as well as the `training_set` table to create two NumPy arrays: `X_train` and `Y_train`.
# 
# - `X_train` should be a 2-dimensional array of 0s and 1s created by using your `words_in_texts` function on all the emails in the training set.
# - `Y_train` should be a 1-dimensional array of the correct labels for each email in the training set.

# In[251]:


some_words = ['drug', 'bank', 'prescription', 'memo', 'private']

X_train = np.array(words_in_texts(some_words, training_set['email'])).astype(int)
Y_train = np.array(training_set['spam'])

X_train
Y_train


# In[252]:


# TEST
assert X_train.shape == (7513, 5) # X matrix should have a certain size
assert np.all(np.unique(X_train) == np.array([0, 1])) # X matrix should consist of only 0 or 1


# ### Question 5
# 
# We will use [`LogisticRegression`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) in the scikit-learn package for logistic regression.
# 
# #### Question 5.1
# 
# Here we will train a model with `X_train` and `Y_train`. Use the `fit` function to fit the model to the data. 

# In[253]:


model = LogisticRegression()

model.fit(X_train,Y_train)


# Now we can compute the accuracy of the model on the training data set. You should get an accuracy around 0.75.

# In[254]:


# TEST

training_accuracy = model.score(X_train, Y_train)
assert training_accuracy > 0.72


# In[255]:


training_accuracy


# ### Question 6
# 
# While we have high accuracy, the model has some shortcomings. We are evaluating accuracy on the training set. The in-sample accuracy is misleading because we determined the model from the training set. We need to think about the out-of-sample accuracy.
# 
# Remember that our model would will be used for preventing messages labeled `spam` from reaching someone's inbox. There are two kinds of errors we can make:
# - False positive (FP): a ham email gets flagged as spam and filtered out of the inbox.
# - False negative (FN): a spam email gets mislabeled as ham and ends up in the inbox.
# 
# These definitions depend both on the true labels and the predicted labels. False positives and false negatives may be of differing importance, leading us to consider more ways of evaluating a classifier, in addition to overall accuracy:
# 
# - **Precision** measures the proportion $\frac{\text{TP}}{\text{TP} + \text{FP}}$ of emails flagged as spam that are actually spam.
# 
# - **Recall** measures the proportion $\frac{\text{TP}}{\text{TP} + \text{FN}}$ of spam emails that were correctly flagged as spam. 
# 
# - **False-positive rate** measures the proportion $\frac{\text{FP}}{\text{FP} + \text{TN}}$ of ham emails that were incorrectly flagged as spam. 
# 
# Note that a true positive (TP) is a spam email that is classified as spam, and a true negative (TN) is a ham email that is classified as ham.
# 
# ### Question 6.1
# 
# Suppose we have a classifier `zero_predictor` that always predicts 0 (never predicts positive). How many false positives and false negatives would this classifier have if it were evaluated on the training set and its results were compared to `Y_train`? Fill in the variables below.

# In[256]:


spam = 0 
for i in Y_train: 
    if i == 1: 
        spam += 1 
spam


# In[257]:


zero_predictor_fp = 0
zero_predictor_fn = 1918


# In[258]:


# TEST
assert zero_predictor_fp >= 0
assert zero_predictor_fn >= 0


# In[259]:


TP = 0 
TN = sum(Y_train == 0)
TN


# #### Question 6.2
# 
# What are the accuracy and recall of `zero_predictor` (classifies every email as ham) on the training set? Do NOT use any `sklearn` functions.

# In[260]:


zero_predictor_acc = TN/len(Y_train)
zero_predictor_recall = 0

zero_predictor_acc


# In[261]:


# TEST
assert zero_predictor_acc >= 0
assert zero_predictor_recall >= 0


# We learn from Question 6a and Question 6b that for `zero_predictor`:
# 
# - There are no false positives ($\text{FP} = 0$) because nothing is labeled spam. 
# - Every spam email is mislabeled as ham, so the number of false negatives is equal to the number of spam emails in the training data ($\text{FN} = 1918$).
# - The classifier correctly labels nearly 75% of observations in the training data.
# - The classifier recalls none (0%) of the spam observations.
# 

# #### Question 6.3
# 
# Consider the the `LogisticRegression` model from Question 5. Without using any `sklearn` functions, compute the precision, recall, and false-alarm rate of on the training set.
# 

# In[262]:


Y_train_hat = model.predict(X_train)
ob = list(Y_train)
pd = list(Y_train_hat)
TN = 0 
FP = 0 
TP = 0
FN = 0 
for i in range(len(ob)):
    if ob[i] == 0 and pd[i] == 0:
        TN += 1 
    if ob[i] == 0 and pd[i] == 1:
        FP += 1 
    if ob[i] == 1 and pd[i] == 1:
        TP += 1 
    if ob[i] == 1 and pd[i] == 0: 
        FN  += 1


# In[263]:


(TP + TN) / len(Y_train)


# From the numbers, we can calculate the metrics.

# In[264]:


logistic_predictor_precision = TP / (TP + FP) 
logistic_predictor_recall = TP / (TP + FN) 
logistic_predictor_fpr = FP / (FP + TN) 


# In[265]:


# TEST
assert logistic_predictor_precision >= 0
assert logistic_predictor_recall >= 0
assert logistic_predictor_fpr >= 0


# In[266]:


logistic_predictor_precision


# In[267]:


logistic_predictor_recall


# In[268]:


logistic_predictor_fpr


# In[269]:


validation_set


# #### Question 6.4
# 
# Without using any `sklearn` functions, compute the precision, recall, and false-alarm rate of on the validation set.

# In[270]:


X_val = words_in_texts(some_words, validation_set['email']) 
Y_val = np.array(validation_set['spam']) 
Y_val_hat = model.predict(X_val)
ob = list(Y_val)
pd = list(Y_val_hat)
TN = 0 
FP = 0 
TP = 0
FN = 0 
for i in range(len(ob)):
    if ob[i] == 0 and pd[i] == 0:
        TN += 1 
    if ob[i] == 0 and pd[i] == 1:
        FP += 1 
    if ob[i] == 1 and pd[i] == 1:
        TP += 1 
    if ob[i] == 1 and pd[i] == 0: 
        FN  += 1


# In[271]:


(TP + TN) / len(ob)


# From the numbers, we can calculate the metrics.

# In[272]:


logistic_predictor_precision_validation = TP / (TP + FP) 
logistic_predictor_recall_validation = TP / (TP + FN) 
logistic_predictor_fpr_validation = FP / (FP + TN) 


# In[273]:



logistic_predictor_fpr_validation 


# In[274]:


logistic_predictor_precision_validation


# In[275]:


logistic_predictor_recall_validation


# In[276]:


# TEST
assert logistic_predictor_precision_validation >= 0
assert logistic_predictor_recall_validation >= 0
assert logistic_predictor_fpr_validation >= 0


# #### Question 6.5
# 
# We can visualize these numbers on the validation set with a confusion matrix.

# In[277]:


def plot_confusion(confusion):
    sns.heatmap(confusion, annot=True, fmt='d',
                cmap="Purples", annot_kws={'fontsize': 24}, square=True,
                xticklabels=[1, 0], yticklabels=[1, 0], cbar=False)
    plt.gca().xaxis.set_label_position('top')
    plt.xlabel('True')
    plt.ylabel('Predicted')

confusion = np.array([
    [TP, FP],
    [FN, TN],
])
    
plot_confusion(confusion)


# **True or False**: There are equal number of True Positives and True Negatives when using the logistic regression classifier from Question 5?
# 

# In[278]:


q6e = False 


# In[279]:


# TEST 

assert q6e in [True, False]


# We can make a few observations about the accuracy of the model.
# 
# 1. Our logistic regression classifier got 75.6% prediction accuracy. Remember accuracy is number of correct predictions / total. How does this compare with predicting 0 for every email?
#  * An accuracy of 75% means that we're only doing slightly better than guessing ham for every email.
# 1. What is a problem with the features in the model?
#  *  The encoded features in `X_train` have many rows with all 0. That is, the words we've chosen as our features aren't actually present in many of the emails so the classifier can't use them to distinguish between ham/spam emails.
# 1. Which of these two classifiers would you prefer for a spam filter and why?
#  * The false-alarm rate for logistic regression is too high, even at 2%: ideally false-alarms would almost never happen. I might rather wade through thousands of spam emails than get 2% of legitimate emails filtered out of my inbox."""
# 

# ### Question 7
# 
# We can balance precision and recall. Usually we cannot get both precision equal to 1 (i.e. no false positives) and recall equal to 1 (i.e. no false negatives). 
# 
# Remember that logistic regression calculates the probability that an example belongs to a particular category. 

# In[280]:


Y_val_hat_prob = model.predict_proba(X_val)[:, 1]


# #### Question 7.1
# 
# Here we use the `predict_proba` function to predict a probability. 
# 
# From a probability, we can determine a category by comparison with a threshold. With a threshold of 0.5, we label an email as spam for a predicted probability $\geq 0.5$. However, we could take the threshold to be $0.25$ or $0.75$. Adjusting the threshold allows us balance false positives and false negatives.
# 
# The [Receiver Operating Characteristic](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html) metric compares true positives and false positives for each possible cutoff probability. In the cell below, we plot the value on the validation set.
# 

# In[281]:


fpr, tpr, threshold = roc_curve(Y_val, Y_val_hat_prob)

threshold_values = [5,8,10]

for idx, x,y in zip([0,1,2], fpr[threshold_values],tpr[threshold_values]):
    plt.plot(x,y, "ro", zorder = 10)
    plt.text(x, y + 0.05, str(idx), fontdict={"color":"red", "size":15})

plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC Curve')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right");


# We have labeled three points. These points correspond to thresholds in the set 0.2, 0.3, and 0.6. Assign the thresholds to the points.

# In[282]:


q7 = [0.6,0.3,0.2]

# YOUR CODE HERE


# In[283]:


# TEST 

assert type(q7) == list
assert set(q7).issubset({0.6, 0.3, 0.2})


# ### Question 8
# 
# Take the following function for computing the accuracy of classifications.
# 

# In[284]:


def accuracy(y_pred, y_actual):
    return np.mean(y_pred == y_actual)


# We want to perform cross validation to compare different choices of words for the classfication. By training and validating multiple times, we can gauge the accuracy and consistency of the classfications.

# In[285]:


def compute_CV_accs(model, X_train, Y_train, vocabulary, number_splits):
    kf = KFold(n_splits=number_splits, shuffle=True, random_state=42)
    
    vocabulary_accs = dict()
    for words in vocabulary:
        X_train_features = words_in_texts(words, X_train) 

        validation_accs = []        
        for train_idx, valid_idx in kf.split(X_train):
            # split the data
            split_X_train, split_X_valid = X_train_features[train_idx], X_train_features[valid_idx]
            split_Y_train, split_Y_valid = Y_train.iloc[train_idx], Y_train.iloc[valid_idx]

            # Fit the model on the training split
            model.fit(split_X_train,split_Y_train)

            # Compute the accuracy on the validation split
            acc = accuracy(model.predict(split_X_valid), split_Y_valid)

            validation_accs.append(acc)
    
        #average validation accuracies
        print("For vocabulary {0}".format(",".join(words)), "\n Mean: {0}".format(np.mean(validation_accs)), "\n Standard Deviation {0}\n\n".format(np.std(validation_accs)))
        
        vocabulary_accs[tuple(words)] = {'mean': np.mean(validation_accs), 'std': np.std(validation_accs)}
        
    return vocabulary_accs


# #### Question 8.1
# Consider the collection of words `vocabulary1` and `vocabulary2`

# In[286]:


vocabulary1 = ['drug', 'bank', 'prescription', 'memo', 'private']
vocabulary2 = ['please', 'money', 'offer', 'receive', 'contact', 'free']


# We want to perform 5-fold cross validation with `compute_CV_accs`. Run the function on `original_training_data` with `LogisticRegression` model for `vocabulary1` and `vocabulary2`. Call the output `vocabulary_accs`.

# In[287]:


model = LogisticRegression()
X_train = original_training_data['email']
Y_train =  original_training_data['spam']
vocabulary = [vocabulary1, vocabulary2]
number_splits = 5


vocabulary_accs = compute_CV_accs(model, X_train, Y_train, vocabulary, number_splits)

vocabulary_accs


# In[288]:


# TEST 
assert np.isclose(vocabulary_accs[tuple(vocabulary1)]['mean'], 0.7557479648252925)
assert np.isclose(vocabulary_accs[tuple(vocabulary1)]['std'], 0.012498431306465675)


# #### Question 8.2
# 
# Which collection of words is more accurate?

# In[289]:


q8b_1 = "vocabulary2"


# In[290]:


# TEST 

assert q8b_1 in ["vocabulary1", "vocabulary2"]


# Which collection of words is more consistent? 

# In[291]:


q8b_2 = "vocabulary2"


# In[292]:


# TEST 

assert q8b_2 in ["vocabulary1", "vocabulary2"]


# Which should you choose for determining the features of your model?

# In[293]:


q8b_3 = "vocabulary2"


# In[294]:


# TEST 

assert q8b_3 in ["vocabulary1", "vocabulary2"]


# ### Question 9 
# 
# Google has a platform called Kaggle for hosting modeling competitions. We have configured a competition with the testing data. 

# In[297]:


testing_set = pd.read_csv(testing_set_path)
testing_set['email'] = testing_set['email'].str.lower()
testing_set


# For example, if we want to submit the model from Question 8 with `vocabulary2` then we would make output the predictions to a csv file.

# In[298]:


vocabulary2 = ['please', 'money', 'offer', 'receive', 'contact', 'free']

X_train = words_in_texts(vocabulary2, original_training_data['email'])
Y_train = original_training_data['spam']


model = LogisticRegression()
model.fit(X_train, Y_train)

X_test = words_in_texts(vocabulary2, testing_set['email'])
Y_test_hat = model.predict(X_test)

testing_set_predictions = pd.DataFrame(data = Y_test_hat, columns = ["Category"])
testing_set_predictions


# Now we can output to a csv file for submission to the Kaggle website (https://www.kaggle.com/c/mg-gy-8413-business-analytics-spring-2021)

# In[299]:


testing_set_predictions.to_csv("Kaggle_submission.csv", index = True, index_label="Id")


# In[300]:


vocabulary_hat = ['body', 'html', 'please', 'money', 'remove', 'offer','!','wish']

X_train1 = words_in_texts(vocabulary_hat, original_training_data['email'])
Y_train2 = original_training_data['spam']


model = LogisticRegression()
model.fit(X_train1, Y_train2)

X_test1 = words_in_texts(vocabulary_hat, testing_set['email'])
Y_test_hat1 = model.predict(X_test1)

testing_set_predictions1 = pd.DataFrame(data = Y_test_hat1, columns = ["Category"])
testing_set_predictions1.to_csv("Kaggle_submission.csv", index = True, index_label="Id")


# Can you make the spam filter more accurate? Try to get at least **88%** accuracy on the test set. Call your predictions `Y_test_hat`. This should be a numpy array consisting of 0 and 1 for each every email in the `testing_set` table. 
# 
# 
# Here are some ideas for improving your model:
# 
# 1. Finding better features based on the email text. Some example features are:
#     1. Number of characters in the subject / body
#     1. Number of words in the subject / body
#     1. Use of punctuation (e.g., how many '!' were there?)
#     1. Number / percentage of capital letters 
#     1. Whether the email is a reply to an earlier email or a forwarded email
# 1. Finding better words to use as features. Which words are the best at distinguishing emails? This requires digging into the email text itself. 
# 1. Better data processing. For example, many emails contain HTML as well as text. You can consider extracting out the text from the HTML to help you find better words. Or, you can match HTML tags themselves, or even some combination of the two.
# 1. Model selection. You can adjust parameters of your model (e.g. the regularization parameter) to achieve higher accuracy. Recall that you should use cross-validation to do feature and model selection properly! Otherwise, you will likely overfit to your training data.
