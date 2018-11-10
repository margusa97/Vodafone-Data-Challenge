
# coding: utf-8

# # Vodafone Challenge group X
# ## Comprehensive Notebook
# This notebook includes most of the things we have tried, organized in a nice way. In the file VodafoneChallenge_Clustering file only the necessary parts are taken, based on what worked best on test sets here.
# ## Structure
# 1. Pipeline 
# 2. Issues Encountered
# 3. Miscellaneous
# 4. Code
#     1. Imports
#     2. Cleaning
#     3. Imputation
#     4. Weights Optimization
#     5. Clustering
#     
#     
# ## Pipeline
# * **Cleaning**
#     1. Remove Unnamed: 0 column
#     2. Add 'ID' column
#     3. Fix ZipCodes
#     4. Convert NumericAge column
#     5. Group DeviceOperatingSystem
#     6. Add Rural Information  
#     
#     
# * **Data imputation**
#     1. OperativeSystem. Use RF to infer it using traffic_columns as they are.
#     2. Urbanization. Use MLP to infer it using traffic_columns as they are.
#     3. NumericAge. Group them in 3 categories: 0:10-40, 1:40-80, 2:else. Use Perceptron to infer it using traffic_columns as they are.
#     4. DataAllowance. Remove values < 0.5. Group them into 10 categories: 0-0.1, 0.1-0.2 and so on. Use MLP to infer it using traffic_columns as they are.
#     5. MonthlyDataTraffic. Group them into 10 categories: 0-0.1, 0.1-0.2 and so on. Use RF to infer it using traffic_columns as they are.
#     6. DataARPU. Group them into 10 categories: 0-0.1, 0.1-0.2 and so on. Use Logistic Regression to infer it using traffic_columns as they are.
#     7. MonthlyVoiceTrafficCount. Group them into 10 categories: 0-0.1, 0.1-0.2 and so on. Use Logistic Regression to infer it using traffic_columns as they are.
#     
#     
# * **Weight Optimization**
#     1. Convert OS and Product column from string to numbers
#     1. OneHot encode dataset
#     1. Perform weight optimization on KNN using greedy algorithm
#     2. Perform weight optimization on KNN using Simulated Annealing
#     3. Compare the two and take best weights
#     
#     
# * **Clustering**
#     1. Apply weight from previous step
#     2. Perform K-Means, find optimal number of clusters
#     3. Perform HierarchicalClustering, use as number of neighbors the optimal number of clusters found before
#     4. Compare the two using the labels
#     5. Train RF on the dataset 
#     6. Train MLP on the dataset
#     7. Compare the two on test set and use best to predict probabilities
#     
#     
# ## Issues Encountered
# We will briefly list here the most relevant issues we have encountered. A more throughout explanation of them and the solution we decided to adopt can be found in the relevant section.
# * **Cleaning**
#     1. Problem with ZipCodes, Vodafone dataset had some old ones which have been updated.
#     
#     
# * **Data Imputation**
#     1. Non-categorical features and Linear Regression
#     2. Unbalanced dataset
#     3. Presence of outliers for a given feature (skewed distribution). Ex: DataAllowance
#     
#     
# * **Weight Optimization**
#     1. How to tackle the problem
#     2. Time complexity of the solution, how to reduce it
#     3. Score in the plot not monotonically increasing, how to also consider the overall score
#     
#     
# * **Clustering**
#     1. How to choose optimal number of clusters for K-Means (which metric to use)
#     2. Compare unsupervised algorithms
#     
#     
# ## Miscellaneous
# We devote this section to additional thoughts we might have which didn't really fit anywhere else.
# 
# 1. Why not PCA? We decided not to use PCA to pre-process data for clustering since the interpretability of the features would be lost. That means that even if we did find some good clustering, they would not be useful for drawing any business related conclusion because we can't inspect what a group of customers that the algorithm clustered together have in common.
# 2. Looks like clustering is not really good, according to the statistics of the optimal cluster and the V score measure (v_measure_score) from scikit.metrics. Furthermore, even the supervised algorithms that we trained on the final dataset(RF and MPL) had sensible problems in classifying correctly the data point, the accuracy is quite low. Taken as a whole, this suggests that the data are not informative enough with respect to the classes of product that we need to predict. We have had also a graphical interpretation of this using PCA (even though explained variance wasn't so high), where it was clear that the classes would almost completely overlap one onto the other.
# 
# 
# ## Code

# ### Imports

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering


import time

from VodafoneChallenge_Classes import *

pd.set_option('display.max_columns', None)
np.set_printoptions(threshold=np.NaN)


# ### Cleaning
# The first cleaning block is done following more or less what Baldassi did in class. The second block instead adds urban/rural information to the dataset. The dataset we have used to be able to carry out the conversion, starting from ZipCodes, are the most recent ones, this created problems since some ZipCodes have been updated and Vodafone dataset had the old ones. Theferore we solverd these problems by hand, creating two ad-hoc dictionary (one to move from ZipCodes to CodiceIstat, and the second one from CodiceIstat to urban/rural info). During that phase errors might happen if the procedure is fed new data with old ZipCodes.
# 
# **DeviceOperatingSystem**: We preferred not to create a specific category for 'windows' because too few observations.

# In[3]:


df_clean = df_backup.copy()

del df_clean['Unnamed: 0']

c = list(df_clean.columns)
c[0] = 'ID'
df_clean.columns = c

df_clean['ZipCode'] = df_clean['ZipCode'].map(lambda x: '%05i' % x, na_action='ignore')

traffic_columns = ['File-Transfer', 'Games',
       'Instant-Messaging-Applications', 'Mail', 'Music-Streaming',
       'Network-Operation', 'P2P-Applications', 'Security',
       'Streaming-Applications', 'Terminals', 'Unclassified', 'VoIP',
       'Web-Applications']
df_clean[traffic_columns]

cats = df_clean['CustomerAge'].astype('category').cat.categories
d = {cat:(15+10*i)/100 for i,cat in enumerate(cats)}
df_clean['NumericAge'] = df_clean['CustomerAge'].map(lambda x: d[x], na_action='ignore')

d = {}
for elem in df_clean['DeviceOperatingSystem']:
    d[elem] = d.get(elem, 0) + 1
print(d) #some categories have very few values, group them
OS_other = []
for key in d:
    if d[key] < 10:
        OS_other.append(key)
        d[key] = 'other'
    else:
        d[key] = key
df_clean['OS_clean'] = df_clean['DeviceOperatingSystem'].map(lambda x: d[x], na_action='ignore')


# In[4]:


#Adding rural/urban information
df_zip_istat = pd.read_csv('databases/database.csv')
df_istat_urb = pd.read_csv('databases/it_postal_codes.csv/Foglio 2-Tabella 1.csv', error_bad_lines=False, sep = ';')
my_urb_dict = {'Basso' : 0, 'Medio' : 1, 'Elevato' : 2}
df_istat_urb['GradoUrbaniz'] = df_istat_urb['GradoUrbaniz'].map(lambda x: my_urb_dict[x], na_action = 'ignore')

#check there are no datapoint for which we don't have zip but we've region
df_clean['ZipCode'].isnull()
df_clean['Region'][df_clean['ZipCode'].isnull()]
len(df_clean['Region'][df_clean['ZipCode'].isnull()]) == np.sum(df_clean['Region'][df_clean['ZipCode'].isnull()].isnull())

#we need to insert x for multiple cap cities
isnan = lambda x: x != x
#nan is unique type not equal to itself, so with this lambda function we get True only when the type is NaN

for i in range(df_zip_istat.shape[0]):
    cap = df_zip_istat.loc[i, 'cap/0']
    cap  = '%05d' % cap
    if not isnan(df_zip_istat.loc[i,'cap/1']):
        if not isnan(df_zip_istat.loc[i,'cap/10']):   
            cap = cap[:-2]+'xx'
        else:
            cap = cap[:-1]+'x'
    df_zip_istat.loc[i, 'cap/0'] = cap

d_zip_istat = df_zip_istat.set_index('cap/0').to_dict()['codice']
d_istat_urb = df_istat_urb.set_index('ISTAT').to_dict()['GradoUrbaniz']

mask = df_clean['ZipCode'].isnull()
urban_col = np.zeros(df_clean.shape[0])
urban_col_masked = urban_col[~ mask]
d_zip_istat.update([('51021', 47023),( '83026', 64121),( '74025', 73007),( '55062', 46007),( '38039', 22217),('50037', 48053)])
d_istat_urb.update([(22250, 0),( 78157, 1)])

c = 0
for i in df_clean['ZipCode'][~ mask]:
    try:
        temp = d_zip_istat[i]
        urban_col_masked[c] = d_istat_urb[int(temp)]
    except KeyError:
        i = '%05d' % int(i)
        if i[:-1]+'x' in d_zip_istat:
            temp = d_zip_istat[i[:-1]+'x']
        elif i[:-2]+'xx' in d_zip_istat:
            temp = d_zip_istat[i[:-2]+'xx']
        else:
            raise()
    c += 1
    
df_clean['Urban'] = df_clean['ZipCode'].copy()
df_clean['Urban'][~ mask] = urban_col_masked


# ## Data imputation
# We have decided to exploit supervised learning to fill up the database, it seemd to us the most reasonable approach, even though not for all categories we obtain an high score. We speculated that it would be nonetheless better than plugging in the mean or fill up missing data using sampling from the empirical distribution.  
# We have trained four supervised algorithms (Multiple Layer Perceptron, Perceptron, Random Forest and Logistic Regression) on each feature of the dataset which had missing data, using traffic_columns for trainig. These algorithms have been used as classifiers, therefore we have converted those features which weren't stricly categorical in nature. This might look as a loss of information, especially considering that Linear Regression would seem a natural approach to solve problems of this kind, when the response variable is not categorical. However, we chose not to use based on the following reasons:
# 1. There is not a clear relationship between traffic_columns and the feature we would be regressing them on, therefore even if the fitting would be nearly perfect, it would be questionable to use it for prediction;
# 2. The R-squared of regression is not alwasys straightforward to interpret, while the score of classifier algorithms is much easier;
# 3. By using classifiers, we could apply and compare what we have studied throughtout the course
# 
# For each feature, the four algorithms have been trained on the same training set, validated on the same validation set and finally tested on the same test set, to choose which of them performed at best. 
# 
# For some features, the dataset turned out to be unbalanced, therefore, when possible, we adjusted for this by using class weighting.
# 
# Once optimized the parameters for each classifier, the code we wrote automatically computes the best performing algorithm on the shared test set, keeps the best one and makes data imputation on the missing values using the predict() functionality of that. At the end we obtain a full dataset for eah feature.
# 
# While working we have noticed that the distribution of Data Allowance is skewed. In its section we plot the estimated density of the data for a graphical representation. We then thought that maybe it could help to cut the outliers out and train the classifiers on the remainig data points, we compared the outcome on the test set and it did help the training a bit. Due to time constraints, we limited this kind of analysis to that section only, although it might deliver better results to check also the other features and in case drop the outliers. 
# ### OS

# In[5]:


df_filled = df_clean.copy()
percentage_used = (0.70,0.15,0.15)
X = df_filled[traffic_columns]
y = df_filled['OS_clean']
build_seed = 456245
my_perc = perc(build_seed)
my_MLP = MLP(build_seed)
my_lr = LogReg(build_seed)
my_forest = trees(build_seed)


# In[6]:


my_perc.train(X, y, percentage=percentage_used, std=False, pca=0.9, threshold_unbalanced=0.6, epochs=200,  loss='log', 
              penalty='none', alpha=1e-8, power_t=0.5, it_interval=100, learning_rate='constant', eta0=1e-8,
              class_weight={'Android': 1.65, 'iOS': 2.70, 'other': 50.2})


# In[7]:


my_MLP.train(X, y, percentage=percentage_used, std=False, pca=0, threshold_unbalanced=0.6, epochs=500,
             hidden_layer_sizes = (200,), batch_size = 100, learning_rate_init=1e-2, solver = 'adam', 
             learning_rate = 'constant', momentum = 0.5, nesterovs_momentum = False,
             alpha = 0.01, tol = 1e-4)


# In[8]:


my_lr.train(X, y, percentage=percentage_used, std=False, pca=0.9, threshold_unbalanced=0.6, epochs=100,
            penalty="l2", dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, 
            class_weight=None,  solver="newton-cg", max_iter=100, multi_class="multinomial")


# In[9]:


my_forest.train(X, y, percentage=percentage_used, std=False, pca=0, threshold_unbalanced=0.6, tree_type='XRF',
                n_estimators = 50, max_features = "auto", criterion = "entropy", max_depth = 15,
                min_samples_split = 50, n_jobs = -1)


# In[10]:


best = test_sup(my_forest, my_lr, my_MLP, my_perc)


# Therefore we use this XRT to predict the rest of the column OS and keep the predictions to imput into our dataset later.

# In[11]:


os_missing = best.predict(X,y, fill_up=True)


# # Urbanization

# In[12]:


X = df_filled[traffic_columns]
df_filled['Urban'] = df_filled['Urban'].map(lambda x: int(x), na_action = 'ignore')
y = df_filled['Urban']
build_seed = 4562
my_perc = perc(build_seed)
my_MLP = MLP(build_seed)
my_lr = LogReg(build_seed)
my_forest = trees(build_seed)


# In[13]:


my_perc.train(X, y, percentage=percentage_used, std=False, pca=0, threshold_unbalanced=0.6, epochs=200,  loss='squared_hinge', 
              penalty='none', alpha=1e-8, power_t=0.7, it_interval=100, learning_rate='constant', eta0=1e-8,
              class_weight=None)


# In[14]:


my_MLP.train(X, y, percentage=percentage_used, std=False, pca=0, threshold_unbalanced=0.6, epochs=90,
             hidden_layer_sizes = (200,), batch_size = 50, learning_rate_init=1e-4, solver = 'adam', 
             learning_rate = 'constant', momentum = 0.5, nesterovs_momentum = False,
             alpha = 0.10, tol = 1e-4)


# In[15]:


my_lr.train(X, y, percentage=percentage_used, std=False, pca=0, threshold_unbalanced=0.6, epochs=100,
            penalty="l2", dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, 
            class_weight=None,  solver="newton-cg", max_iter=100, multi_class="multinomial")


# In[16]:


my_forest.train(X, y, percentage=percentage_used, std=False, pca=0, threshold_unbalanced=0.6, tree_type='XRF',
                n_estimators = 50, max_features = "auto", criterion = "entropy", max_depth = 15,
                min_samples_split = 50, n_jobs = -1)


# In[17]:


best = test_sup(my_forest, my_lr, my_MLP, my_perc)


# In[18]:


urbanization = best.predict(X,y, fill_up=True)


# # Numeric Age

# In[19]:


dict_numage_to_agecat = {0.85: 2, 0.65: 1, 0.35: 0, 0.75: 1, 0.55: 1, 0.45: 1, 0.25: 0, 0.15: 0}
df_filled["NumericAge"] = df_filled["NumericAge"].map(lambda x: dict_numage_to_agecat[x], na_action = 'ignore')
X = df_filled[traffic_columns]
y = df_filled['NumericAge']
build_seed = 456222
my_perc = perc(build_seed)
my_MLP = MLP(build_seed)
my_lr = LogReg(build_seed)
my_forest = trees(build_seed)


# In[20]:


my_perc.train(X, y, percentage=percentage_used, std=False, pca=0, threshold_unbalanced=0.6, epochs=50,  
              loss='log', penalty='none', alpha=1e-8, power_t=0.7, it_interval=100, learning_rate='constant', 
              eta0=1e-4, class_weight={2.0: 261.4, 1.0: 1.54, 0.0: 4.20})


# In[21]:


my_MLP.train(X, y, percentage=percentage_used, std=False, pca=0, threshold_unbalanced=0.6, epochs=500,
             hidden_layer_sizes = (200,), batch_size = 100, learning_rate_init=1e-2, solver = 'adam', 
             learning_rate = 'constant', momentum = 0.5, nesterovs_momentum = False,
             alpha = 0.2, tol = 1e-4)


# In[22]:


my_lr.train(X, y, percentage=percentage_used, std=False, pca=0, threshold_unbalanced=0.6, epochs=100,
            penalty="l2", dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, 
            class_weight=None,  solver="newton-cg", max_iter=100, multi_class="multinomial")


# In[23]:


my_forest.train(X, y, percentage=percentage_used, std=False, pca=0, threshold_unbalanced=0.6, tree_type='XRF',
                n_estimators = 50, max_features = "auto", criterion = "entropy", max_depth = 15,
                min_samples_split = 50, n_jobs = -1)


# In[24]:


best = test_sup(my_forest, my_lr, my_MLP, my_perc)


# In[25]:


#pay attention: MLP not weighted
num_age = best.predict(X,y, fill_up=True)


# # Data Allowance

# Most of the points looks like they're concentrated between 0 and 0.5. To better investigate this, let's look at the density of this column:

# In[26]:


df_clean['DataAllowance'].plot.density()


# In[27]:


mask = df_clean['DataAllowance'] > 0.5
np.sum(~mask)


# Therefore there are 1575 (over the 1636 not nan) which are below 0.5, so our conjecture was quite good. Given this, our idea is to take into consideration for our imputation just these X's, below 0.5, not using the outliers. We proceed using the above mask:

# In[28]:


X = df_filled[traffic_columns][~mask]
df_filled['DataAllowance'] = df_filled['DataAllowance'].map(lambda x: '%.1f'%x, na_action = 'ignore')
y = df_filled['DataAllowance'][~mask]

build_seed = 4562
my_perc = perc(build_seed)
my_MLP = MLP(build_seed)
my_lr = LogReg(build_seed)
my_forest = trees(build_seed)


# In[29]:


my_perc.train(X, y, percentage=percentage_used, std=False, pca=0, threshold_unbalanced=0.6, epochs=50,  
              loss='log', penalty='none', alpha=1e-8, power_t=0.7, it_interval=100, learning_rate='invscaling', 
              eta0=1e4, class_weight=None)


# In[30]:


my_MLP.train(X, y, percentage=percentage_used, std=False, pca=0, threshold_unbalanced=0.6, epochs=50,
             hidden_layer_sizes = (400,), batch_size = 100, learning_rate_init=1e-4, solver = 'adam', 
             learning_rate = 'constant', momentum = 0.5, nesterovs_momentum = False,
             alpha = 0.05, tol = 1e-4)


# In[31]:


my_lr.train(X, y, percentage=percentage_used, std=False, pca=0.9, threshold_unbalanced=0.6, epochs=100,
            penalty="l2", dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, 
            class_weight=None,  solver="newton-cg", max_iter=100, multi_class="multinomial")


# In[32]:


my_forest.train(X, y, percentage=percentage_used, std=False, pca=0, threshold_unbalanced=0.6, tree_type='XRF',
                n_estimators = 50, max_features = "auto", criterion = "entropy", max_depth = 15,
                min_samples_split = 50, n_jobs = -1)


# In[33]:


best = test_sup(my_forest, my_lr, my_MLP, my_perc)


# To predict, we must use the entire X and y, but with our trained MLP just on masked data.

# In[34]:


X = df_filled[traffic_columns]
y = df_filled['DataAllowance']


# In[35]:


data_all = best.predict(X,y, fill_up=True)


# # Monthly Data Traffic

# In[36]:


X = df_filled[traffic_columns]
df_filled['MonthlyDataTraffic'] = df_filled['MonthlyDataTraffic'].map(lambda x: '%.1f'%x, na_action = 'ignore')
y = df_filled['MonthlyDataTraffic']
build_seed = 4562
my_perc = perc(build_seed)
my_MLP = MLP(build_seed)
my_lr = LogReg(build_seed)
my_forest = trees(build_seed)


# In[37]:


my_perc.train(X, y, percentage=percentage_used, std=False, pca=0, threshold_unbalanced=0.6, epochs=50,  
              loss='log', penalty='none', alpha=1e-8, power_t=0.7, it_interval=100, learning_rate='invscaling', 
              eta0=1e4, class_weight={'0.2': 15.86, '0.0': 1.6, '0.1': 4.50, '0.5': 1316.0, '0.3': 101.23, '0.4': 188.0, '1.0': 658.0, '0.9': 1316.0})


# In[38]:


my_MLP.train(X, y, percentage=percentage_used, std=False, pca=0, threshold_unbalanced=0.6, epochs=500,
             hidden_layer_sizes = (200,), batch_size = 100, learning_rate_init=1e-2, solver = 'adam', 
             learning_rate = 'constant', momentum = 0.5, nesterovs_momentum = False,
             alpha = 0.01, tol = 1e-4)


# In[39]:


my_lr.train(X, y, percentage=percentage_used, std=False, pca=0, threshold_unbalanced=0.6, epochs=100,
            penalty="l2", dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, 
            class_weight=None,  solver="newton-cg", max_iter=100, multi_class="multinomial")


# In[40]:


my_forest.train(X, y, percentage=percentage_used, std=False, pca=0, threshold_unbalanced=0.6, tree_type='XRF',
                n_estimators = 50, max_features = "auto", criterion = "entropy", max_depth = 15,
                min_samples_split = 50, n_jobs = -1)


# In[41]:


best = test_sup(my_forest, my_lr, my_MLP, my_perc)


# In[42]:


data_traffic = best.predict(X,y, fill_up=True)


# # Data ARPU

# In[43]:


X = df_filled[traffic_columns]
df_filled['DataArpu'] = df_filled['DataArpu'].map(lambda x: '%.1f'%x, na_action = 'ignore')
y = df_filled['DataArpu']
build_seed = 4562
my_perc = perc(build_seed)
my_MLP = MLP(build_seed)
my_lr = LogReg(build_seed)
my_forest = trees(build_seed)


# In[44]:


my_perc.train(X, y, percentage=percentage_used, std=False, pca=0, threshold_unbalanced=0.6, epochs=50,  
              loss='log', penalty='none', alpha=1e-8, power_t=0.7, it_interval=100, learning_rate='invscaling', 
              eta0=1e4, class_weight=None)


# In[45]:


my_MLP.train(X, y, percentage=percentage_used, std=False, pca=0, threshold_unbalanced=0.6, epochs=100,
             hidden_layer_sizes = (200,50), batch_size = 100, learning_rate_init=1e-2, solver = 'adam', 
             learning_rate = 'constant', momentum = 0.5, nesterovs_momentum = False,
             alpha = 0.01, tol = 1e-4)


# In[46]:


my_lr.train(X, y, percentage=percentage_used, std=False, pca=0, threshold_unbalanced=0.6, epochs=100,
            penalty="l2", dual=False, tol=0.1, C=1.0, fit_intercept=True, intercept_scaling=1, 
            class_weight=None,  solver="newton-cg", max_iter=100, multi_class="multinomial")


# In[47]:


my_forest.train(X, y, percentage=percentage_used, std=False, pca=0, threshold_unbalanced=0.6, tree_type='XRF',
                n_estimators = 50, max_features = "auto", criterion = "entropy", max_depth = 15,
                min_samples_split = 50, n_jobs = -1)


# In[48]:


best = test_sup(my_forest, my_lr, my_MLP, my_perc)


# In[49]:


data_arpu = best.predict(X,y, fill_up=True)


# # Monthly Voice Traffic Count

# In[50]:


X = df_filled[traffic_columns]
df_filled['MonthlyVoiceTrafficCount'] = df_filled['MonthlyVoiceTrafficCount'].map(lambda x: '%.1f'%x, na_action = 'ignore')
y = df_filled['MonthlyVoiceTrafficCount']
build_seed = 4562
my_perc = perc(build_seed)
my_MLP = MLP(build_seed)
my_lr = LogReg(build_seed)
my_forest = trees(build_seed)


# In[51]:


my_perc.train(X, y, percentage=percentage_used, std=False, pca=0, threshold_unbalanced=0.6, epochs=50,  
              loss='log', penalty='none', alpha=1e-8, power_t=0.7, it_interval=100, learning_rate='invscaling', 
              eta0=1e4, class_weight=None)


# In[52]:


my_MLP.train(X, y, percentage=percentage_used, std=False, pca=0, threshold_unbalanced=0.6, epochs=100,
             hidden_layer_sizes = (200,50), batch_size = 100, learning_rate_init=1e-2, solver = 'adam', 
             learning_rate = 'constant', momentum = 0.5, nesterovs_momentum = False,
             alpha = 0.01, tol = 1e-4)


# In[53]:


my_lr.train(X, y, percentage=percentage_used, std=False, pca=0, threshold_unbalanced=0.6, epochs=100,
            penalty="l2", dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, 
            class_weight=None,  solver="newton-cg", max_iter=100, multi_class="multinomial")


# In[54]:


my_forest.train(X, y, percentage=percentage_used, std=False, pca=0, threshold_unbalanced=0.6, tree_type='XRF',
                n_estimators = 50, max_features = "auto", criterion = "entropy", max_depth = 15,
                min_samples_split = 50, n_jobs = -1)


# In[55]:


best = test_sup(my_forest, my_lr, my_MLP, my_perc)


# In[56]:


voice_traffic_c = best.predict(X,y, fill_up=True)


# In[57]:


df_good = df_filled.copy()
col_to_del = ['CustomerAge', 'DevicePrice', 'Province', 'Region', 'ZipCode', 'DeviceOperatingSystem']
df_good.drop(col_to_del, axis=1, inplace=True)
df_good['MonthlySmsTrafficCount'][df_good['MonthlySmsTrafficCount'].isnull()] = df_good['MonthlySmsTrafficCount'].mean()
df_good.info()


# ## Weights Optimization
# Once we have obtained the full dataset, we are ready to do clustering. Being the features really different, it doesn't not look reasonable to assume that they all have the same importance. To tackle this problem, we have decided to weight every feature. Since we had no prior guess regarding what these weights could be (i.e. a promising subset of the space that we could explore, even manually), we decided to use a supervised algorithm to optimize the parameters: KNearestNeighbors. More in details, we transformed the dataset applying some weights (still to be determined) and then we would run KNN on a training set, and test the score on a different set. The goodness of two different weights (we call them configurations) are evaluated based on the relative score of KNN under that configuration.
# 
# Once figured out how to compare weights, we just needed to come up with a way to find the best parameters. The most reasonable approach is to carry out a grid search, however an exact grid search would be computationally infeasible. Therefore we first thought of a randomized version. In our idea at the beginning, intstead of doing grid search on all the features at the same time, do it on a random subset and then repeat a few times (code of this approach is not attached). It did not perform very well and was still too costly computationally speaking (we have to consider that for each configuration we have one run of KNN, which, depending on the input size, can take between 0.2-0.9 on this dataset. On top of that, say that we propose 3 values to test for each feature, that makes 3^n_random_features * n_times to repeat. As the number of values or the number of random features to optimize over increases, the problem doesn't scale well. Then we adopted the following tactics:
# 1. Greedy approach: over a set of possible values for a specific feature, find the one that, keeping all the other weights constant, maximizes the score function. Repeat this procedure for a number of randomly sampled features, which should be much more of the number of features, to let the algorithm converge.  
# 2. Dataset Reduction: consider a very small percentage of the dataset, say 20%, onto which perform KNN. It is true that we lose precision because of less data points are used during training, though we can sensibly decrease the overall time, since KNN must be run for **every** configuration that is tested
# 
# Regarding the grid, we decided to let the weight range between 0 and 1 included (initializing the weights=1, which actually means that we are leaving the dataset unaffected). There is no need to use different intervals, as what is important is the relative scaling of the features i.e. if we multiply all the features by the same constant, nothing should change in the way the clusters are created (we observed this by changing the interval). What becomes important is the number of intermediate steps that make up the grid. Indeed, if a 100-step grid is created, at most one feature can be 100 times more important than another. Also, another thing that we observed was that, by including 0 as well among the possible values, it allows the algorithm to greedily (approximatively) decide that a certain feature is not relevant at all i.e. the best weight that can be given is 0, or alternatively, that all the values tested have the same score as the one when weight=0, which means that we can as well disregard that feature.
# 
# Good then, we have an algorithm to compare configurations, and another to greedily propose them, let's test it. We let it run overnight using 600 intermidiate values between 0 and 1, repeating the optimization 40 * n_features times (if n_features=2, then 80 times a feature as been extracted at random, and on that feature, given the actual configuration, its weight has been optimized). Below there is the graph produced, where we plot the KNN score every time a new onfiguration is accepted.
# 
# ![graph](graph.jpeg "graph")
# 
# Analysing the graph, a question rises: why is the score going up and down? We expected it to be monotonically increasing, as a new configuration is accepted if the score improves. However, we noticed that due to the way we wrote the algorithm, for a given feature, we selected the best weight value by comparing them among themselves, not with respect to the overall score achieved so far.  
# Also, the graph shows an interesting pattern: the score increases up to 8%, to then converge back to about 4%. Speculating on this we could say that, at first, the algorithm was exploring the space and it finally converged to a local minimum which was not optimal, things could have been different if, after a certian number of iterations, we would have compard new configurations against the best overall score so far.   
# These two reasoning made us realize that we could improve the optimization by using Simulated Annealing, which we have studied last semester and is in fact made for this: randomly explore the space at first (allows to accept worsening solution, which is what our algorithm unintentionally did), and then focus on the most promising space of solutions observed and converge there to a local optimal, as the temperature decreases, only accepting strictly improving solutions anymore, as we probably should have done. Therefore we implemented Simulated Annealing for this problem and preferred to keep that output.
# 
# We have attached both codes, as we deemed interesting how we usually start from the most straightforward solution (greedy gridsearch) and then, while doing, new ideas or strategies to tackle the problem come to mind that can be better suited.
# 
# Regarding the code below, we commented out the weight optimization function call because it would take too long, we let it run overnight and we just use the weights found in that instance, instead of computing them every time.

# In[58]:


df = df_good.copy()
col = df.columns[2:]

d_map = {'iOS': 1, 'Android': 2, 'other': 3}
df['OS_clean'] = df['OS_clean'].map(d_map, na_action='ignore')
cat_map = {'V-Bag': 1, 'V-Auto': 2, 'V-Pet': 3, 'V-Camera': 4}
df['Product'] = df['Product'].map(cat_map, na_action='ignore')

cat_col = [i for i in col if i not in traffic_columns]
non_cat_col = [i for i in col if i not in cat_col]
cat_col.pop(cat_col.index('MonthlySmsTrafficCount'))

X = df[col]
y = df['Product']
data = buildTrain(X, y,  perc=(0.3,0.2,0.5), std=False, pca=0, seed=None, one_hot=True, cat_col=cat_col)

knn1 = KNeighborsClassifier(n_neighbors=4)
weights = np.linspace(0, 10.0, num = 70)

grid = GridSearch(build_seed=647645)

#We do not repeat optimal_weight search since it is very time-consuming, for instance we had it running for one night, greedily
#exploring the space. Instead, for the analysis that follows, we use the weights we got after that analysis.
#optimal_weights,_ = grid.get_best(X, y, knn1, percentage=(0.3,0.2,0.5), std=False, pca=0, one_hot=True, cat_col=cat_col, epochs=1, 
                 #wmin=0, wmax=1, weights=None, start_config=None, data=data)


# For this specific instance of Simulated Annealing, acceptance rate is often somewhat high at the beginning because in general most of the time the greedy approach improves the score. As it run longer though, it will converge.

# In[59]:


weights = np.linspace(0, 10.0, num = 7)
grid_search = GridSearch_Sim(data.get_one_hot(), y, knn1, percentage=(0.2,0.2,0.6), std=False, pca=0, weights=weights)
grid_search, best = simann(grid_search, iters=10, seed=34534, beta0=100.0, beta1=1000, beta_steps=10)


# In[60]:


optimal_weights = np.array([0, 0, 0, 0.86956522, 0, 7.53623188, 0, 0, 6.08695652, 0,
        0, 2.89855072, 0.43478261, 0.43478261, 1.01449275, 1.01449275, 1.01449275, 1.01449275, 1.01449275, 1.88405797,
        0, 1.88405797, 0.28985507, 0, 0, 2.02898551, 0.72463768, 1.01449275, 0.14492754, 1.88405797,
        0, 0, 0, 0, 0, 1.15942029, 1.01449275, 0.28985507, 1.01449275, 1.01449275,
        1.01449275, 1.15942029, 1.01449275, 0, 0, 0, 0, 0, 1.01449275, 1.01449275,
        2.17391304, 0.43478261, 0.86956522, 0.86956522, 0, 0.57971014, 0, 1.30434783, 1.01449275, 2.17391304, 0, 0.72463768, 
        1.01449275, 2.17391304, 1.01449275, 1.01449275, 0, 0.86956522, 1.01449275, 1.01449275])

mask = optimal_weights>0


# It is interesting to see which features the algorithm has decided not to be useful in classifying the product, i.e. those that have weight=0

# In[61]:


data.get_one_hot().columns[~mask]


# ### Visual analysis of complete dataset
# Once reached this point, we have finished preprocessing on data, before clustering we try to visually inspect the data. We decide to do that on three different datasets, the filled one as obtained out of data imputation, the same one after performing OneHot encoding and finally, the one-hot encoded dataset, applying the optimal weights found. We transformed the data using PC to be able to plot it. Exception made for the first dataset, where the variance explained by the first two components is almost 50%, in the other two datasets it is really low, so the plot is not really significant. In any case, in all plots we notice how the data are not clusterized with respect to the labels, even though we can identify some clusters that unsupervised learning could derive.  
# We now move onto clustering.

# In[62]:


X_one_hot = data.get_one_hot().loc[:, mask]
temp = np.eye(X_one_hot.shape[1]) * optimal_weights[mask]

X_mod = pd.DataFrame(np.dot(X_one_hot, temp))


# In[63]:


one_hot = X_one_hot
xs = [df[col], one_hot, X_mod]

for X in xs:
    pca = PCA(0.9)
    x_pca = pca.fit_transform(X)
    print(pca.explained_variance_ratio_, pca.n_components_)
    plt.figure()
    plt.plot(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)
    plt.figure(figsize=(12,8))
    c = ['r','g','b','black']
    cat_map = {'V-Bag': 1, 'V-Auto': 2, 'V-Pet': 3, 'V-Camera': 4}
    for i in range(1, 5):
        plt.plot(x_pca[:,0][y==i], x_pca[:,1][y==i], '.', c=c[i-1])


# # Clustering
# In this section we show the application (and comparison of) two supervised learning and two unsupervised learning algorithms on the dataset.
# 
# **Unsupervised (clustering).**  
# We used K-Means and Hierarchical Clustering (AgglomerativeClustering). We posed the question of how can we a) compare the two and b) find the optimal parameters. Regarding the latter, it is not entirely clear how it can be done, especially for hierarchical clustering, we could guess that a good number of clusters could be around 4, since we have previous information that we can exploit. K-Means allows for a more rigorous (yet still subjective) decision rule: by plotting the loss function for different values of k, we choose the one for which the line plottet forms an elbow. In the plot below, we judged the elbow to be approximatively around k=8. Therefore we decided to use the same value as a guess for the number of neighbors for hierarchical clustering. We didn't judge it reasonable to explot the labels to compare different clusterizations, as that would "force" to prefer numbers of k resembling more the behavior of a supervised algorithm, while the whole goal of using unsupervised is to discover new patterns, not to stick or prefer the ones we already have. Having said that, we had to somehow compare the two clustering and choose the best, addressing question a) above. Definitely not an easy answer for us, we decided to exploit the labels and use the function v_measure_score from scikit.metrics to compare the clustering obtained with the labels. As already discussed, we don't believe it to be a perfect solution, but we could not identify another way.
# 
# **Supervised Learning (prediction).**
# For this task, we decided to use Random Forest and Multiple Layer Perceptron, for not particular reason, other than possibly the fact that RF is not influenced by weighting the features, and MLP should be able to figure out that by itself up to a certain extent (although weighting it already can certainly be of help for the algorithm). We will see that there will not be a sensible difference between the two appoaches. Obviously, since they are trained on the true labels, they can't be used for clustering the existing data points, however it can be interesting to see with which probability a new data point is predicted to belong to a specific category. After training, we compared MLP and RF to see the best performing one, to be used to predict such probability.  
# A further step that could be taken, instead of choosing just one of them, is to combine them together (ensambling) to make the prediction more robust. We haven't been able to do that due time constraints.

# #### K-means analysis

# In[64]:


data = buildTrain(X_mod, y, perc=(0.7,0.15,0.15), std=False, pca=False, seed = 222253)
scores = []
a, b = 2, 15
for k in range(a, b):
    print(('\n ****** K-Means: %i ******' % k))
    km = KMeans(n_clusters=k)
    km.fit(*data.get_train())
    check_clusters(y=data.get_train()[1], clust_labels=km.labels_, v=True)
    scores.append(-km.score(*data.get_valid()))
plt.figure()
plt.xlabel('k')
plt.ylabel('K-Means Score')
plt.plot(range(a, b), scores)


# #### Hierarchiacal Clustering Analysis

# In[65]:


k = 8
print(('\n ****** Hierarchical Clustering: %i ******' % k))
hc = AgglomerativeClustering(n_clusters=k, affinity='euclidean',linkage='ward')
hc.fit(*data.get_test())
check_clusters(y=data.get_test()[1], clust_labels=hc.labels_, v=True)
score_hc = metrics.v_measure_score(data.get_test()[1], hc.labels_)

print(('\n ****** K-Means: %i ******' % k))
km = KMeans(n_clusters=k)
km.fit(*data.get_test())
check_clusters(y=data.get_test()[1], clust_labels=km.labels_, v=True)
score_km = metrics.v_measure_score(data.get_test()[1], km.labels_)

print(('\n ****** Results: %i ******' % k))
print('Scores on tests sets:\n-Hierarchical Clustering: %.2f%%\n-K-Means: %.2f%%'%(score_hc*100, score_km*100))


# #### Predict probability of belonging to a class
# Random Forest

# In[66]:


X = df[col]
y = df['Product']
#data = buildTrain(X, y,  perc=(0.3,0.2,0.5), std=False, pca=0, seed=None, one_hot=True, cat_col=cat_col)

my_forest = trees(seed=3465, data=data)

my_forest.train(one_hot, y, percentage=(0.70,0.15,0.15), std=False, pca=0, threshold_unbalanced=0.6, tree_type='XRF',
                max_depth=3, max_features=None, min_samples_split=0.4)
#my_forest.view_tree(feature_names=feature_names)
#my_forest.train('sfdg', y, percentage=(0.70,0.15,0.15), std=False, pca=0, threshold_unbalanced=0.6, tree_type='XRF',
                #n_estimators = 100, max_features = 'auto', criterion = 'entropy', max_depth=5, min_samples_split = 50, n_jobs = -1)


# Multiple Layer Perceptron

# In[67]:


my_MLP = MLP(data=data)
my_MLP.train(X, y, percentage=percentage_used, std=False, pca=0, threshold_unbalanced=0.6, epochs=300,
             hidden_layer_sizes = (200,50), batch_size = 600, learning_rate_init=1e-4, solver = 'adam', 
             learning_rate = 'constant', momentum = 0.5, nesterovs_momentum = False,
             alpha = 13.5, tol = 1e-4)


# In[68]:


best = test_sup(my_forest, my_MLP)


# In[69]:


my_MLP.obj.predict_proba(data.get_test()[0])

