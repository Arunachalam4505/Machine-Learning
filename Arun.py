#!/usr/bin/env python
# coding: utf-8

# In[1]:


pwd


# In[2]:


cd Desktop/ml/train


# In[3]:


dtypes = {
        'MachineIdentifier':                                    'category',
        'ProductName':                                          'category',
        'EngineVersion':                                        'category',
        'AppVersion':                                           'category',
        'AvSigVersion':                                         'category',
        'IsBeta':                                               'int8',
        'RtpStateBitfield':                                     'float16',
        'IsSxsPassiveMode':                                     'int8',
        'DefaultBrowsersIdentifier':                            'float16',
        'AVProductStatesIdentifier':                            'float32',
        'AVProductsInstalled':                                  'float16',
        'AVProductsEnabled':                                    'float16',
        'HasTpm':                                               'int8',
        'CountryIdentifier':                                    'int16',
        'CityIdentifier':                                       'float32',
        'OrganizationIdentifier':                               'float16',
        'GeoNameIdentifier':                                    'float16',
        'LocaleEnglishNameIdentifier':                          'int8',
        'Platform':                                             'category',
        'Processor':                                            'category',
        'OsVer':                                                'category',
        'OsBuild':                                              'int16',
        'OsSuite':                                              'int16',
        'OsPlatformSubRelease':                                 'category',
        'OsBuildLab':                                           'category',
        'SkuEdition':                                           'category',
        'IsProtected':                                          'float16',
        'AutoSampleOptIn':                                      'int8',
        'PuaMode':                                              'category',
        'SMode':                                                'float16',
        'IeVerIdentifier':                                      'float16',
        'SmartScreen':                                          'category',
        'Firewall':                                             'float16',
        'UacLuaenable':                                         'float32',
        'Census_MDC2FormFactor':                                'category',
        'Census_DeviceFamily':                                  'category',
        'Census_OEMNameIdentifier':                             'float16',
        'Census_OEMModelIdentifier':                            'float32',
        'Census_ProcessorCoreCount':                            'float16',
        'Census_ProcessorManufacturerIdentifier':               'float16',
        'Census_ProcessorModelIdentifier':                      'float16',
        'Census_ProcessorClass':                                'category',
        'Census_PrimaryDiskTotalCapacity':                      'float32',
        'Census_PrimaryDiskTypeName':                           'category',
        'Census_SystemVolumeTotalCapacity':                     'float32',
        'Census_HasOpticalDiskDrive':                           'int8',
        'Census_TotalPhysicalRAM':                              'float32',
        'Census_ChassisTypeName':                               'category',
        'Census_InternalPrimaryDiagonalDisplaySizeInInches':    'float16',
        'Census_InternalPrimaryDisplayResolutionHorizontal':    'float16',
        'Census_InternalPrimaryDisplayResolutionVertical':      'float16',
        'Census_PowerPlatformRoleName':                         'category',
        'Census_InternalBatteryType':                           'category',
        'Census_InternalBatteryNumberOfCharges':                'float32',
        'Census_OSVersion':                                     'category',
        'Census_OSArchitecture':                                'category',
        'Census_OSBranch':                                      'category',
        'Census_OSBuildNumber':                                 'int16',
        'Census_OSBuildRevision':                               'int32',
        'Census_OSEdition':                                     'category',
        'Census_OSSkuName':                                     'category',
        'Census_OSInstallTypeName':                             'category',
        'Census_OSInstallLanguageIdentifier':                   'float16',
        'Census_OSUILocaleIdentifier':                          'int16',
        'Census_OSWUAutoUpdateOptionsName':                     'category',
        'Census_IsPortableOperatingSystem':                     'int8',
        'Census_GenuineStateName':                              'category',
        'Census_ActivationChannel':                             'category',
        'Census_IsFlightingInternal':                           'float16',
        'Census_IsFlightsDisabled':                             'float16',
        'Census_FlightRing':                                    'category',
        'Census_ThresholdOptIn':                                'float16',
        'Census_FirmwareManufacturerIdentifier':                'float16',
        'Census_FirmwareVersionIdentifier':                     'float32',
        'Census_IsSecureBootEnabled':                           'int8',
        'Census_IsWIMBootEnabled':                              'float16',
        'Census_IsVirtualDevice':                               'float16',
        'Census_IsTouchEnabled':                                'int8',
        'Census_IsPenCapable':                                  'int8',
        'Census_IsAlwaysOnAlwaysConnectedCapable':              'float16',
        'Wdft_IsGamer':                                         'float16',
        'Wdft_RegionIdentifier':                                'float16',
        'HasDetections':                                        'int8'
        }


# In[4]:


import pandas as pd
train=pd.read_csv('train.csv',dtype=dtypes,nrows=100000)


# In[5]:


import numpy as np 
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('fivethirtyeight')
import time
train_sample_fraction = None
na_rate_threshold = 0.9
train_sample_num = 15000
unbalanced_feature_rate_threshold = 0.9


# In[6]:


numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_and_binary_columns = [c for c,v in dtypes.items() if v in numerics]


# In[7]:


train.info()


# In[8]:


train.describe()


# In[9]:


stats = []
for col in train.columns:
    stats.append((col, train[col].dtype, train[col].nunique(), train[col].isnull().sum() * 100 / train.shape[0], train[col].value_counts(normalize=True, dropna=False).values[0] * 100))
    
stats_df = pd.DataFrame(stats, columns=['Feature', "type", 'Unique_values', 'Percentage of missing values', 'Percentage of values in the biggest category'])

stats_df.sort_values('Percentage of missing values', ascending=False)


# In[10]:


good_cols = list(train.columns)

for col in train.columns:
    
    # remove columns with high NA rate
    na_rate = train[col].isnull().sum() / train.shape[0]
    
    # remove columns with high Unbalanced values rate
    unbalanced_rate = train[col].value_counts(normalize=True, dropna=False).values[0]
    
    if na_rate > na_rate_threshold:
        good_cols.remove(col)
    elif unbalanced_rate > unbalanced_feature_rate_threshold:
        good_cols.remove(col)


# In[11]:


good_cols


# In[12]:


print("train data set reduced size in memory:", train.memory_usage().sum() / (1000000000))


# In[13]:


train = train[good_cols]


# In[14]:


del stats_df


# In[15]:



import gc

gc.collect()


# In[16]:


train.head()


# In[17]:


train.shape


# In[18]:


train['HasDetections'].value_counts()


# In[19]:


train['HasDetections'].value_counts().plot(kind="pie", figsize=(12,9), colormap="coolwarm")


# In[20]:


categorical_columns = list(train.loc[:, train.dtypes =="category"].columns)
numerical_and_binary_columns = list(train.loc[:, train.dtypes !="category"].columns)
numerical_columns = numerical_and_binary_columns

categorical_columns.remove("MachineIdentifier")

binary_columns = []
for col in (numerical_and_binary_columns):
    if train[col].nunique() == 2:
        binary_columns.append(col)
        numerical_columns.remove(col)


# In[21]:


categorical_columns


# In[22]:


len(numerical_columns)


# In[23]:


binary_columns


# In[24]:


categories_list = []
categories_list.append(len(categorical_columns))
categories_list.append(len(numerical_columns))
categories_list.append(len(binary_columns))

categories_df = pd.DataFrame(categories_list, 
                             index=["categorical_features", "numerical_features", "binary_features"])

categories_df = categories_df.transpose().plot(kind="barh", figsize=(21, 10), title="number of different features")


# In[25]:


if train_sample_fraction is not None:
    train_sample = train.sample(frac=15000, random_state=42)
elif train_sample_num is not None:
    train_sample = train.sample(15000, random_state=42)
else:
    train_sample = train.sample(n=15000, random_state=42)

gc.collect()


# In[26]:


categorical_columns


# In[27]:


#test_dtypes = {k: v for k, v in dtypes.items() if k in good_cols}


# In[28]:


#test_dtypes


# In[29]:


#cd ..


# In[30]:


#pwd


# In[31]:


#cd test


# In[32]:


#test = pd.read_csv('test.csv', dtype=dtypes, usecols=good_cols[:-1],nrows=100000)


# In[33]:


#test.head()


# In[34]:


#new = train_sample["OsBuildLab"].str.split(".", expand = True)
#train_sample["OsBuildLab_1"] = new[2]

#new = test["OsBuildLab"].str.split(".", expand = True)
#test["OsBuildLab_1"] = new[2]


#categorical_columns.append("OsBuildLab_1")
#del new


# In[35]:


train_sample = train_sample.drop(['MachineIdentifier'], axis=1)
#test = test.drop(['MachineIdentifier'], axis=1)


# In[36]:


train_sample = train_sample.reset_index(drop=True)


# In[37]:


train_sample


# In[38]:



modes = train_sample.mode()

for col in train_sample.columns:
    train_sample[col] = np.where(train_sample[col].isnull(), modes[col], train_sample[col])

del modes


# In[39]:


#modes_test = test.mode()

#for col in test.columns:
    #test[col] = np.where(test[col].isnull(), modes_test[col], test[col])

#train_sample.shape
#del modes_test


# In[40]:


train_sample


# In[41]:


#test.head()


# In[42]:


tra=train_sample


# In[43]:


true_numerical_columns = [
    'Census_ProcessorCoreCount',
    'Census_PrimaryDiskTotalCapacity',
    'Census_SystemVolumeTotalCapacity',
    'Census_TotalPhysicalRAM',
    'Census_InternalPrimaryDiagonalDisplaySizeInInches',
    'Census_InternalPrimaryDisplayResolutionHorizontal',
    'Census_InternalPrimaryDisplayResolutionVertical',
    'Census_InternalBatteryNumberOfCharges', 
    'Cat1', 'Cat2', 'Cat3', 'Cat4', 'Cat5', 'Cat6'
    
]

# binary variables:
binary_variables = [c for c in tra.columns if (tra[c].nunique() == 2) & (c not in ['PuaMode'])]

# to finally make a census of the categorical variables:
categorical_columns = [c for c in tra.columns 
                       if c in ['PuaMode'] or (c not in true_numerical_columns) & (c not in binary_variables)]

variables = {
    'categorical_columns': len(categorical_columns),
    'binary_variables': len(binary_variables),
    'true_numerical_columns': len(true_numerical_columns)
}


# In[44]:


true_numerical_columns


# In[45]:


binary_variables


# In[46]:


categorical_columns


# In[47]:


cardinality = []
for c in categorical_columns:
    if c == 'MachineIdentifier': continue
    cardinality.append([c, tra[c].nunique()])
cardinality.sort(key = lambda x:x[1], reverse=False)


# In[48]:


cardinality


# In[49]:


from tqdm import tqdm


# In[50]:


def frequency_encoding(variable):

    t = tra[variable].value_counts().reset_index()
    t = t.reset_index()
    t.loc[t[variable] == 1, 'level_0'] = np.nan
    t.set_index('index', inplace=True)
    max_label = t['level_0'].max() + 1
    t.fillna(max_label, inplace=True)
    return t.to_dict()['level_0']


# In[51]:


frequency_encoded_variables = [
    'Census_OEMModelIdentifier',
    'CityIdentifier',
    'Census_FirmwareVersionIdentifier',
#    'AvSigVersion',
    'Census_ProcessorModelIdentifier',
    'Census_OEMNameIdentifier',
    
]


# In[52]:


for variable in tqdm(frequency_encoded_variables):
    freq_enc_dict = frequency_encoding(variable)
    tra[variable] = tra[variable].map(lambda x: freq_enc_dict.get(x, np.nan))


# In[53]:


tra


# In[54]:


#test[variable] = test[variable].map(lambda x: freq_enc_dict.get(x, np.nan))


# In[55]:


#test


# In[56]:


indexer = {}
for col in tqdm(categorical_columns):
    if col == 'MachineIdentifier': continue
    _, indexer[col] = pd.factorize(tra[col])
    
for col in tqdm(categorical_columns):
    if col == 'MachineIdentifier': continue
    tra[col] = indexer[col].get_indexer(tra[col])
#    test[col] = indexer[col].get_indexer(test[col])
    

tra[:5]


# In[ ]:





# In[57]:


tra


# In[58]:


tra.info()


# In[59]:


y = tra['HasDetections']
X = tra.drop(['HasDetections'], axis=1)


# In[60]:


X = tra.iloc[:,0:55]
y = tra.iloc[:,-1]    
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(30).plot(kind='barh')
plt.show()


# In[61]:


feat_importances.nlargest(20)


# In[62]:


R=['SmartScreen','AvSigVersion','Census_FirmwareVersionIdentifier','Census_ProcessorModelIdentifier','Census_SystemVolumeTotalCapacity','Census_OEMModelIdentifier','EngineVersion','CityIdentifier','GeoNameIdentifier','CountryIdentifier','Wdft_RegionIdentifier','Census_OEMNameIdentifier','Census_OSVersion','Census_OSInstallTypeName','Census_TotalPhysicalRAM','Census_FirmwareManufacturerIdentifier','Census_OSBuildRevision','Census_PrimaryDiskTotalCapacity','Census_InternalPrimaryDiagonalDisplaySizeInInches','LocaleEnglishNameIdentifier']


# In[63]:


import numpy
from pandas import read_csv
from sklearn.decomposition import PCA
X = tra.iloc[:,0:55]
y = tra.iloc[:,-1]  
pca = PCA(n_components=3)
fit = pca.fit(X)
# summarize components
print("Explained Variance: %s" % fit.explained_variance_ratio_)
print(fit.components_)


# In[64]:


R=['AvSigVersion','CityIdentifier','SmartScreen','Census_OEMModelIdentifier','Census_ProcessorModelIdentifier',
   'CountryIdentifier','Census_SystemVolumeTotalCapacity','Census_OEMNameIdentifier',
   'GeoNameIdentifier','LocaleEnglishNameIdentifier',
  'Census_InternalPrimaryDiagonalDisplaySizeInInches','Census_TotalPhysicalRAM','Census_OSBuildRevision','Census_OSVersion',
  'Census_PrimaryDiskTotalCapacity','EngineVersion','AppVersion','Census_ProcessorCoreCount','AVProductStatesIdentifier','AVProductsInstalled']


# In[65]:


X=tra[R]


# In[66]:


X.head(4)


# In[176]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import time
from sklearn.model_selection import cross_val_score

# create a 70/30 split of the data 
xtrain, xtest, ytrain, ytest = train_test_split(X, y, random_state=42, test_size=0.3)


# In[177]:


from sklearn.ensemble import RandomForestClassifier
m = RandomForestClassifier(n_estimators=40, min_samples_leaf=100, max_features=0.5, n_jobs=-1, oob_score=False)
m.fit(xtrain, ytrain)
scores=cross_val_score(m, X, y, cv=10)
y_pred = m.predict(xtest)


# In[178]:


scores


# In[179]:


from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score


# In[200]:


cm = confusion_matrix(ytest, m.predict(xtest))
print('Confusuion Matrix:')
print(cm)

# accuracy: (tp + tn) / (p + n)
print("Accuracy:",metrics.accuracy_score(ytest, y_pred))


auc = roc_auc_score(ytest, y_pred)
print('ROC AUC: %f' % auc)



# In[201]:


from sklearn.metrics import classification_report
print(classification_report(ytest,y_pred))


# In[202]:


#Plotting of confusion Matrix
from sklearn.metrics import confusion_matrix
import scikitplot as skplt

sns.set(rc={'figure.figsize':(8,8)})
skplt.metrics.plot_confusion_matrix(ytest,y_pred, cmap="BrBG")


# In[ ]:





# In[203]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)


# In[204]:


model = model.fit(xtrain,ytrain)
scores1=cross_val_score(model, X, y, cv=10)
y_pred = model.predict(xtest)


# In[205]:


scores1


# In[188]:


y_pred


# In[206]:


cm = confusion_matrix(ytest, m.predict(xtest))
print('Confusuion Matrix:')
print(cm)

# accuracy: (tp + tn) / (p + n)
print("Accuracy:",metrics.accuracy_score(ytest, y_pred))

auc = roc_auc_score(ytest, y_pred)
print('ROC AUC: %f' % auc)


# In[207]:


from sklearn.metrics import classification_report
print(classification_report(ytest,y_pred))


# In[208]:


#Plotting of confusion Matrix
from sklearn.metrics import confusion_matrix
import scikitplot as skplt

sns.set(rc={'figure.figsize':(8,8)})
skplt.metrics.plot_confusion_matrix(ytest,y_pred, cmap="BrBG")


# In[198]:


from sklearn.linear_model import LogisticRegression


# In[199]:


logistic_regression= LogisticRegression(solver='lbfgs', multi_class='ovr')
model=logistic_regression.fit(xtrain,ytrain)
scores2=cross_val_score(model, X, y, cv=10)


# In[83]:


y_pred=logistic_regression.predict(xtest)


# In[144]:


scores2


# In[145]:


cm = confusion_matrix(ytest, m.predict(xtest))
print('Confusuion Matrix:')
print(cm)

# accuracy: (tp + tn) / (p + n)
print("Accuracy:",metrics.accuracy_score(ytest, y_pred))
auc = roc_auc_score(ytest, y_pred)
print('ROC AUC: %f' % auc)


# In[86]:


from sklearn.metrics import classification_report
print(classification_report(ytest,y_pred))


# In[87]:


#Plotting of confusion Matrix
from sklearn.metrics import confusion_matrix
import scikitplot as skplt

sns.set(rc={'figure.figsize':(8,8)})
skplt.metrics.plot_confusion_matrix(ytest,y_pred, cmap="BrBG")


# In[209]:


from sklearn.dummy import DummyClassifier


# In[210]:


dummy_classifier = DummyClassifier(strategy="most_frequent")
dummy_classifier.fit( X,y )


# In[211]:


scores3=cross_val_score(model, X, y, cv=10)
y_pred = model.predict(xtest)


# In[212]:


cm = confusion_matrix(ytest, m.predict(xtest))
print('Confusuion Matrix:')
print(cm)

# accuracy: (tp + tn) / (p + n)
print("Accuracy:",metrics.accuracy_score(ytest, y_pred))

auc = roc_auc_score(ytest, y_pred)
print('ROC AUC: %f' % auc)


# In[213]:


from sklearn.metrics import classification_report
print(classification_report(ytest,y_pred))


# In[214]:


#Plotting of confusion Matrix
from sklearn.metrics import confusion_matrix
import scikitplot as skplt

sns.set(rc={'figure.figsize':(8,8)})
skplt.metrics.plot_confusion_matrix(ytest,y_pred, cmap="BrBG")


# In[215]:


import xgboost as xgb

clf_xgb = xgb.XGBClassifier(learning_rate=0.03, 
                            n_estimators=3000, 
                            max_depth=11,
                            min_child_weight=9,
                            gamma=0.2,
                            subsample=1,
                            colsample_bytree=0.4,
                            objective= 'binary:logistic',
                            nthread=-1,
                            scale_pos_weight=1,
                            reg_alpha = 0.6,
                            reg_lambda = 3,
                            seed=42)

clf_xgb.fit(xtrain, ytrain, eval_set=[(xtrain, ytrain), (xtest, ytest)], 
            early_stopping_rounds=100, eval_metric='auc', verbose=100)

y_pred = clf_xgb.predict(xtest)


# In[216]:


cm = confusion_matrix(ytest, m.predict(xtest))
print('Confusuion Matrix:')
print(cm)

# accuracy: (tp + tn) / (p + n)
print("Accuracy:",metrics.accuracy_score(ytest, y_pred))

auc = roc_auc_score(ytest, y_pred)
print('ROC AUC: %f' % auc)


# In[217]:


#Plotting of confusion Matrix
from sklearn.metrics import confusion_matrix
import scikitplot as skplt

sns.set(rc={'figure.figsize':(8,8)})
skplt.metrics.plot_confusion_matrix(ytest,y_pred, cmap="BrBG")


# In[147]:


from sklearn.naive_bayes import GaussianNB


# In[152]:


model = GaussianNB()
model = model.fit(xtrain,ytrain)
scores=cross_val_score(model, X, y, cv=10)
y_pred = model.predict(xtest)
print("Accuracy:",metrics.accuracy_score(ytest, y_pred))
auc = roc_auc_score(ytest, y_pred)
print('ROC AUC: %f' % auc)


# In[153]:


from sklearn.metrics import classification_report
print(classification_report(ytest,y_pred))


# In[154]:


from sklearn.metrics import confusion_matrix
import scikitplot as skplt

sns.set(rc={'figure.figsize':(8,8)})
skplt.metrics.plot_confusion_matrix(ytest,y_pred, cmap="BrBG")


# In[100]:


from sklearn import tree


# In[155]:


clf = tree.DecisionTreeClassifier(random_state=0)


# In[158]:


clf = clf.fit(xtrain,ytrain)
scores4=cross_val_score(clf, X, y, cv=10)
y_pred = clf.predict(xtest)
print("Accuracy:",metrics.accuracy_score(ytest, y_pred))
auc = roc_auc_score(ytest, y_pred)
print('ROC AUC: %f' % auc)


# In[159]:


from sklearn.metrics import classification_report
print(classification_report(ytest,y_pred))


# In[104]:


from sklearn.metrics import confusion_matrix
import scikitplot as skplt

sns.set(rc={'figure.figsize':(8,8)})
skplt.metrics.plot_confusion_matrix(ytest,y_pred, cmap="BrBG")


# In[105]:


scores4


# In[106]:



df=pd.DataFrame(data=scores2,columns=['Logistic'])
df['DecisionTree']=scores4
df['KNN']=scores1
df['RandomForest']=scores
df
from scipy.stats import ttest_rel
from scipy.stats import friedmanchisquare
def ptest(p):
  if p > 0.05:
    print('Probably the same distribution')
  else:
    print('Probably different distributions')

stat, p = ttest_rel(scores,scores1)
#stat, p = friedmanchisquare(RandomForest,KNN)
print('stat=%.3f, p=%.3f' % (stat, p))
ptest(p)
#stat, p = friedmanchisquare(Logistic,DecisionTree)
stat, p = ttest_rel(scores2,scores4)
print('stat=%.3f, p=%.3f' % (stat, p))
ptest(p)


# In[218]:


from sklearn.ensemble import BaggingClassifier


# In[219]:


from sklearn.ensemble import RandomForestClassifier


# In[220]:


clf = BaggingClassifier(base_estimator=RandomForestClassifier(),
                       n_estimators=10, random_state=0).fit(X, y)


# In[221]:


scores_bag=cross_val_score(clf, X, y, cv=10)
y_pred = clf.predict(xtest)


# In[222]:


print("Accuracy:",metrics.accuracy_score(ytest, y_pred))
auc = roc_auc_score(ytest, y_pred)
print('ROC AUC: %f' % auc)
from sklearn.metrics import classification_report
print(classification_report(ytest,y_pred))


# In[223]:


from sklearn.metrics import confusion_matrix
import scikitplot as skplt

sns.set(rc={'figure.figsize':(8,8)})
skplt.metrics.plot_confusion_matrix(ytest,y_pred, cmap="BrBG")


# In[224]:


from sklearn.model_selection import cross_val_score


# In[225]:


from sklearn.ensemble import AdaBoostClassifier


# In[226]:


clf = AdaBoostClassifier(n_estimators=100)


# In[227]:


scores = cross_val_score(clf, X, y, cv=5)


# In[228]:


clf=clf.fit(xtrain,ytrain)


# In[229]:


y_pred = clf.predict(xtest)


# In[172]:


print("Accuracy:",metrics.accuracy_score(ytest, y_pred))


# In[173]:


print("Accuracy:",metrics.accuracy_score(ytest, y_pred))
auc = roc_auc_score(ytest, y_pred)
print('ROC AUC: %f' % auc)
from sklearn.metrics import classification_report
print(classification_report(ytest,y_pred))


# In[230]:


from sklearn.metrics import confusion_matrix
import scikitplot as skplt

sns.set(rc={'figure.figsize':(8,8)})
skplt.metrics.plot_confusion_matrix(ytest,y_pred, cmap="BrBG")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




