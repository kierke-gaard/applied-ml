#%% [markdown]
# ## Packages and Tools
#%%
#get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings(action='ignore')

import pandas as pd
import seaborn as sns
import matplotlib as plt
import sklearn as sl
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score

#%% [markdown]
# ## Import Data
#%%
data = pd.read_csv('c:\\dev\\applied-ml\\bike\\data\\hour.csv')
data.describe()

#%% [markdown]
# ## Description
#%%
# FILTER, GROUP-BY
data[(data.holiday != 0)][['hr', 'cnt']].groupby('hr').agg('sum')
#%%
