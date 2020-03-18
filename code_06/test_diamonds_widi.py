# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %% [markdown]
## Global Variables

# %%
from IPython import get_ipython

import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
import numpy as np
import os
import seaborn as sns
import urllib.request, urllib.error, urllib.parse
import sys
import unittest
from diamonds_widi_06 import  DiamondsPredictive

get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', 20)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.width', 1000)



# %%
