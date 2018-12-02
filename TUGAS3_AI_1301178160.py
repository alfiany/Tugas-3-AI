
# coding: utf-8

# # TUGAS 3 AI
# #### Nama : Alfian Yulianto
# #### Nim : 1301178160
# #### Kelas : IFX-41-GAB05

# In[1]:


#List import sesuai keperluan dari program yang akan dibuat
from copy import deepcopy
import numpy as np
import pandas as pd
import math
from random import randint


# In[2]:


dtrain=pd.read_csv('DataTrain_Tugas3_AI.csv').as_matrix()
dtr_x=dtrain[:,1:-1]
dtr_y= np.squeeze(np.array(dtrain[:,6:]))


# In[3]:


dtest=pd.read_csv('DataTest_Tugas3_AI.csv').as_matrix()
dts_x=dtest[:,1:-1]


# In[4]:


def euclidean(x,y):
    return math.sqrt((dts_x[x][0] - dtr_x[y][0])**2 + (dts_x[x][1] - dtr_x[y][1])**2 +
                      (dts_x[x][2] - dtr_x[y][2])**2 + (dts_x[x][3] - dtr_x[y][3])**2 +
                      (dts_x[x][4] - dtr_x[y][4])**2)
    


# In[5]:


best_k = []
for i in range(len(dts_x)):
    distance= []
    for j in range(len(dtr_x)):
        distance.append(euclidean(i,j))

    best = [a for _, a in sorted(zip(distance, dtr_y))]
    best_k.append(np.bincount(best[0:10]).argmax())


# In[6]:


print(best_k)


# In[7]:


report = pd.DataFrame({
    "No" : best_k,
})


# In[8]:


report.to_csv('TebakanTugas3.csv',index = False)

