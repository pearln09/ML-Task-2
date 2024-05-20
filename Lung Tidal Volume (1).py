#!/usr/bin/env python
# coding: utf-8

# In[78]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


np.random.seed(42)
subject_num = 50000


# In[79]:


age = np.random.randint(18, 81, size=subject_num)
gender = np.random.choice([0, 1], size=subject_num, p=[0.6, 0.4])
height = np.random.randint(150, 201, size=subject_num)
cough = np.random.choice([0, 1], size=subject_num, p=[0.7, 0.3])
smoker = np.random.choice([0, 1], size=subject_num, p=[0.7, 0.3])
asthma = np.random.choice([0, 1], size=subject_num, p=[0.9, 0.1])


# In[80]:


base_tidal_volume = 500  
age_diff = 5 * (age - 30) / 50
gender_diff = 50 * gender  
height_diff = 2 * (height - 170)  
symptom_diff = -50 * (cough + smoker + asthma)


# In[81]:


tidal_volume = base_tidal_volume + age_diff + gender_diff + height_diff + symptom_diff


# In[82]:


dataset = pd.DataFrame({
    'Age': age,
    'Gender': gender,
    'Height': height,
    'Cough': cough,
    'Smoker': smoker,
    'Asthma': asthma,
    'Tidal Volume': tidal_volume
})


# In[83]:


dataset.to_csv('tidal_volume_data.csv', index=False)


# In[84]:


print(dataset.head())


# In[85]:


dataset = pd.read_csv("tidal_volume_data.csv")


# In[86]:


data_set= pd.read_csv('tidal_volume_data.csv')    
X= data_set.iloc[:, 0:-1].values  
y= data_set.iloc[:,-1].values  


# In[87]:


print(X)


# In[88]:


print(y)


# In[89]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.25, random_state=0)  


# In[90]:


from sklearn.preprocessing import StandardScaler    
sc= StandardScaler()    
X_train= sc.fit_transform(X_train)    
X_test= sc.transform(X_test)    


# In[91]:


print(X_test)


# In[92]:


from sklearn.ensemble import RandomForestRegressor
regressor1=RandomForestRegressor(n_estimators=10,random_state=0)
regressor1.fit(X_train,y_train)


# In[93]:


y_pred1=regressor1.predict(X_test)


# In[94]:


from sklearn.linear_model import LinearRegression
regressor2=LinearRegression()
regressor2.fit(X_train,y_train)


# In[95]:


y_pred2=regressor2.predict(X_test)


# In[96]:


from sklearn.tree import DecisionTreeRegressor
regressor3=DecisionTreeRegressor(random_state=0)
regressor3.fit(X_train,y_train)


# In[97]:


y_pred3=regressor3.predict(X_test)


# In[98]:


print(y_pred1)


# In[99]:


print(y_pred2)


# In[100]:


print(y_pred3)


# In[101]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

mae1 = mean_absolute_error(y_test, y_pred1)
mse1 = mean_squared_error(y_test, y_pred1)
rmse1 = mse1 ** 0.5

print(f'MAE 1: {mae1}')
print(f'MSE 1: {mse1}')
print(f'RMSE 1: {rmse1}')


# In[102]:


mae2 = mean_absolute_error(y_test, y_pred2)
mse2 = mean_squared_error(y_test, y_pred2)
rmse2 = mse2 ** 0.5

print(f'MAE 2: {mae2}')
print(f'MSE 2: {mse2}')
print(f'RMSE 2: {rmse2}')


# In[103]:


mae3 = mean_absolute_error(y_test, y_pred3)
mse3 = mean_squared_error(y_test, y_pred3)
rmse3 = mse3 ** 0.5

print(f'MAE 3: {mae3}')
print(f'MSE 3: {mse3}')
print(f'RMSE 3: {rmse3}')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




