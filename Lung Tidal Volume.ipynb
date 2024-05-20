{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "07c888be-b763-49e3-9ad3-5381f1b59bcc",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt \n",
    "\n",
    "\n",
    "np.random.seed(42)\n",
    "subject_num = 50000"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "3114a5c4-671f-4cc1-bbff-42611ab993cc",
   "metadata": {},
   "outputs": [],
   "source": [
    "age = np.random.randint(18, 81, size=subject_num)\n",
    "gender = np.random.choice([0, 1], size=subject_num, p=[0.5, 0.5])\n",
    "height = np.random.randint(150, 201, size=subject_num)\n",
    "symptom1 = np.random.choice([0, 1], size=subject_num, p=[0.7, 0.3])\n",
    "symptom2 = np.random.choice([0, 1], size=subject_num, p=[0.8, 0.2])\n",
    "symptom3 = np.random.choice([0, 1], size=subject_num, p=[0.9, 0.1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "77ae025b-a65c-4fdc-b8f5-58754000774d",
   "metadata": {},
   "outputs": [],
   "source": [
    "base_tidal_volume = 500  \n",
    "age_factor = 5 * (age - 30) / 50\n",
    "gender_factor = 50 * gender  \n",
    "height_factor = 2 * (height - 170)  \n",
    "symptom_factor = -50 * (symptom1 + symptom2 + symptom3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "0e318fca-f0f2-4386-9352-04715f8aa306",
   "metadata": {},
   "outputs": [],
   "source": [
    "tidal_volume = base_tidal_volume + age_factor + gender_factor + height_factor + symptom_factor"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "ea54463d-21fe-4637-9877-519cf9020767",
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset = pd.DataFrame({\n",
    "    'Age': age,\n",
    "    'Gender': gender,\n",
    "    'Height': height,\n",
    "    'Cough': symptom1,\n",
    "    'Smoker': symptom2,\n",
    "    'Asthma': symptom3,\n",
    "    'Tidal_Volume': tidal_volume\n",
    "})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "4eb57f7f-9726-439d-aa23-17c70d514507",
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset.to_csv('tidal_volume_data.csv', index=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "80057b9c-1803-4de1-a749-acdc570dd9e8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   Age  Gender  Height  Cough  Smoker  Asthma  Tidal_Volume\n",
      "0   56       1     194      0       0       0         600.6\n",
      "1   69       1     173      0       1       0         509.9\n",
      "2   46       0     195      0       0       0         551.6\n",
      "3   32       0     166      1       0       0         442.2\n",
      "4   60       0     164      0       0       0         491.0\n"
     ]
    }
   ],
   "source": [
    "print(dataset.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "5ad0f4e8-105e-4432-bc85-de89460a40e2",
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset = pd.read_csv(\"tidal_volume_data.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "5ceb2150-a0b3-460f-aa92-03d86ad327ba",
   "metadata": {},
   "outputs": [],
   "source": [
    "data_set= pd.read_csv('tidal_volume_data.csv')    \n",
    "X= data_set.iloc[:, 0:-1].values  \n",
    "y= data_set.iloc[:,-1].values  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "8d03107c-aeac-463b-b1ab-e0126f7aa0d7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 56   1 194   0   0   0]\n",
      " [ 69   1 173   0   1   0]\n",
      " [ 46   0 195   0   0   0]\n",
      " ...\n",
      " [ 52   0 164   0   0   0]\n",
      " [ 37   1 195   1   0   1]\n",
      " [ 52   0 200   0   0   0]]\n"
     ]
    }
   ],
   "source": [
    "print(X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "8ba9ab4a-adfa-40ba-8501-7e076904af94",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[600.6 509.9 551.6 ... 490.2 500.7 562.2]\n"
     ]
    }
   ],
   "source": [
    "print(y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "fa4fbc61-3db1-4f79-95c3-f416c9e78192",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split  \n",
    "X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.25, random_state=0)  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "a14c3293-9c1f-40c4-9e92-003b81a9d6ac",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import StandardScaler    \n",
    "sc= StandardScaler()    \n",
    "X_train= sc.fit_transform(X_train)    \n",
    "X_test= sc.transform(X_test)    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "9c1f9769-4d89-48a1-b17c-414c2d5c7d4a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 1.53930165  1.00545488 -1.62837047  1.51329141 -0.49949995  2.95212227]\n",
      " [-0.66193549 -0.99457472 -1.69627197 -0.66081126  2.0020022  -0.33873936]\n",
      " [-0.16665713 -0.99457472 -1.69627197 -0.66081126 -0.49949995 -0.33873936]\n",
      " ...\n",
      " [-1.2672757   1.00545488  1.56300033 -0.66081126 -0.49949995 -0.33873936]\n",
      " [-0.05659527 -0.99457472  0.47657623 -0.66081126  2.0020022  -0.33873936]\n",
      " [ 0.6588068  -0.99457472  0.40867472 -0.66081126 -0.49949995 -0.33873936]]\n"
     ]
    }
   ],
   "source": [
    "print(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "5786312b-8bbe-4233-bc6d-0508780ac1d0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-4 {color: black;background-color: white;}#sk-container-id-4 pre{padding: 0;}#sk-container-id-4 div.sk-toggleable {background-color: white;}#sk-container-id-4 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-4 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-4 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-4 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-4 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-4 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-4 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-4 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-4 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-4 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-4 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-4 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-4 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-4 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-4 div.sk-item {position: relative;z-index: 1;}#sk-container-id-4 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-4 div.sk-item::before, #sk-container-id-4 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-4 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-4 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-4 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-4 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-4 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-4 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-4 div.sk-label-container {text-align: center;}#sk-container-id-4 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-4 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-4\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>RandomForestRegressor(n_estimators=10, random_state=0)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-4\" type=\"checkbox\" checked><label for=\"sk-estimator-id-4\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">RandomForestRegressor</label><div class=\"sk-toggleable__content\"><pre>RandomForestRegressor(n_estimators=10, random_state=0)</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "RandomForestRegressor(n_estimators=10, random_state=0)"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.ensemble import RandomForestRegressor\n",
    "regressor1=RandomForestRegressor(n_estimators=10,random_state=0)\n",
    "regressor1.fit(X_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "36ee56c4-15d3-4f42-8472-bb1ae01e0a81",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred1=regressor1.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "efaa7b4b-0a3c-44c6-b361-837bde863fc3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-5 {color: black;background-color: white;}#sk-container-id-5 pre{padding: 0;}#sk-container-id-5 div.sk-toggleable {background-color: white;}#sk-container-id-5 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-5 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-5 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-5 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-5 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-5 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-5 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-5 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-5 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-5 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-5 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-5 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-5 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-5 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-5 div.sk-item {position: relative;z-index: 1;}#sk-container-id-5 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-5 div.sk-item::before, #sk-container-id-5 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-5 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-5 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-5 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-5 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-5 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-5 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-5 div.sk-label-container {text-align: center;}#sk-container-id-5 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-5 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-5\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-5\" type=\"checkbox\" checked><label for=\"sk-estimator-id-5\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">LinearRegression</label><div class=\"sk-toggleable__content\"><pre>LinearRegression()</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "LinearRegression()"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.linear_model import LinearRegression\n",
    "regressor2=LinearRegression()\n",
    "regressor2.fit(X_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "9728f24c-ba7e-4082-a168-2abbeff07d63",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred2=regressor2.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "f69d87a2-2a20-471d-b861-143f3d9b48f2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-6 {color: black;background-color: white;}#sk-container-id-6 pre{padding: 0;}#sk-container-id-6 div.sk-toggleable {background-color: white;}#sk-container-id-6 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-6 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-6 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-6 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-6 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-6 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-6 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-6 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-6 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-6 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-6 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-6 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-6 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-6 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-6 div.sk-item {position: relative;z-index: 1;}#sk-container-id-6 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-6 div.sk-item::before, #sk-container-id-6 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-6 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-6 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-6 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-6 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-6 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-6 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-6 div.sk-label-container {text-align: center;}#sk-container-id-6 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-6 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-6\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>DecisionTreeRegressor(random_state=0)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-6\" type=\"checkbox\" checked><label for=\"sk-estimator-id-6\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">DecisionTreeRegressor</label><div class=\"sk-toggleable__content\"><pre>DecisionTreeRegressor(random_state=0)</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "DecisionTreeRegressor(random_state=0)"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.tree import DecisionTreeRegressor\n",
    "regressor3=DecisionTreeRegressor(random_state=0)\n",
    "regressor3.fit(X_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "3720a802-56ab-4f68-bc1e-3eb5c8ce490f",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred3=regressor3.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "b863ede2-c196-466b-9373-81a7cce50862",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[415.46 410.63 461.6  ... 605.58 475.8  525.1 ]\n"
     ]
    }
   ],
   "source": [
    "print(y_pred1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "a53b83b0-712d-4b22-b138-f9a8dff52706",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[416.7 410.7 461.6 ... 605.6 475.8 525.1]\n"
     ]
    }
   ],
   "source": [
    "print(y_pred2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "d5770c4b-f438-49ab-89ca-e1472da16c57",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[416.6 410.7 461.6 ... 605.5 475.9 525.1]\n"
     ]
    }
   ],
   "source": [
    "print(y_pred3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "e9427794-0699-4895-a1da-ace12576c940",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "MAE 1: 0.07545680000001864\n",
      "MSE 1: 0.05359740000000016\n",
      "RMSE 1: 0.2315111228429428\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import mean_squared_error\n",
    "from sklearn.metrics import mean_absolute_error\n",
    "\n",
    "mae1 = mean_absolute_error(y_test, y_pred1)\n",
    "mse1 = mean_squared_error(y_test, y_pred1)\n",
    "rmse1 = mse1 ** 0.5\n",
    "\n",
    "print(f'MAE 1: {mae1}')\n",
    "print(f'MSE 1: {mse1}')\n",
    "print(f'RMSE 1: {rmse1}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "d1bf0588-f6c1-45fc-b588-e53219d11114",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "MAE 2: 1.3526005204766988e-13\n",
      "MSE 2: 2.675153799784776e-26\n",
      "RMSE 2: 1.6355897406699444e-13\n"
     ]
    }
   ],
   "source": [
    "mae2 = mean_absolute_error(y_test, y_pred2)\n",
    "mse2 = mean_squared_error(y_test, y_pred2)\n",
    "rmse2 = mse2 ** 0.5\n",
    "\n",
    "print(f'MAE 2: {mae2}')\n",
    "print(f'MSE 2: {mse2}')\n",
    "print(f'RMSE 2: {rmse2}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "745b4b2f-bab4-4883-b435-a96f6aa299a2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "MAE 3: 0.07700000000000203\n",
      "MSE 3: 0.08305519999999973\n",
      "RMSE 3: 0.28819299089325495\n"
     ]
    }
   ],
   "source": [
    "mae3 = mean_absolute_error(y_test, y_pred3)\n",
    "mse3 = mean_squared_error(y_test, y_pred3)\n",
    "rmse3 = mse3 ** 0.5\n",
    "\n",
    "print(f'MAE 3: {mae3}')\n",
    "print(f'MSE 3: {mse3}')\n",
    "print(f'RMSE 3: {rmse3}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4d3b81d2-52b9-4c8f-9702-7aaee99246dc",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "08fe0050-4643-4c47-b57c-7e303908d440",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "60614868-a507-46be-b40f-6ea9d0867a71",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "988e8cbe-4b8e-41e7-8d83-730443ce8c69",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8c894986-2807-4f11-8245-71f532dafa80",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
