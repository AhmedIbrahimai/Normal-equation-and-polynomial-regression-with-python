import pandas as pd
import numpy as np
import os 

os.chdir(r"C:\Users\amb\Downloads")

dataset = pd.read_csv('market_data.csv')
dataset

after_drop = dataset.drop('sales', axis =1 )
y = dataset['sales']

dataset.shape


np_array = after_drop.to_numpy()


x1 = dataset['youtube']
x2 = dataset['facebook']
x3 = dataset['newspaper']
#y =  dataset['sales']

n= len(x1)      
x_bais= np.ones((n,1))

x_new = np.append(x_bais ,np_array, axis = 1)   # X


x_new_transpose = np.transpose(x_new)  #X.T

x_new_transpose_dot_x_new = x_new_transpose.dot(x_new)   # (X.T X)

temp_1 = np.linalg.inv(x_new_transpose_dot_x_new) #(X X.T) inverse

temp_2 = x_new_transpose.dot(y)   # X.T * y

theta = temp_1.dot(temp_2)  #(X X.T) inverse * X.T * y



theta_0 = theta[0]
theta_1 = theta[1]
theta_2 = theta[2]
theta_3 = theta[3]
print(theta_0)
print(theta_1)
print(theta_2)
print(theta_3)


def pred_values(theta_0,theta_1,theta_2,theta_3,youtube,facebook,newspaper):
    predicted_value = youtube * theta_1 + facebook * theta_2 + newspaper * theta_3 + theta_0   
    return predicted_value


youtube = 84.72
facebook = 19.20
newspaper = 48.96
print(pred_values(theta_0,theta_1,theta_2,theta_3,youtube,facebook,newspaper))


youtube =351.48
facebook = 33.96
newspaper = 51.84
print(pred_values(theta_0,theta_1,theta_2,theta_3,youtube,facebook,newspaper))


youtube = 29
facebook = 93
newspaper = 96
print(pred_values(theta_0,theta_1,theta_2,theta_3,youtube,facebook,newspaper))









