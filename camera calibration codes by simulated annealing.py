from msilib.schema import Complus
from re import T
import time
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
#------------------------------------------------------------------------------
# Customization section:
initial_temperature = 100
cooling =2  # cooling coefficient
number_variables = 7
upper_bounds =[2.496033296 ,	  2.49978129,    1.923062286,	 2.093090958,	 2.160555887,	 1.899159365,	 1.782515064,	  5.38492541]  
lower_bounds =[-1.790766259,	-1.763715096,	-1.680438686,	-1.661196131,	-1.633221195,	-2.837427373,	-2.329164736,	-0.205754863]     
computing_time = 30 # second(s)
#######
#upper_bounds = [6,5,6,2,3, 7, 4, 8,9] 
#lower_bounds = [ -3, -3, -3, -3,-3,-3, -3, -3,-3]  





 ######################################################################################## 
def objective_function(X):
    fx=X[0]
    cx=X[1]
    cy=X[2]
    k1=X[3]
    k2=X[4]
    p1=X[5]
    p2=X[6]
    

    #Object points in (mm) unit
    Points = np.float32([[0.0,0.0,0.0], [162.8000,0.0000,0.0000], [324.5999,0.2058,0.0000], [-0.3667,-216.9997,0.0000],[161.8302,-217.9978,0.0000] ,[323.2301,-217.7899,0.0000]])
    objectPoints=[]
    objectPoints.append(Points)
      
    #image points in (pixel unit)
    imPoints1 = np.float32([[719, 377], [1701, 368], [2682, 346], [729, 1696], [1737, 1692], [2699, 1669]])
    imagePoints1=[]
    imagePoints1.append(imPoints1)
    ##image points in (pixel unit)
    imPoints2 = np.float32([[2400, 1837], [3496, 1851], [4618, 1875], [2360, 3338], [3496, 3371], [4612, 3380]])
    imagePoints2=[]
    imagePoints2.append(imPoints2)


    device_1=[]
    rmse=[]




    cameraMatrix1 =np.array([[fx                    ,0.000000000000000000e+00,                       cx],
                             [0.000000000000000000e+00,fx                    ,                       cy],
                             [0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]])
    cameraMatrix2 =np.array([[fx                    ,0.000000000000000000e+00,                       cx],
                             [0.000000000000000000e+00,fx                    ,                       cy],
                             [0.000000000000000000e+00, 0.000000000000000000e+00,1.000000000000000000e+00]])                       

   
    distCoeffs1 = np.array([[ k1                    ,k2                       ,p1                    ,p2                      ,0                    ]])
    distCoeffs2 = np.array([[ k1                    ,k2                       ,p1                    ,p2                      ,0                    ]])
   

    imageSize=(5056,3792)                 # According to camera type
    criteria = (cv2.TermCriteria_COUNT+cv2.TermCriteria_EPS, 3, 3)
    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F, perViewErrors=cv2.stereoCalibrateExtended(	objectPoints, imagePoints1, imagePoints2, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize,R=None,T=None,E=None,F=None,flags=cv2.CALIB_USE_INTRINSIC_GUESS,criteria=(cv2.TermCriteria_COUNT+cv2.TermCriteria_EPS, 3, 3))

  

     

    rmse.append(perViewErrors)
    RMSE=np.array(rmse).ravel() 
   #print(RMSE) 
    device_1= RMSE[::2] 
    device_2= RMSE[1::2]
    
       # print result
    res_list = []
    for i in range(0, len(device_1)):
        res_list.append(((device_1[i]**2) + (device_2[i]**2))**0.5)
        value=np.array(res_list).flatten()      
    
    return value
  
#------------------------------------------------------------------------------
# Simulated Annealing Algorithm:
initial_solution=np.zeros((number_variables))
for v in range(number_variables):
    initial_solution[v] = random.uniform(lower_bounds[v],upper_bounds[v])
      
current_solution = initial_solution
best_solution = initial_solution
n = 1  # no of solutions accepted
best_fitness = objective_function(best_solution)
current_temperature = initial_temperature # current temperature
start = time.time()
no_attempts = 100 # number of attempts in each level of temperature
record_best_fitness =[]
  
for i in range(9999999):
    for j in range(no_attempts):
  
        for k in range(number_variables):
            current_solution[k] = best_solution[k] -  (random.uniform(lower_bounds[k],upper_bounds[k]))
            current_solution[k] =max(min(current_solution[k],upper_bounds[k]),lower_bounds[k])


        current_fitness=objective_function(current_solution)
        E=abs(current_fitness-best_fitness)
        if i==0 and j==0:
            EA=E
        
        if current_fitness>best_fitness:
            p=math.exp(-E/(EA*current_temperature))
            #make a decision to accept the worse solution or not
            if random.random ()>p:
                accept=True  #this worse solution is accepted
            else:
                accept=False #this worse solution is not accepted
        else:
            accept=True #accept better solution 

        if accept==True:
            best_solution=current_solution    #update the best solution 
            best_fitness=objective_function(best_solution)
            n=n+1   #count the solution accepted
            EA =(EA*(n-1)+E)/n    #update EA

    print('interation: {},best_solution: {},best_fitness: {},'.format(i,best_solution,best_fitness))
    record_best_fitness.append(best_fitness)
    ##cooling the tempreture
    current_temperature=current_temperature*cooling
    #stop by computing time
    end =time.time()
    if end-start>=computing_time:
        break

plt.plot(record_best_fitness)
plt.show()



z=[i,best_solution,best_fitness]
print(best_solution)

record_best_fitness_array=np.array(record_best_fitness)
rerprojectin_error =np.array(record_best_fitness_array).flatten()
print(rerprojectin_error)


min_RMSE=min(rerprojectin_error)
print("min_RMSE     :", min_RMSE)

#index number for (min_RMSE)
index_no=np.argmin(rerprojectin_error)
print("index_no_min     :",index_no)

r=np.array([31.14623852,	18.12021094,	20.17441574,	0.007189079,	0.057616969,	0.000970024,	0.108826018])
q=np.array(best_solution)
w=np.array([3692.251492,	2530.883266,	1899.014451,	0.251236743,	-0.71715457,	0.002371288,	0.020507206])

zx=r*q+w
print(zx)