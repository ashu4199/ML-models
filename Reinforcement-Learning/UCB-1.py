import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
data=pd.read_csv('../Dataset/Ads_CTR_Optimisation.csv')

N=500
d=10
ads_selected=[]
numbers_of_selections=[0]*d
sums_of_rewards=[0]*d
total_reward=0
# ******* UCB Algorithm 

for n in range(0,N):
   ad=0
   max_upper_bound=0
   for i in range(0,d):
      if numbers_of_selections[i]>0:
         avg_reward=sums_of_rewards[i]/numbers_of_selections[i]
         delta_i=math.sqrt(3/2 * math.log(n+1)/numbers_of_selections[i])
         upper_bound=avg_reward+delta_i
      else:
         upper_bound=1e400
      if(upper_bound>max_upper_bound):
         max_upper_bound=upper_bound
         ad=i
   ads_selected.append(ad)
   numbers_of_selections[ad]+=1
   sums_of_rewards[ad]+=data.values[n,ad]
   total_reward=total_reward+data.values[n,ad]
   
plt.hist(ads_selected)
plt.title('Histogram of ads selection')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show() 