import numpy as np
import matplotlib.pyplot as plt
def getTax(x):
    
    tax = 0
    rate1=0.0
    rate2=0.0
    dedu1=0
    dedu2=0
    if x-4045 <= 0:
       rate1=0.0
       dedu1=0
    elif 0<x-4045<=1500:
        rate1=0.03
        dedu1=0
    elif 1500<x-4045<=4500:
        rate1=0.1
        dedu1=105   
    elif 4500<x-4045<=9000:
        rate1=0.2
        dedu1=555
    elif 9000<x-4045<=35000:
        rate1=0.25
        dedu1=1005
    elif 35000<x-4045<=55000:
        rate1=0.3
        dedu1=2755
    elif 55000<x-4045<=80000:
        rate1=0.35
        dedu1=5505
    else:
        rate1=0.45
        dedu1=13505    
        
    if 7855-x <= 0:
       rate2=0.0
       dedu2=0
    elif 0<7855-x<=1500:
        rate2=0.03
        dedu2=0
    elif 1500<7855-x<=4500:
        rate2=0.1
        dedu2=105   
    elif 4500<7855-x<=9000:
        rate2=0.2
        dedu2=555
    elif 9000<7855-x<=35000:
        rate2=0.25
        dedu2=1005
    elif 35000<7855-x<=55000:
        rate2=0.3
        dedu2=2755
    elif 55000<7855-x<=80000:
        rate2=0.35
        dedu2=5505
    else:
        rate2=0.45
        dedu2=13505 
    
       
    tax = (x-4045)*rate1-dedu1+(7855-x)*rate2-dedu2
    
    return tax
t=np.arange(1,12800,1)
y=[]

for i in t:
    y_1=getTax(i)
    y.append(y_1)

# min_indx=np.argmin(y)#min value index
for i in t:
    if getTax(i) == min(y):
       print( i)
# print(y.index(min(y))) 
print("min tax == ",min(y))
# print(y.index(171.0)) 
plt.plot(t,y)
plt.xlabel("salary")
plt.ylabel("tax")
plt.title("findMinTax")
plt.show()
