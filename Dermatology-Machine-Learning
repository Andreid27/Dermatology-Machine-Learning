#libraries

import numpy as np 
from sklearn import svm
import random
from sklearn.metrics import classification_report


#loadind the data
data = np.genfromtxt('dermatology.csv',delimiter=',')
target_names = ['psoriasis', 'seboreic dermatitis', 'lichen planus','pityriasis rosea', 'chronic dermatitis', 'pityriasis rubra pilaris  ']
random.shuffle(data)

X=np.array(data[:366,:33]) 
Y=np.array(data[:366,34]) 

#in total am 366 de linii(date)=> 75%=275->train si 25%=91->test
X_train=X[275:]
Y_train=Y[275:]
X_test=X[:275]
Y_test=Y[:275]

#variem costul
cost = [1/32,1/8,1/2,2,8,32,128]

for i in range(len(cost)):
    count = 0;
    #antrenez un svm pentru clasificare (svc)
    clf = svm.SVC(kernel='linear',C=cost[i]).fit(X_train, Y_train)
    #pentru a verifica ce prezice sistemul, am aplicat metoda predict care are ca parametru X_test cu date de testare
    pred=clf.predict(X_test)
    for j in range(len(pred)):
        if Y_test[j]==pred[j]:
         count+=1
    print('Pentru un cost de ' +str(cost[i]))
    print(classification_report(Y_test, pred, target_names=target_names))
