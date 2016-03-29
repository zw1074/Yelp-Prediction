import matplotlib.pyplot as plt
import re
import numpy as np
# Read file
accurate_test = []
accurate_train = []
# fobj = open('ex5_test.rtf')
# print re.findall(r'(?<=\s)[0-9]\.[0-9]{4}e\+[0-9]{2}',fobj.read())
for i,name in enumerate(['Original-two layers', 'ReLU function', 'Sigmoid function', 'kernel size 2', '1 conv+1 fully connect', 'without normalization']):
	fobj = open('ex'+str(i+1)+'_test.rtf')
	accurate_test.append(100 - float(re.findall(r'(?<=\s)[0-9]\.[0-9]{4}e\+[0-9]{2}',fobj.read())[-1]))
	fobj.close()
	fobj = open('ex'+str(i+1)+'_train.rtf')
	accurate_train.append(100 - float(re.findall(r'(?<=\s)[0-9]\.[0-9]{4}e\+[0-9]{2}',fobj.read())[-1]))
	fobj.close()
fobj = open('test.log')
accurate_test.append(100 - float(fobj.readlines()[-1]))
fobj.close()
fobj = open('train.log')
accurate_train.append(100 - float(fobj.readlines()[-1]))
fobj.close()
plt.subplots(figsize=(10,8))
bar_width = 0.3    
opacity   = 0.4
index = np.arange(7)
rects1 = plt.bar(index, accurate_train,    bar_width, alpha=opacity, color='r', label='Train')
rects2 = plt.bar(index + bar_width, accurate_test, bar_width,alpha=opacity,color='b',label='Test')  
plt.xlabel('Experiment name')
plt.ylabel('Global Incorrect(%)')
plt.legend()
plt.xticks(index + bar_width, ('Original\ntwo layers', 'ReLU \nfunction', 'Sigmoid \nfunction', 'kernel \nsize 2', '1 fully-connect\n1 conv', 'without \nnormalize', 'Final Model'))
plt.savefig('Experiment.png')
plt.show()
