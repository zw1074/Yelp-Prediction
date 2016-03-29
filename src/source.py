import json
import os
import numpy as np
import pandas as pd
from kmeans import *
from sklearn import mixture
from data_constructor import *


def data_saver(b_id=[],opt = 'biz'):
#save data_source into json libraries
    if opt == 'biz':
        data_source = []
        l_c= 0
        with open('yelp_academic_dataset_business.json') as data_file:    
            for line in data_file:
                res = json.loads(str(line))
                if (res['longitude'] < -109) and (res['longitude'] > -114) and (res['latitude'] < 37) and (res['latitude'] > 31):
                    data_source.append(res)
                    l_c+=1
                    print("loading line: "+ str(l_c))                
        return data_source
    
    elif opt == 'rev':
        data_source=  []
        l_c= 0
        with open('yelp_academic_dataset_review.json') as data_file:    
            for line in data_file:
                res = json.loads(str(line)) 
                if res['business_id'] in b_id:
                    data_source.append(res)
                    l_c+=1
                    print("loading line: "+ str(l_c))                     
        return data_source        
    
    
 
def categories_selector(data_source,tags):
    data = []
    b_id = []
    for i in data_source:
        if any(tag in i['categories'] for tag in tags):
            data.append(i)
            b_id.append(i['business_id'])
    return (data,b_id)



def loc_saver(filename,data_source):
# save location for clustering into 'filename.txt' with 2 column 
    msg = ""
    l_c =0
    if os.path.exists(filename):
        while msg not in ['Y','N']:
            print("please input Y/N")
            msg = str(raw_input("file exist, Y for cover,N for quit \n"))
            print("\n")
    if msg == "N":
        return
    else:
        if os.path.exists(filename):
            os.remove(filename)
        loc_1 = []
        loc_2 = []
        fobj = open(filename,'w+')
        for i,j in enumerate(data_source):
            loc_1.append(j['longitude'])
            loc_2.append(j['latitude'])
            fobj.write(str(loc_1[i])+',' +str(loc_2[i]))
            fobj.write('\n')
            l_c+=1
            print("loading line: "+ str(l_c)) 
    return [loc_1,loc_2]
            
def user_data_saver():
    #Save user data into numpy object
    data_source = []
    l_c = 0
    with open('/Users/cryan/Desktop/1001project/data/yelp_academic_dataset_user.json') as data_file:    
        for line in data_file:
            res = json.loads(str(line))
            data_source.append(res.values())
            l_c+=1
            print("loading line: "+ str(l_c))
    return np.array(data_source)


