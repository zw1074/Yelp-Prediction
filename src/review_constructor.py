from cluster import *
from collections import *


def rating_class(x):
    t = x -np.array([3,3.5,4,4.5,5])
    return int(np.where(t<=0)[0][0])+1


f2 = open('review.txt','rb')
review = pickle.load(f2)
f2.close()

bus_review = defaultdict(list)

for ele in review:
    bus_review[ele['business_id']].append([ele['stars'], np.datetime64(ele['date']),ele['text']])



cnt = Counter()
for ele in review:
    cnt[np.datetime64(ele['date'])]+=1
    
date = []
count = []
for i in cnt.items():
    date.append(i[0])
    count.append(i[1])

date_sta = np.sort(np.transpose(np.array([date,count])),axis = 0)
plt.plot(date,count)
plt.plot(date_sta[:,0],date_sta[:,1],color = 'r',lw = 2)


#transfer to 4 year slot
bus_year= defaultdict(list)
for bis in bus_review.keys():
    bus_year[bis]= [[],[],[],[]]
    for item in bus_review[bis]:
        if item[1] <= np.datetime64('2011-12-31'):
            bus_year[bis][0].append([item[0],item[2]])
        elif item[1] <= np.datetime64('2012-12-31'):
            bus_year[bis][1].append([item[0],item[2]])
        elif item[1] <= np.datetime64('2013-12-31'):
            bus_year[bis][2].append([item[0],item[2]])  
        else:
            bus_year[bis][3].append([item[0],item[2]])  


bus_rating =  defaultdict(list)  
for bis in bus_year.keys():
    bus_rating[bis]= [[],[],[],[]]
    for  i in list(xrange(4)):
        if bus_year[bis][i]:
            temp = np.array(bus_year[bis][i])[:,0].astype(int)
            bus_rating[bis][i].extend([len(temp),rating_class(np.mean(temp))])
        else:
            continue
        
combine_rating = []
for eles in bus_rating.items():
    for i,ele in enumerate(eles[1]):
        if (i==1) and (eles[1][0] != []) and (eles[1][1] != []):
            combine_rating.append([eles[0],int(eles[1][0][1]),int(eles[1][i][1]),int(eles[1][0][0]),int(eles[1][i][0])])
        if (i == 2) and (eles[1][1] != []) and (eles[1][2] != []):
            combine_rating.append([eles[0],int(eles[1][1][1]),int(eles[1][i][1]),int(eles[1][1][0]),int(eles[1][i][0])])
        if (i == 3) and (eles[1][2] != []) and (eles[1][3] != []):
            combine_rating.append([eles[0],int(eles[1][2][1]),int(eles[1][i][1]),int(eles[1][2][0]),int(eles[1][i][0])])
combine_rating = np.array(combine_rating)

np.save("review_combined", combine_rating)