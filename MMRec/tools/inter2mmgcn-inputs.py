import numpy as np
import pandas as pd

dataset_name = 'sports'

dataset_path = '../../data/'+dataset_name+'/'
ui = pd.read_csv(dataset_path+dataset_name+'.inter', sep='\t')

print(ui.shape)
max_user_id = max(ui['userID'].values)
ui['itemID'] = ui['itemID'].map(lambda x: x + max_user_id + 1)

traindict=ui.loc[ui['x_label'] == 0]
inter=traindict.groupby('userID')
user_item_dict={}
for i in inter:
    user_item_dict[i[0]] = i[1]['itemID'].values.tolist()

#print(user_item_dict)
np.save(dataset_path + '/user_item_dict_sample.npy',user_item_dict)

test=ui.loc[ui['x_label'] == 2]
m=test.groupby('userID')
te=[]
for i in m:
    list=[]
    list.append(i[0])
    list.extend(i[1]['itemID'].values.tolist())
    te.append(list)

#print(te)
np.save(dataset_path + 'test_sample.npy',te)

val=ui.loc[ui['x_label'] == 1]
n=val.groupby('userID')
va=[]
for i in n:
    list=[]
    list.append(i[0])
    list.extend(i[1]['itemID'].values.tolist())
    va.append(list)

#print(va)
np.save(dataset_path + 'val_sample.npy',va)
#
tr=[]
train=ui.loc[ui['x_label'] == 0]

for idx,row in train.iterrows():
    list=[]
    #print(row['userID'])
    list.append(row['userID'])
    list.append(row['itemID'])
    tr.append(list)
#print(tr)
np.save(dataset_path + 'train_sample.npy',tr)
print('done!!!')

