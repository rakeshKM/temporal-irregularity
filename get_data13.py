import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as misc
import os
import random


num_row = 227
num_col = 227

data_root_Train = '/home/rakesh/tad/UCSD/UCSDped2/Train'
folder_list=[os.path.join(data_root_Train, f) for f in os.listdir(data_root_Train)]
folder_list=sorted(folder_list)
Data_train = None
for fol in folder_list:
	flst = sorted([f for f in os.listdir(fol) if f.endswith('tif')])
	temp = [misc.imresize(plt.imread(os.path.join(fol, f)),(num_row,num_col)) for f in flst]
	temp = np.array(temp, dtype='uint8')
	Data_train_temp = []
	bat = temp.shape[0] - 10
	for i in range(bat):
		Data_train_temp.append(temp[i:i+10])
	Data_train_temp = np.array(Data_train_temp)
	Data_train_temp = np.reshape(Data_train_temp, (Data_train_temp.shape[0], 1, Data_train_temp.shape[1], Data_train_temp.shape[2], Data_train_temp.shape[3]))
	if Data_train is None:
		Data_train = Data_train_temp
	else:		
		Data_train = np.concatenate((Data_train, Data_train_temp), axis=0)	


# creating data with temporal stride 1
Data_train_stride1 = None
for fol in folder_list:
	flst = sorted([f for f in os.listdir(fol) if f.endswith('tif')])
	temp = [misc.imresize(plt.imread(os.path.join(fol, f)),(num_row,num_col)) for f in flst]
	temp = np.array(temp, dtype='uint8')
	Data_train_temp = []
	bat = temp.shape[0] - 20
	for i in range(bat):
		Data_train_temp.append(temp[i:i+20:2])
	Data_train_temp = np.array(Data_train_temp)
	Data_train_temp = np.reshape(Data_train_temp, (Data_train_temp.shape[0], 1, Data_train_temp.shape[1], Data_train_temp.shape[2], Data_train_temp.shape[3]))
	if Data_train_stride1 is None:
		Data_train_stride1 = Data_train_temp
	else:		
		Data_train_stride1 = np.concatenate((Data_train_stride1, Data_train_temp), axis=0)	

Data_train = np.concatenate((Data_train, Data_train_stride1), axis=0)

# creating data with temporal stride 2
Data_train_stride2 = None
for fol in folder_list:
	flst = sorted([f for f in os.listdir(fol) if f.endswith('tif')])
	temp = [misc.imresize(plt.imread(os.path.join(fol, f)),(num_row,num_col)) for f in flst]
	temp = np.array(temp, dtype='uint8')
	Data_train_temp = []
	bat = temp.shape[0] - 30
	for i in range(bat):
		Data_train_temp.append(temp[i:i+30:3])
	Data_train_temp = np.array(Data_train_temp)
	Data_train_temp = np.reshape(Data_train_temp, (Data_train_temp.shape[0], 1, Data_train_temp.shape[1], Data_train_temp.shape[2], Data_train_temp.shape[3]))
	if Data_train_stride2 is None:
		Data_train_stride2 = Data_train_temp
	else:		
		Data_train_stride2 = np.concatenate((Data_train_stride2, Data_train_temp), axis=0)	

Data_train = np.concatenate((Data_train, Data_train_stride2), axis=0)


# creating data with temporal stride 3

Data_train_stride3 = None
for fol in folder_list:
	flst = sorted([f for f in os.listdir(fol) if f.endswith('tif')])
	temp = [misc.imresize(plt.imread(os.path.join(fol, f)),(num_row,num_col)) for f in flst]
	temp = np.array(temp, dtype='uint8')
	Data_train_temp = []
	bat = temp.shape[0] - 40
	for i in range(bat):
		Data_train_temp.append(temp[i:i+40:4])
	Data_train_temp = np.array(Data_train_temp)
	Data_train_temp = np.reshape(Data_train_temp, (Data_train_temp.shape[0], 1, Data_train_temp.shape[1], Data_train_temp.shape[2], Data_train_temp.shape[3]))
	if Data_train_stride3 is None:
		Data_train_stride3 = Data_train_temp
	else:		
		Data_train_stride3 = np.concatenate((Data_train_stride3, Data_train_temp), axis=0)	

Data_train = np.concatenate((Data_train, Data_train_stride3), axis=0)

#lshp = np.arange(Data_train.shape[0])
#print Data_train.shape[0]
#random.shuffle(lshp)
np.savez('/home/rakesh/tad/exp13/data_train.npz',Data_train)


