import numpy as np

B =10
C=10
W=227
H=227

# load means
frame_mean_path='/home/rakesh/tad/exp13/mean_frame_227_UCSDped1.npy'
mean_frame = np.load(frame_mean_path)

#pre-process
data = np.load('/home/rakesh/tad/exp13/data_train.npz')['arr_0']
data = np.array(data, dtype='float16')
data *= 1.0/255.
#data=data-mean_frame
t_split = int(0.7 * data.shape[0])
v_split = data.shape[0] - t_split

def get_train_batch(batchsize):
	global data, t_split
	train_set = data[:t_split]
	batch = (train_set.shape[0] / batchsize) + (train_set.shape[0] % batchsize > 0)
	for i in range(batch-1):
		yield np.reshape(train_set[i*batchsize : (i+1)*batchsize,:, :10],(-1,C,train_set.shape[-2], train_set.shape[-1]))


def get_valid_batch(batchsize):
	global data, t_split, v_split
	valid_set = data[t_split:]
	batch = (valid_set.shape[0] / batchsize) + (valid_set.shape[0] % batchsize > 0)
	for i in range(batch-1):
		yield np.reshape(valid_set[i*batchsize : (i+1)*batchsize,:, :10],(-1,C,valid_set.shape[-2], valid_set.shape[-1]))


