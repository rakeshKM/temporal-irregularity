from random import shuffle
from sklearn.metrics import *
num_epochs = 100


print('getting data')
execfile('use_data13.py')

print('Creating Network')
execfile('network13.py')

net,train_fn,pred_fn,test_fn = get_net()


train_error=[]
valid_error=[]
k=0
print('Starting Training')
for epoch in range(num_epochs):
	start_time = time.time()
	tr_bat = 0
	tr_err = 0
	for batch in get_train_batch(B):
		x_tr = batch
		net_output,loss2= train_fn(x_tr)
		tr_err +=loss2
		tr_bat += 1
	train_error.append(float(tr_err / tr_bat))
	vl_bat = 0
	vl_err = 0
	for batch in get_valid_batch(B):
		x_vl = batch
		vl_err += test1_fn(x_vl)
		vl_bat += 1
	valid_error.append(float(vl_err / vl_bat))

	print('Epoch {} of {} done in {:.2f}s'.format(epoch+1, num_epochs, time.time() - start_time))
	print('Training Error:\t\t{}\nValidation Error:\t{}\n'.format( tr_err / tr_bat , vl_err / vl_bat))
	if (epoch+1)%5 ==0 :
		np.savez("/home/rakesh/tad/exp13/model/weight13_epoch_%d_TI.npz" %(epoch), *lasagne.layers.get_all_param_values(net['deconv3']))

	f = open("training_err_anomaly_nextframe.txt", "a")
	f.write("%s\n" % train_error[k])
	f.close()
	f = open("valid_err_anomaly_nextframe.txt", "a")
	f.write("%s\n" % valid_error[k])
	f.close()
	k+=1

import matplotlib.pyplot as plt
plt.subplots_adjust(bottom=0.2)
p1 = plt.plot(train_error,'bo-',label='train_error')
p2 = plt.plot(valid_error,'go-',label='valid_error')
plt.legend()
plt.grid(True)
plt.savefig('/home/rakesh/tad/exp13/exp12_errorplot_anomaly_nextframe.jpg')   # save the figure to file


