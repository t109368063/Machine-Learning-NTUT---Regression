import random as rd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# original get data
# def getData(path):
#     df = pd.read_csv(path)
#     Y = np.array([])
#     dataset = np.array(df.values)
#     if "price" in df.columns:
#         Y = dataset[:, 1]
#         dataset = np.delete(dataset, 1, 1)
#     dataset = np.delete(dataset, 0, 1)
#     return dataset, Y

# transfer zipcode to one hot encoding
def getData(path):
    df = pd.read_csv(path)
    Y = np.array([])
    zip = pd.get_dummies(df['zipcode'])
    df = df.join(zip)
    #df = df.drop(columns=['id', 'zipcode', 'sale_yr', 'sale_month', 'sale_day'])
    df.drop(columns=['id', 'zipcode', 'sale_yr', 'sale_month', 'sale_day'])
    dataset = np.array(df.values)
    if "price" in df.columns:
        Y = dataset[:, 1]
        dataset = np.delete(dataset, 1, 1)
        #dataset = np.delete(dataset, 0, 1)
    dataset = np.delete(dataset, 0, 1)
    return dataset, Y

# Read training dataset into X and Y
X_train, Y_train = getData('./train-v3.csv')

np.savetxt('./X_train.csv', X_train, delimiter=',', fmt='%i')
# Read validation dataset into X and Y
X_valid, Y_valid = getData('./valid-v3.csv')

# Read test dataset into X
X_test, _ = getData('./test-v3.csv')


# Read dataset into X and Y
#df = pd.read_csv('./test-v3.csv', delim_whitespace=True, header=None)
#dataset = df.values
#dataset = dataset[1:]
#X = []
#for i in dataset:
#	for j in i:
#		test = [float(x) for x in j.split(',')]
#		X.append(test[1:21]+test[22:23])

#X_test = np.array(X)


def normalize(train,valid,test):
	tmp=train
	mean=tmp.mean(axis=0)
	std=tmp.std(axis=0)
	# print("tmp.shape=",tmp.shape)
	# print("mean.shape=",mean.shape)
	# print("std.shape=",std.shape)
	# print("mean=",mean)
	# print("std=",std)
	train=(train-mean)/std
	valid=(valid-mean)/std
	test=(test-mean)/std
	return train,valid,test

X_train,X_valid,X_test=normalize(X_train,X_valid,X_test)


from tensorflow import keras
import tensorflow as tf

model = keras.Sequential([
    keras.layers.Dense(40, input_dim=X_train.shape[1]),
    keras.layers.Dense(60, activation='relu'),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(150, activation='relu'),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(1)
])


model.compile(optimizer='adam',
              loss='mae')



#print('Training -----------')
#for step in range(500):
# 	x_batch, y_batch = extractBatch(X_train, Y_train, 30)
# 	for i in range(len(x_batch)):
# 		cost = model.train_on_batch(x_batch[i], y_batch[i])
# 	valid_cost = model.test_on_batch(X_valid, Y_valid)
# 	print('train cost: {0:e}, val. cost: {1:e}'.format(cost, valid_cost))

#model.fit(X_train, Y_train, batch_size=30, epochs=650, validation_data=(X_valid, Y_valid))

history = model.fit(X_train, Y_train, batch_size=30, epochs=50, validation_data=(X_valid, Y_valid))
model.save('h5/model.h5')

Y_predict = model.predict(X_test)

n = len(Y_predict) + 1
for i in range(1, n):
	b = np.arange(1, n, 1)
	b = np.transpose([b])
	Y = np.column_stack((b, Y_predict))

np.savetxt('./test.csv', Y, delimiter=',', fmt='%i')



plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss (Lower is better)')
plt.xlabel('Training Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig('loss.png')
plt.show()


model = keras.models.load_model('h5/model.h5') 

# 驗證模型
score = model.evaluate(X_train, Y_train, verbose=0)
score_val = model.evaluate(X_valid, Y_valid, verbose=0)

print("val_loss: {}".format(score))
print("loss: {}".format(score_val))


# 輸出結果
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])