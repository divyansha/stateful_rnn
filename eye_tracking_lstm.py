import numpy as np
import datetime
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input, merge
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
import pandas as pd


input_tensor_x = np.empty([7, 50001])
input_tensor_y = np.empty([7, 50001])

atoms4 = pd.read_csv('atoms4 pre-processed.csv')
atoms9 = pd.read_csv('atoms9 pre-processed.csv')
atoms10 = pd.read_csv('atoms10 pre-processed.csv')
atoms14 = pd.read_csv('atoms14 pre-processed.csv')
atoms18 = pd.read_csv('atoms18 pre-processed.csv')
atoms23 = pd.read_csv('atoms23 pre-processed.csv')
atoms26 = pd.read_csv('atoms26 pre-processed.csv')
atoms33 = pd.read_csv('atoms33 pre-processed.csv')
atoms49 = pd.read_csv('atoms49 pre-processed.csv')


input_tensor_x[0] = atoms10['L_Raw_X'][:50001]
input_tensor_x[1] = atoms14['L_Raw_X'][:50001]
input_tensor_x[2] = atoms23['L_Raw_X'][:50001]
input_tensor_x[3] = atoms26['L_Raw_X'][:50001]
input_tensor_x[4] = atoms4['L_Raw_X'][:50001]
input_tensor_x[5] = atoms9['L_Raw_X'][:50001]
input_tensor_x[6] = atoms18['L_Raw_X'][:50001]


input_tensor_y[0] = atoms10['L_Raw_Y'][:50001]
input_tensor_y[1] = atoms14['L_Raw_Y'][:50001]
input_tensor_y[2] = atoms23['L_Raw_Y'][:50001]
input_tensor_y[3] = atoms26['L_Raw_Y'][:50001]
input_tensor_y[4] = atoms4['L_Raw_Y'][:50001]
input_tensor_y[5] = atoms9['L_Raw_Y'][:50001]
input_tensor_y[6] = atoms18['L_Raw_Y'][:50001]

atoms28 = pd.read_csv('atoms28 pre-processed.csv')
atoms38 = pd.read_csv('atoms38 pre-processed.csv')
#atoms33 = pd.read_csv('atoms33 pre-processed.csv')

test_tensor_x = np.zeros([4, 50001])
test_tensor_y = np.zeros([4, 50001])

test_tensor_x[0] = atoms28['L_Raw_X'][:50001]
test_tensor_x[1] = atoms38['L_Raw_X'][:50001]
test_tensor_x[2] = atoms33['L_Raw_X'][:50001]
test_tensor_x[3] = atoms49['L_Raw_X'][:50001]

test_tensor_y[0] = atoms10['L_Raw_Y'][:50001]
test_tensor_y[1] = atoms14['L_Raw_Y'][:50001]
test_tensor_y[2] = atoms33['L_Raw_Y'][:50001]
input_tensor_y[3] = atoms49['L_Raw_Y'][:50001]



def create_dataset(data, num_timesteps):
    dataX, dataY = [], []
    #iterate over no. of samples
    for i in range(data.shape[0]):
        dataX.append(data[i][:-num_timesteps])
        dataY.append(data[i][num_timesteps:])
    print(np.array(dataX).shape), print(np.array(dataY).shape)
    return np.array(dataX), np.array(dataY)

#trainX is x coordinates, trainY is y labels. trainX_x is input x coordinates, trainX_y is labels for x coordinates
trainX_x, trainY_x = create_dataset(input_tensor_x, 1)
trainX_y, trainY_y = create_dataset(input_tensor_y, 1)

train2d = []
for i in range(len(trainX_x)):
    train2d.append(np.array([trainX_x[i], trainX_y[i]]))
train2d = np.array(train2d)

labels2d = []
for i in range(len(trainY_x)):
    labels2d.append(np.array([trainY_x[i], trainY_y[i]]))
labels2d = np.array(labels2d)


testX_x, testY_x = create_dataset(test_tensor_x, 1)
testX_y, testY_y = create_dataset(test_tensor_y, 1)

test2d = []
for i in range(len(testX_x)):
    test2d.append(np.array([testX_x[i], testX_y[i]]))
test2d = np.array(test2d)

test_labels2d = []
for i in range(len(testY_x)):
    test_labels2d.append(np.array([testY_x[i], testY_y[i]]))
test_labels2d = np.array(test_labels2d)

#testX, testY = create_dataset(test_set, 1)
'''newShape = list(input_tensor_x.shape)

newShape[1] = newShape[1] - 1
newShape = tuple(newShape)
trainX_x = np.reshape(trainX_x, newShape)
trainY_x = np.reshape(trainY_x, newShape)
trainX_y = np.reshape(trainX_y, newShape)
trainY_y = np.reshape(trainY_y, newShape)'''
'''except ValueError:
    print("Got an error")
    import sys
    sys.exit("Error message")'''
'''try:
    trainY = np.reshape(trainY_x, (1, trainY.shape[1], 1))
except ValueError:
    print(data.shape)
    print(trainY.shape)
    import sys
    sys.exit("Error message")'''

#testX = np.reshape(testX, (testX.shape[0], 1, 1))'''


#create and fit LSTM network
def construct_model(hidden_size, layers, lrate, opt, trainX_x, trainX_y):
    model = Sequential()
    for i in range(layers):
        model.add(LSTM(hidden_size, batch_input_shape=(4, 50000, 2), return_sequences=True, dropout_W=0.2))
    model.add(TimeDistributed(Dense(2)))
    model.add(Activation('sigmoid'))
    opt = opt(lr=lrate)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
    return model


def construct_chopped_nonstateful(hidden_size, layers, lrate, opt, window_size):
    model = Sequential()
    for i in range(layers):
        model.add(LSTM(hidden_size, batch_input_shape=(32, window_size, 2), return_sequences=True, dropout_W=0.2))
    model.add(TimeDistributed(Dense(2)))
    model.add(Activation('sigmoid'))
    opt = opt(lr=lrate)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
    return model

def construct_stateful(hidden_size, layers, lrate, opt, window_size):
    #e_size = e_size
    #HIDDEN_SIZE = hidden_size
    #LAYERS = layers
    lrate = lrate
    print('building a model')
    model = Sequential()

    #window_size = int(max_len / 10)

    for i in range(layers):
        model.add(LSTM(hidden_size, batch_input_shape=(4, window_size, 2), return_sequences=True, dropout_W=0.2, stateful=True))

    # if i == (LAYERS - 1):
    #            model.add(Dropout(.2))
    model.add(TimeDistributed(Dense(2)))
    model.add(Activation('sigmoid'))
    opt = opt(lr=lrate)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
    return model

    '''x_input = Input(shape=(49999, 1), dtype='float32', name='x_input')
    #print(x_input.input_shape)
    #print(x_input.shape)
    y_input = Input(shape=(49999, 1), dtype='float32', name='y_input')

    lstm_x_out = LSTM(1, input_shape=(49999, 1), return_sequences=True)(x_input)
    lstm_y_out = LSTM(1, input_shape=(49999, 1), return_sequences=True)(y_input)

    merged_vector = merge([lstm_x_out, lstm_y_out], mode='concat')
    predictions = TimeDistributed(Dense(2, activation='sigmoid'))(merged_vector)

    model = Model(input=[x_input, y_input], output=predictions)'''


'''def compute_val_acc(self, model):
    test_x = self.testX
    test_y = self.test_padded_y_windows
    validation_i = self.validation_i
    vocab_size = self.vocab_size
    max_len = self.max_len
    padding_size = self.padding_size
    # padded_y_windows = self.padded_y_windows


    print("Computing Validation Hill Climbing Accuracy")
    # test_x = testX[validation_i:]
    # test_y = testY[validation_i:]
    predictions = model.predict(test_x, batch_size=8)

    # per_student_accuracies = []

    student_correct = np.zeros(len(test_x))
    student_incorrect = np.zeros(len(test_x))

    total_correct = 0
    total_incorrect = 0

    for i, x in enumerate(test_x):
        current_predictions = list(predictions[i])
        current_answers = list(test_y[i])
        current_student_correct = 0
        current_student_incorrect = 0
        for j in range(len(x)):
            current_value = x[j]
            if current_value == 0:
                continue
            else:
                current_softmax = list(current_predictions[j])
                best_prediction = current_softmax.index(max(current_softmax)) + 1
                correct_answer = current_answers[j]
                if best_prediction == correct_answer:
                    student_correct[i] += 1
                    # current_student_correct += 1
                    total_correct += 1
                else:
                    student_incorrect[i] += 1
                    # current_student_incorrect += 1
                    total_incorrect += 1'''


def compute_val_acc(model, test2d, test_labels2d):
    print("Computing Validation Hill Climbing Accuracy")
    # test_x = testX[validation_i:]
    # test_y = testY[validation_i:]
    predictions = model.predict(test2d, batch_size=4)

    euclid_distance = 0
    vals = []

    for i, x in enumerate(test2d):
        current_predictions = list(predictions[i])
        current_answers = list(test_labels2d[i])

        for j in range(len(x)):
            current_value = x[j]
            '''if current_value[0] == 0:
                continue
            else:'''
            current_pred = list(current_predictions[j])
            current_answer = list(current_answers[j])
            curr_dist = (current_pred[0] - current_answer[0])**2 + (current_pred[1] - current_answer[1])**2
            euclid_distance += curr_dist
            vals.append(curr_dist)
        '''try:
            acc = float(current_student_correct) / (current_student_incorrect + current_student_correct)
            per_student_accuracies.append(acc)

        except ZeroDivisionError:
            continue'''
    euclid_distance = np.sqrt(euclid_distance)
    return euclid_distance/(len(test2d) * len(test2d[0])), vals


def compute_val_acc_chopped(model, test2d, test_labels2d):
    print("Computing Validation Hill Climbing Accuracy")
    # test_x = testX[validation_i:]
    # test_y = testY[validation_i:]
    predictions = model.predict(test2d, batch_size=32)

    euclid_distance = 0

    vals = []

    for i, x in enumerate(test2d):
        current_predictions = list(predictions[i])
        current_answers = list(test_labels2d[i])

        for j in range(len(x)):
            current_value = x[j]
            '''if current_value[0] == 0:
                continue
            else:'''
            current_pred = list(current_predictions[j])
            current_answer = list(current_answers[j])
            curr_dist = (current_pred[0] - current_answer[0])**2 + (current_pred[1] - current_answer[1])**2
            euclid_distance += curr_dist
            vals.append(curr_dist)
    euclid_distance = np.sqrt(euclid_distance)
    return euclid_distance/(len(test2d) * len(test2d[0])), vals

def compute_val_acc_stateful(model, test2d, test_labels2d, batchsize, window_size):
    euclid_distance = 0
    vals = []
    for j in range(0, len(test2d), batchsize):
        for k in range(0, len(test2d[0]), window_size):  # go over all the sequences
            # if(k + num_timesteps <= max_len):

            test_x = test2d[j:j + batchsize, k:k + window_size]
            # else:
            # print("Test padded y windows", self.test_padded_y_windows[0])

            test_y = test_labels2d[j: j + batchsize, k: k + window_size]

            if (len(test_x) < batchsize):
                # curr =
                diff = batchsize - len(test_x)
                temp = np.zeros((diff, window_size))
                test_x = np.concatenate((test_x, temp))
                test_y = np.concatenate((test_y, temp))
                # continue

            predictions = model.predict_on_batch(test_x)  # predicts for batch size of 4
            for i, x in enumerate(test_x):  # go over each row/student
                current_predictions = list(predictions[i])
                current_answers = list(test_y[i])

                for l in range(len(x)):
                    current_value = x[l]
                    current_answer = list(current_answers[j])
                    '''if current_value[0] == 0:
                        continue'''
                    #else:
                    current_pred = list(current_predictions[l])
                    curr_dist = (current_pred[0] - current_answer[0]) ** 2 + (current_pred[1] - current_answer[1]) ** 2
                    euclid_distance += curr_dist
                    vals.append(curr_dist)
        model.reset_states()
    euclid_distance = np.sqrt(euclid_distance)
    return euclid_distance/(len(test2d) * len(test2d[0])), vals


def baseline(model, test2d, test_labels2d):
    print("Computing Validation Hill Climbing Accuracy")
    # test_x = testX[validation_i:]
    # test_y = testY[validation_i:]

    euclid_distance = 0

    for i, x in enumerate(test2d):
        if(i > 0):
            naive_prediction = list(test2d[i - 1])

            current_answers = list(test_labels2d[i])
    
            for j in range(len(x)):
                current_value = x[j]
                current_answer = list(current_answers[j])
                if current_value[0] == 0:
                    continue
                else:
                    current_pred = list(naive_prediction[j])
                    euclid_distance += (current_pred[0] - current_answer[0]) ** 2 + (current_pred[1] - current_answer[1]) ** 2
        '''try:
            acc = float(current_student_correct) / (current_student_incorrect + current_student_correct)
            per_student_accuracies.append(acc)

        except ZeroDivisionError:
            continue'''
    euclid_distance = np.sqrt(euclid_distance)
    return euclid_distance / (len(test2d) * len(test2d[0]))



def baseline_chopped(model, test2d, test_labels2d, window_size):
    print("Computing Validation Hill Climbing Accuracy")
    # test_x = testX[validation_i:]
    # test_y = testY[validation_i:]
    batchsize = 32

    euclid_distance = 0

    for i, x in enumerate(test2d):
        for k in range(0, len(test2d), window_size):

            '''test_x = test2d[i:i + batchsize, k:k + window_size]
            # else:
            # print("Test padded y windows", self.test_padded_y_windows[0])

            test_y = test_labels2d[i: i + batchsize, k: k + window_size]'''



            if(i > 0):
                naive_prediction = list(test2d[i - 1])

                current_answers = list(test_labels2d[i])

                for j in range(len(x)):
                    current_value = x[j]
                    current_answer = list(current_answers[j])
                    if current_value[0] == 0:
                        continue
                    else:
                        current_pred = list(naive_prediction[j])
                        euclid_distance += (current_pred[0] - current_answer[0]) ** 2 + (current_pred[1] - current_answer[1]) ** 2
        '''try:
            acc = float(current_student_correct) / (current_student_incorrect + current_student_correct)
            per_student_accuracies.append(acc)

        except ZeroDivisionError:
            continue'''
    euclid_distance = np.sqrt(euclid_distance)
    return euclid_distance / (len(test2d) * len(test2d[0]))

def train_model(model, epochs, train2d, labels2d, test2d, test_labels2d):
    #print(trainX.shape)
    #trainX_x = trainX_x[0]
    #print(trainX_x.shape)
    #trainX_y = trainX_y[0]
    #trainX_x = np.reshape(trainX_x, (4, 49999, 1))
    #trainX_y = np.reshape(trainX_y, (4, 49999, 1))
    try:
        train2d = train2d.reshape(7, 50000, 2)
    except ValueError:
        print("ACtual shape", train2d.shape)
    labels2d = labels2d.reshape(7, 50000, 2)

    test2d = test2d.reshape(4, 50000, 2)
    test_labels2d = test_labels2d.reshape(4, 50000, 2)
    model.fit(train2d, labels2d, batch_size = 4, nb_epoch=epochs)
    #model.fit([trainX_x, trainX_y], [trainY_x, trainY_y], nb_epoch=epochs, batch_size=1)

    error, vals = compute_val_acc(model, test2d, test_labels2d)
    #baseline_error = baseline(model, test2d, test_labels2d)
    print("Euclidean distance", error)
    #print("Baseline error", baseline_error)
    return error, vals

def train_chopped_model(model, max_len, epochs, train2d, labels2d, test2d, test_labels2d):
    try:
        train2d = train2d.reshape(7, max_len, 2)
    except ValueError:
        print("ACtual shape", train2d.shape)

    window_size = int(max_len / 10)
    epochs = epochs
    num_data_points = 0

    train2d = train2d.reshape(7, max_len, 2)
    train_reshaped = train2d.reshape(7*10, window_size, 2)

    labels2d = labels2d.reshape(7, max_len, 2)
    labels_reshaped = labels2d.reshape(7 * 10, window_size, 2)

    test2d = test2d.reshape(4, max_len, 2)
    test_reshaped = test2d.reshape(4 * 10, window_size, 2)

    test_labels2d = test_labels2d.reshape(4, max_len, 2)
    testlabels_reshaped = test_labels2d.reshape(4*10, window_size, 2)

    model.fit(train_reshaped, labels_reshaped, nb_epoch=epochs, batch_size = 32 )
    error, vals = compute_val_acc(model, test_reshaped, testlabels_reshaped)
    #baseline_error = baseline_chopped(model, test2d, test_labels2d, window_size)
    print("Euclidean distance", error)
    #print("Baseline error", baseline_error)
    return error, vals


    ''''# length_X = len(X)
    for i in range(epochs):
        print("epoch: ", i)
        #write shuffle code
        x = train2d[j:j + batchsize, k:k + window_size]
        small_y = labels2d[j:j + batchsize, k:k + window_size]'''



def train_stateful(model, epochs, batchsize, max_len, train2d, labels2d, test2d, test_labels2d, shuffle = True):
    #train2d = train2d.reshape(7, 50000, 2)
    #labels2d = labels2d.reshape(7, 50000, 2)

    train2d = train2d.reshape(7, 50000, 2)
    labels2d = labels2d.reshape(7, 50000, 2)
    window_size = int(max_len / 10)
    print("Window size")

    epochs = epochs
    num_data_points = 0

    #length_X = len(X)
    for i in range(epochs):
        print("epoch: ", i)
        # print()
        # sorting
        if (shuffle):
            # print(X[:2])
            temp = np.random.randint(1, 10000)  # randomly select a value for seeding
            np.random.seed(temp)  # seed
            np.random.shuffle(train2d)
            np.random.seed(temp)  # set the same seed so that y shuffles in the exact same way as X
            np.random.shuffle(labels2d)

        for j in range(0, len(train2d), batchsize):
            for k in range(0, max_len, window_size):  # train on sequences of size n/10

                # stateful training looks for same user in same position in each epoch
                # read the FAQ
                # Now write code to actually train on the batches, and not the entire training data
                # try:
                if (j + batchsize > len(train2d)):
                    continue
                else:
                    x = train2d[j:j + batchsize, k:k + window_size]
                    small_y = labels2d[j:j + batchsize, k:k + window_size]

                if (len(x) < batchsize):
                    # curr =
                    diff = batchsize - len(x)
                    temp = np.zeros((diff, window_size))
                    temp_y = np.zeros((diff, window_size, 2))
                    x = np.concatenate((x, temp))
                    small_y = np.concatenate((small_y, temp_y))
                    # x = np.concatenate((x, temp))'''
                    #   continue

                if (np.count_nonzero(x) == 0):  # if all 0's in the training data
                    continue
                # elif(np.count_nonzero(x) < batchsize * num_timesteps * 0.15): #if less than 10% of the batch is not zero
                #     #less than 50% giving worse results
                #    continue
                else:
                    num_data_points += np.count_nonzero(x)
                    training_results = model.train_on_batch(x, small_y)
                    # print(model.metrics_names)
                    # print(training_results)
                '''except:
                    print(X[j].shape)
                    print(X[j:j + batchsize, k:k+num_timesteps].shape)
                    print(k)
                    print(num_timesteps)
                    print(y[j:j + batchsize, k:k+num_timesteps].shape)'''
            # print(training_results)
            model.reset_states()
    test2d = test2d.reshape(4, 50000, 2)
    test_labels2d = test_labels2d.reshape(4, 50000, 2)
    error, vals = compute_val_acc_stateful(model, test2d, test_labels2d, 4, int(50000/10))
    print("EUclidean distance", error)
    return error, vals

    #_accuracy_stateful(model, batchsize, int(padding_size / 10), max_len)

#model = construct_chopped_nonstateful(0.01, RMSprop) #construct_model(64, 1, 1, 0.01, RMSprop, trainX_x, trainX_y)
#train_chopped_model(model, 50000, 10, train2d, labels2d, test2d, test_labels2d)

import csv

results_1 = 'eye_tracking_results_march4_1.csv'
results_2 = 'eye_tracking_results_march4_2.csv'
results_3 = 'eye_tracking_results_march4_3.csv'

per_timestep_1_1 = '1timestep_exp1.csv'
per_timestep_1_2 = '1timestep_exp2.csv'
per_timestep_1_3 = '1timestep_exp3.csv'


entries = []
for j in range(1):
    window_size = int(train2d.shape[2]/10)
    model = construct_chopped_nonstateful(100, 1, 0.01, RMSprop, window_size)
    chopped_1_100, vals_chopped_1_100 = train_chopped_model(model, train2d.shape[2], 10, train2d, labels2d, test2d, test_labels2d)
    print(chopped_1_100)
    print("1 layer 100 hidden nodes")

    model = construct_model(100, 1, 0.01, RMSprop, trainX_x, trainX_y)
    unchopped_1_100, vals_unchopped_1_100 = train_model(model, 10, train2d, labels2d, test2d, test_labels2d)

    stateful_model = construct_stateful(100, 1, 0.01, RMSprop, window_size)
    stateful_1_100, vals_stateful_1_100 = train_stateful(stateful_model, 10, 4, train2d.shape[2], train2d, labels2d, test2d, test_labels2d, shuffle = True)
    print("1 layer 100 hidden nodes")

    model = construct_chopped_nonstateful(200, 2, 0.01, RMSprop, window_size)
                                         #RMSprop)  # construct_model(64, 1, 1, 0.01, RMSprop, trainX_x, trainX_y)
    chopped_2_200, vals_chopped_2_200 = train_chopped_model(model, train2d.shape[2], 10, train2d, labels2d, test2d, test_labels2d)

    #causes memory to break???
    '''model = construct_model(200, 2, 0.01, RMSprop, trainX_x, trainX_y)
    train_model(model, 10, train2d, labels2d, test2d, test_labels2d)'''

    stateful_model = construct_stateful(200, 2, 0.01, RMSprop, window_size)
    stateful_2_200, vals_stateful_2_200 = train_stateful(stateful_model, 10, 4, train2d.shape[2], train2d, labels2d, test2d, test_labels2d, shuffle=True)

    #entries.append([chopped_1_100, unchopped_1_100, stateful_1_100, chopped_2_200, stateful_2_200])


    '''if(j == 1):
        with open(per_timestep_1_1, 'w') as csvfile:
            resultswriter = csv.writer(csvfile)
            resultswriter.writerow(['vals_chopped_1_100', 'vals_unchopped_1_100', 'vals_stateful_1_100', 'vals_chopped_2_200',
                                    'vals_stateful_2_200'])
            for i in range(len(vals_chopped_1_100)):
                resultswriter.writerow([vals_chopped_1_100[i], vals_unchopped_1_100[i], vals_stateful_1_100[i],
                                        vals_chopped_2_200[i], vals_stateful_2_200[i]])
            print('Successfully wrote to file 1')

        with open(results_1, 'w') as csvfile:
            resultswriter = csv.writer(csvfile)
            resultswriter.writerow([chopped_1_100, unchopped_1_100, stateful_1_100, chopped_2_200, stateful_2_200])
    elif(j == 2):
        with open(per_timestep_1_2, 'w') as csvfile:
            resultswriter = csv.writer(csvfile)
            resultswriter.writerow(
                ['vals_chopped_1_100', 'vals_unchopped_1_100', 'vals_stateful_1_100', 'vals_chopped_2_200',
                 'vals_stateful_2_200'])
            for i in range(len(vals_chopped_1_100)):
                resultswriter.writerow([vals_chopped_1_100[i], vals_unchopped_1_100[i], vals_stateful_1_100[i],
                                        vals_chopped_2_200[i], vals_stateful_2_200[i]])

            print('Successfully wrote to file 2')

        with open(results_2, 'w') as csvfile:
            resultswriter = csv.writer(csvfile)
            resultswriter.writerow([chopped_1_100, unchopped_1_100, stateful_1_100, chopped_2_200, stateful_2_200])'''

    if (j == 0):
        with open(per_timestep_1_3, 'w') as csvfile:
            resultswriter = csv.writer(csvfile)
            resultswriter.writerow(
                ['vals_chopped_1_100', 'vals_unchopped_1_100', 'vals_stateful_1_100', 'vals_chopped_2_200',
                 'vals_stateful_2_200'])
            for i in range(len(vals_chopped_1_100)):
                resultswriter.writerow([vals_chopped_1_100[i], vals_unchopped_1_100[i], vals_stateful_1_100[i],
                                        vals_chopped_2_200[i], vals_stateful_2_200[i]])
            print('Successfully wrote to file 3')

        with open(results_3, 'w') as csvfile:
            resultswriter = csv.writer(csvfile)
            resultswriter.writerow([chopped_1_100, unchopped_1_100, stateful_1_100, chopped_2_200, stateful_2_200])










#stateful_model = construct_stateful(0.01, RMSprop, 50000)
#train_stateful(stateful_model, 10, 4, 50000, train2d, labels2d, test2d, test_labels2d, shuffle = True)