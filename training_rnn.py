#from dle_data_helper_v2 import generate_sequences_from_edx
#from data_processing import MOOC_Data, MOOC_Navigation_Only_Filtered_Problem_Check_Time
import numpy as np
import datetime
from collections import OrderedDict
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.wrappers import TimeDistributed
import csv
#import mooc_constants
from comprehensive_model import ComprehensiveModel

# This file provides the framework to train an LSTM model on edx action data from SCRATCH.
# If LSTM model already exists, use a different script.


# STEP 1: Turn student actions into sequences of resource ids
# OPTION 1.A: Directly call generate_sequences_from_edx from dle_data helper
# OPTION 1.B: Read in sequences from an existing csv
# Directly calling generate_sequences_from_edx

#stat2014drops = mooc_constants.stat_2014_drops()
#stat2014exceptions = mooc_constants.stat_2014_exceptions()


#option 1: directly call generate_sequences_from_edx
'''
fname = 'ordered_BerkeleyX_Stat_2.1x_1T2014-events.log'
actions, mappings, userids, times = generate_sequences_from_edx(fname, stat2014drops, stat2014exceptions,
                                                                earliest_time=datetime.datetime(2014, 2, 22),
                                                                latest_time=datetime.datetime(2014, 4, 2),
                                                                minlen=1, min_occurrences = 1)

#construct_action_sequences first, generate pageviewcsvs second

sample_log_file = 'ordered_BerkeleyX_Stat_2.1x_1T2014-events.log'
#sample_log_file = 'DelftX_AE1110x_1T2014.log'
sample_course_axis = 'axis_BerkeleyX_Stat_2.1x_1T2014.csv'

sample_data_1 = MOOC_Data(sample_log_file, sample_course_axis)
print("finished loading from disk...")

sample_navigation_data = MOOC_Navigation_Only_Filtered_Problem_Check_Time(sample_data_1)
print(sample_navigation_data.navigation_data.columns.values)
'''

part = 'directverticals'
actionsfname = 'actiondata/actions.csv'
mappingsfname = 'actiondata/mappings.csv'
useridsfname = 'actiondata/userids.csv'
timesfname = 'actiondata/times.csv'
#trainuseridsfname = 'train_set_userids.csv'
#testuseridsfname = 'test_set_userids.csv

print("Reading in files ...")
actions = []
with open(actionsfname, 'rt') as csvfile:
    '''actionsreader = csv.reader(csvfile)
    for row in actionsreader:
        curr = []
        for elem in row:
            try:
                curr.append(int(elem))
            except ValueError: #if elem is a blank, indicating no action
                curr.append(0)

            if(elem != ' ' or elem != ''):
                curr.append(int(elem))
            else:
                curr.append(0)'''
        #actions.append(curr) #for now, clip it to max_len
        #actions.append([int(elem) for elem in row if elem != ' ' else 0])
    actionsreader = csv.reader(csvfile)
    for row in actionsreader:
        actions.append([int(elem) for elem in row])

'''
non-stateful with full length
non-stateful with half of iqr max len
stateful always on full length
count how many data points are actually trained
add the temporal mode
results in overall accuracy and student weighted accuracy

'''


mappings = {}
with open(mappingsfname, 'rt') as csvfile:
    mappingsreader = csv.reader(csvfile)
    #mappingsreader.next() #avoids error of first line having fault key
    counter = 0
    faulty = 0
    for row in mappingsreader:
        try:
            mappings[int(row[0])] = row[1]
            counter += 1
        except ValueError:
            faulty += 1
            continue
    print("Num mappings", counter)
    print("NUm fault mappings: ", faulty)

userids = []
with open(useridsfname, 'rt') as csvfile:
    useridsreader = csv.reader(csvfile)
    for row in useridsreader:
        userids.append(row[0])
        #print(row[0])

print("len User ids", len(userids))

min_len = 1
#max_len = 346
zipped = zip(actions, userids)

zipped_copy = zip(actions, userids)
num_actions = []
for p in zipped_copy:
    num_actions.append(len(p[0]))

'''num_actions = np.sort(num_actions)
q75, q25 = np.percentile(num_actions, [75 ,25])
print("MEdian", np.median(num_actions))
print(q75, q25)
print("NUm Actions", num_actions)
iqr = q75 - q25'''

def find_max_len(zipped, q75, iqr):
    max = -np.inf
    accepted_values = q75 + 1.5*iqr
    for p in zipped:
        curr = len(p[0])
        if(curr <= accepted_values and max < curr):
            max = curr
    return max

#zipped_copy2 = zip(actions, userids)

max_len = max(num_actions) #find_max_len(zipped_copy2, q75, iqr)
print("Median len", np.median(num_actions))
#min_len = int(max_len/3)
print("max len", max_len)
to_keep = [] #only contains stuff in zipped that has length between min_len and max_len
for p in zipped:
    l = len(p[0])
    if min_len <= l <= max_len:
        #if(l > int(max_len/3)):
        to_keep.append(p)

kept_actions = [p[0] for p in to_keep]
kept_userids = [p[1] for p in to_keep]
actions = kept_actions
userids = kept_userids

vocab_size = len(mappings)




'''def sort(userids):
    userid_len_mapping = {} #len(elem): elem for elem in list(userids)}
    for elem in list(userids):
        try:
            userid_len_mapping[len(elem)].append(elem)
        except KeyError:
            userid_len_mapping[len(elem)] = [elem]
    #lengths = list(userid_len_mapping.keys())
    #sorted_dict = OrderedDict(sorted(userid_len_mapping.items(), key=lambda t: t[0]))
    result = []
    for key in userid_len_mapping.keys():
        for elem in userid_len_mapping[key]:
            result.append(elem)
    print(userids[0] in result)
    #print(result)

    return result'''

#results_stateful = 'stateful_sorted_shuffled_proper_final.csv'

'''sample_indices = np.random.choice(validation_i, validation_i)
X = overall_X[sample_indices, :]
y = overall_y[sample_indices, :]
padded_y_windows = overall_padded_y_windows[sample_indices, :]'''

'''
testXfname = 'actiondata/testX.csv'
#testYfname = 'actiondata/testY.csv'
testpaddedfname = 'actiondata/test_padded_y_windows.csv'
trainXfname = 'actiondata/trainX.csv'
#trainYfname = 'actiondata/trainY.csv'
trainpaddedfname = 'actiondata/train_padded_y_windows.csv'


testX = []
#testY = []
test_padded_y_windows = []

X, padded_y_windows = [], []

with open(testXfname, 'rt') as csvfile:
    actionsreader = csv.reader(csvfile)
    for row in actionsreader:
        testX.append(row)
    testX = np.array(testX)
with open(testYfname, 'rt') as csvfile:
    actionsreader = csv.reader(csvfile)
    for row in actionsreader:
        testY.append(row)
    testY = np.array(testY)

with open(testpaddedfname, 'rt') as csvfile:
    actionsreader = csv.reader(csvfile)
    for row in actionsreader:
        test_padded_y_windows.append(row)

with open(trainXfname, 'rt') as csvfile:
    actionsreader = csv.reader(csvfile)
    for row in actionsreader:
        X.append(row)

    X = np.array(X)

with open(trainYfname, 'rt') as csvfile:
    actionsreader = csv.reader(csvfile)
    for row in actionsreader:
        y.append(row)
    y = np.array(y)'''

'''
with open(trainpaddedfname, 'rt') as csvfile:
    actionsreader = csv.reader(csvfile)
    for row in actionsreader:
        padded_y_windows.append(row)


padding_size = len(padded_y_windows[0])

y = np.zeros((len(padded_y_windows), padding_size, vocab_size), dtype=np.bool)
for i, output in enumerate(padded_y_windows):
    for t, resource_index in enumerate(output):
        if resource_index == 0:
            continue
        else:
            try:
                y[int(i), int(t), int(resource_index) - 1] = 1
            except IndexError:
                print(output)
                print(padded_y_windows[i])
                print("t: ", t)
                print("Resource index: ", resource_index)
                # y[int(i), int(t), int(resource_index) - 1] = 1
                continue
testY = np.zeros((len(test_padded_y_windows), padding_size, vocab_size), dtype=np.bool)
for i, output in enumerate(test_padded_y_windows):
    for t, resource_index in enumerate(output):
        if resource_index == 0:
            continue
        else:
            try:
                testY[int(i), int(t), int(resource_index) - 1] = 1
            except IndexError:
                print(output)
                print(test_padded_y_windows[i])
                print("t: ", t)
                print("Resource index: ", resource_index)
                # y[int(i), int(t), int(resource_index) - 1] = 1
                continue

'''

def sort(actions):
    #dictionary containing key as length of sequence, value as sequence
    userid_len_mapping = {} #len(elem): elem for elem in list(userids)}
    for elem in list(actions):
        try:
            userid_len_mapping[len(elem)].append(elem)
        except KeyError:
            userid_len_mapping[len(elem)] = [elem]
    #lengths = list(userid_len_mapping.keys())
    #created a Dictionary sort by lengths of sequeneces in descending order
    sorted_dict = OrderedDict(sorted(userid_len_mapping.items(), key=lambda t: t[0], reverse=True))
    result = []
    for key in sorted_dict.keys():
        for elem in sorted_dict[key]:
            result.append(elem)
    #print(sorted_dict.keys())
    #print(result)

    for i in range(5):
        print("Length of result val ", len(result[i]))
        print("Length of actual val ", list(sorted_dict.keys())[i])


    return result

'''My approach to training'''
def training_set(userids, toSort):

    #create a new copy of userids
    #processed_userids = list(userids)
    #np.random.shuffle(shuffled_user_ids)

    '''processed_userids = sort(userids)

    num_ids = len(userids)
    print("num ids", num_ids)
    train_size = round(num_ids * sample_proportion)
    trainuserids = processed_userids[:train_size]
    testuserids = processed_userids[train_size:]'''

    #window_len = max_len
    x_windows = []
    y_windows = []
    z = zip(actions, userids)
    kept_count = 0
    left_out_count = 0
    for i in range(len(userids)):
        current_id = userids[i]
        corresponding_index = userids.index(current_id)
        x_windows.append(actions[corresponding_index][:-1]) #all time slices except last of action i
        y_windows.append(actions[corresponding_index][1:]) #all time slices except first of action i
        #print(actions[corresponding_index][1:])

    if(toSort == True):
        x_windows = sort(x_windows)
        y_windows = sort(y_windows)

    #num_ids = len(userids)
    #train_size = round(num_ids * sample_proportion)
    #x_windows = x_windows[:train_size]
    #y_windows = y_windows[:train_size]

    print("len trainuserids", len(x_windows))

    import math
    padding_size = math.ceil(max_len/10) * 10

    print("padding size:", padding_size)
    #print("trainuserids:", trainuserids)
    #print("x_windows", x_windows)
    X = sequence.pad_sequences(x_windows, maxlen = padding_size, padding='post', truncating='post')


    padded_y_windows = sequence.pad_sequences(y_windows, maxlen = padding_size, padding = 'post', truncating='post')
    boolean = 115 in padded_y_windows
    print("Is 115 in padded_y_windows: ", boolean)
    y = np.zeros((len(padded_y_windows), padding_size, vocab_size), dtype=np.bool)
    for i, output in enumerate(padded_y_windows):
        for t, resource_index in enumerate(output):
            if resource_index == 0:
                continue
            else:
                try:
                    y[int(i), int(t), int(resource_index) - 1] = 1
                except IndexError:
                    print(output)
                    print(padded_y_windows[i])
                    print("t: ", t)
                    print("Resource index: ", resource_index)
                    #y[int(i), int(t), int(resource_index) - 1] = 1
                    continue
    print("len of X:", len(X))
    print("len of y:", len(y))
    '''with open('actiondata/trainX.csv', 'wt') as csvfile:
        mywriter = csv.writer(csvfile)
        for row in X:
            mywriter.writerow(row)

    with open('actiondata/trainY.csv', 'wt') as csvfile:
        mywriter = csv.writer(csvfile)
        for row in y:
            mywriter.writerow(row)
    with open('actiondata/train_padded_y_windows.csv', 'wt') as csvfile:
        mywriter = csv.writer(csvfile)
        for row in padded_y_windows:
            mywriter.writerow(row)'''
    return X, y, padded_y_windows, padding_size, vocab_size

#print(X)

def test_set(userids, sample_proportion):
    #processed_userids = sort(userids)
    print("Original userids ", len(userids))
    userids_copy = list(userids)
    num_ids = len(userids)
    print("num ids", num_ids)
    test_size = round(num_ids * sample_proportion)
    x_windows = []
    y_windows = []
    z = zip(actions, userids_copy)
    kept_count = 0
    left_out_count = 0
    for i in range(test_size):
        #curr_item = np.random.choice(userids)
        corresponding_index = np.random.choice(len(userids_copy) - 1, 1, replace=False)[0]
        #print("corresponding index ", corresponding_index)
        #corresponding_index = userids_copy.index(current_id)
        x_windows.append(actions[corresponding_index][:-1])  # all time slices except last of action i
        y_windows.append(actions[corresponding_index][1:])  # all time slices except first of action i
        del userids_copy[corresponding_index]
        del actions[corresponding_index]

    import math
    padding_size = math.ceil(max_len / 10) * 10

    print("padding size:", padding_size)
    # print("trainuserids:", trainuserids)
    # print("x_windows", x_windows)
    X = sequence.pad_sequences(x_windows, maxlen=padding_size, padding='post', truncating='post')

    padded_y_windows = sequence.pad_sequences(y_windows, maxlen=padding_size, padding='post', truncating='post')

    y = np.zeros((len(padded_y_windows), padding_size, vocab_size), dtype=np.bool)
    for i, output in enumerate(padded_y_windows):
        for t, resource_index in enumerate(output):
            if resource_index == 0:
                continue
            else:
                try:
                    y[int(i), int(t), int(resource_index) - 1] = 1
                except IndexError:
                    print(output)
                    print(padded_y_windows[i])
                    print("t: ", t)
                    print("Resource index: ", resource_index)
                    # y[int(i), int(t), int(resource_index) - 1] = 1
                    continue
    print("Processed userids ", len(userids_copy))
    print("len of Test Set:", len(X))
    print("len of y:", len(y))

    #Write the data to CSV, so that it has to be run only once
    '''with open('actiondata/testX.csv', 'wt') as csvfile:
        mywriter = csv.writer(csvfile)
        for row in X:
            mywriter.writerow(row)

    with open('actiondata/testY.csv', 'wt') as csvfile:
        mywriter = csv.writer(csvfile)
        for row in y:
            mywriter.writerow(row)
    with open('actiondata/test_padded_y_windows.csv', 'wt') as csvfile:
        mywriter = csv.writer(csvfile)
        for row in padded_y_windows:
            mywriter.writerow(row)'''
    return X, y, padded_y_windows, userids_copy, actions



testX, testY, test_padded_y_windows, userids, actions = test_set(userids, 0.3)
X, y, padded_y_windows, padding_size, vocab_size = training_set(userids, False)
sortedX, sortedy, sorted_padded_y_windows, padding_size, vocab_size = training_set(userids, False)




non_stateful_X = X #X[:, :round(padding_size/2)]
non_stateful_y = y #y[:, :round(padding_size/2)]
#non_stateful_padded_y_windows = padded_y_windows[:, :round(padding_size/2)]
validation_i = int(0.9 * len(X))
#validation_non_stateful = int(0.9 * len(non_stateful_X))


comprehensive_model = ComprehensiveModel(X, y, validation_i, testX, testY, max_len, vocab_size, padding_size, padded_y_windows, test_padded_y_windows)
sorted_model = ComprehensiveModel(sortedX, sortedy, validation_i, testX, testY, max_len, vocab_size, padding_size, sorted_padded_y_windows, test_padded_y_windows)
#non_stateful_model = ComprehensiveModel(X, y, validation_i, max_len, vocab_size, padding_size, padded_y_windows)
# esize hidden size layers lrate opt

#results_stateful = 'delftfinal2.csv'
results_stateful = 'feb26delft.csv'
#results_stateful = 'temp.csv'

embedding_size = [70]
hsize = [64]# 128, 256]
layers = [1] #, 2, 3]
lrate = [0.01]
opt = [RMSprop]
batchsize = 64
#opt = [Adagrad]

for hidden_size in hsize:
    for esize in embedding_size:
        for l in layers:
            for lr in lrate:
                for o in opt:
                    '''Non-Stateful'''
                    # sorted and shuffled

                    sorted_nonStateful, modelname = sorted_model.construct_model(esize, hidden_size, l, lr, o)
                    acc, ftt = sorted_model.train_model_sorted(sorted_nonStateful, modelname, 64, 10)

                    #sorted and unshuffled
                    sorted_model = ComprehensiveModel(sortedX, sortedy, validation_i, testX, testY, max_len, vocab_size,
                                                      padding_size, sorted_padded_y_windows, test_padded_y_windows)
                    sorted_nonStateful, modelname = sorted_model.construct_model(esize, hidden_size, l, lr, o)
                    acc, ftf = sorted_model.train_model_sorted(sorted_nonStateful, modelname, 64, 10, shuffle = False)

                    #unsorted and shuffled
                    nonStateful_model, modelname = comprehensive_model.construct_model(esize, hidden_size, l, lr, o)
                    acc, fft = comprehensive_model.train_model(nonStateful_model, modelname, 64, 10)

                    #unsorted and unshuffled
                    comprehensive_model = ComprehensiveModel(X, y, validation_i, testX, testY, max_len, vocab_size,
                                                             padding_size, padded_y_windows, test_padded_y_windows)

                    nonStateful_model, modelname = comprehensive_model.construct_model(esize, hidden_size, l, lr, o)
                    acc, fff = comprehensive_model.train_model(nonStateful_model, modelname, 64, 10, shuffle = False)

                    '''Stateful'''
                    #unsorted and shuffled
                    stateful_model, modelname = comprehensive_model.construct_stateful_model(esize, hidden_size, l, lr, o, padding_size, stateful=True)
                    stateful_val_acc, tft = comprehensive_model.train_stateful(stateful_model, modelname, batchsize, 10)

                    #unsorted and unshuffled
                    comprehensive_model = ComprehensiveModel(X, y, validation_i, testX, testY, max_len, vocab_size,
                                                             padding_size, padded_y_windows, test_padded_y_windows)
                    stateful_model, modelname = comprehensive_model.construct_stateful_model(esize, hidden_size, l, lr, o, padding_size, stateful=True)
                    stateful_val_acc, tff = comprehensive_model.train_stateful(stateful_model,
                                                                    modelname, batchsize, 10, shuffle = False)

                    #sorted and shuffled
                    stateful_model, modelname = sorted_model.construct_stateful_model(esize, hidden_size, l, lr,
                                                                                             o, padding_size,
                                                                                             stateful=True)
                    stateful_val_acc, ttt = sorted_model.train_stateful_sorted(stateful_model, modelname, batchsize, 10)

                    #sorted and unshuffled
                    sorted_model = ComprehensiveModel(sortedX, sortedy, validation_i, testX, testY, max_len, vocab_size,
                                                      padding_size, sorted_padded_y_windows, test_padded_y_windows)
                    stateful_model, modelname = sorted_model.construct_stateful_model(esize, hidden_size, l, lr,
                                                                                             o, padding_size,
                                                                                             stateful=True)
                    stateful_val_acc, ttf = sorted_model.train_stateful_sorted(stateful_model, modelname, batchsize, 10,
                                                                               shuffle=False)

                    #non_stateful_val_acc, by_student_accs_nonStateful = non_stateful_model.compute_val_acc(nonStateful_model)

                    #diff = stateful_val_acc - non_stateful_val_acc
                    #print("Validation accuracy", val_acc)
                    with open(results_stateful, 'w') as csvfile:
                        resultswriter = csv.writer(csvfile)
                        #towrite = [val_acc, by_student_accs, l, hsize, lrate, esize, o, batchsize]
                        towrite = ['ftt', 'ftf', 'fft', 'fff', 'ttt', 'ttf', 'tft', 'tff']
                        resultswriter.writerow(towrite)
                        for i in range(len(tft)):
                            #try:
                            #towrite = [tft[i]]
                            towrite = [ftt[i], ftf[i], fft[i], fff[i], ttt[i], ttf[i], tft[i], tff[i]]
                            resultswriter.writerow(towrite)



                    '''with open(results_stateful, 'a') as csvfile:
                        resultswriter = csv.writer(csvfile)
                        towrite = [diff]
                        resultswriter.writerow(towrite)'''
#non-stateful
'''embedding_size = [70]#[4, 8, 16, 40, 70]
hsize = [64] #, 128, 256]
layers = [1] #[1, 2, 3]
lrate = [0.01]
opt = [RMSprop]
#opt = [Adagrad]
for hidden_size in hsize:
    for esize in embedding_size:
        for l in layers:
            for lr in lrate:
                for o in opt:
                    model, modelname = comprehensive_model.construct_model(esize, hidden_size, l, lr, o)
                    accuracy = comprehensive_model.train_model(model, modelname, 64, 10)

                    #val_acc, by_student_accs = compute_val_acc(model, 64)
                    #resultscsv = 'non_stateful_v2.csv'
                    #with open(resultscsv, 'a') as csvfile:
                        #resultswriter = csv.writer(csvfile)
                        #towrite = [accuracy, val_acc, hsize, embedding_size, layers, lrate, opt, max_len]
                        #resultswriter.writerow(towrite)
                        #print(o)'''


print("Done")
