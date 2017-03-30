import numpy as np
import datetime
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
#import csv
#import mooc_constants



class ComprehensiveModel(object):
    '''Class that contains all the methods for the training methods and architectures
    presented in the paper'''

    #max_len = 0
    #vocab_size = 0

    def __init__(self, X, y, validation_i, testX, testY, max_len, vocab_size, padding_size, padded_y_windows,
                 test_padded_y_windows):
        self.X = X
        self.y = y
        self.testX = testX
        self.testY = testY
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.padding_size = padding_size
        self.padded_y_windows = padded_y_windows
        self.test_padded_y_windows = test_padded_y_windows

    # STEP 3: Construct non-stateful LSTM
    def construct_model(self, e_size, hidden_size, layers, lrate, opt):
        X = self.X
        y = self.y
        #validation_i = self.validation_i
        vocab_size = self.vocab_size
        max_len = self.max_len
        padding_size = self.padding_size
        padded_y_windows = self.padded_y_windows

        e_size = e_size
        HIDDEN_SIZE = hidden_size
        LAYERS = layers
        lrate = lrate
        print('building a model')
        model = Sequential()
        model.add(Embedding(vocab_size + 1, e_size, input_length= padding_size, mask_zero=True))

        for i in range(LAYERS):
            print("adding layer " + str(i))

            model.add(LSTM(HIDDEN_SIZE, return_sequences=True, dropout_W=0.2))
            #        if i == (LAYERS - 1):
            #            model.add(Dropout(.2))
        model.add(TimeDistributed(Dense(vocab_size)))
        model.add(Activation('softmax'))
        opt = opt(lr=lrate)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        if isinstance(opt, Adagrad):
            o = 'adagrad'
        elif isinstance(opt, RMSprop):
            o = 'rmsprop'
        modelname = "eachlayer_directvertical_modelweights_" + str(LAYERS) + '_' + str(HIDDEN_SIZE) + '_' + str(
            lrate) + '_' + str(e_size) + '_' + o + '_'
        return model, modelname

    def construct_sw_model(self, e_size, hidden_size, layers, lrate, opt):
        X = self.X
        y = self.y
        #validation_i = self.validation_i
        vocab_size = self.vocab_size
        max_len = self.max_len
        padding_size = self.padding_size
        padded_y_windows = self.padded_y_windows

        e_size = e_size
        HIDDEN_SIZE = hidden_size
        LAYERS = layers
        lrate = lrate
        print('building a model')
        model = Sequential()
        model.add(Embedding(vocab_size + 1, e_size, input_length=padding_size, mask_zero=True))

        for i in range(LAYERS):
            print("adding layer " + str(i))

            model.add(LSTM(HIDDEN_SIZE, return_sequences=True, dropout_W=0.2))
            #        if i == (LAYERS - 1):
            #            model.add(Dropout(.2))
        model.add(TimeDistributed(Dense(vocab_size)))
        model.add(Activation('softmax'))
        opt = opt(lr=lrate)
        model.compile(loss='categorical_crossentropy', optimizer=opt, sample_weight_mode='temporal', metrics=['accuracy'])

        if isinstance(opt, Adagrad):
            o = 'adagrad'
        elif isinstance(opt, RMSprop):
            o = 'rmsprop'

        modelname = "eachlayer_directvertical_modelweights_" + str(LAYERS) + '_' + str(HIDDEN_SIZE) + '_' + str(
            lrate) + '_' + str(e_size) + '_' + o + '_'
        return model, modelname

    # STEP 3: Construct stateful LSTM
    def construct_stateful_model(self, e_size, hidden_size, layers, lrate, opt, max_len, stateful=True, sw = False):
        X = self.X
        y = self.y
        vocab_size = self.vocab_size
        max_len = self.max_len
        padding_size = self.padding_size
        padded_y_windows = self.padded_y_windows

        e_size = e_size
        HIDDEN_SIZE = hidden_size
        LAYERS = layers
        lrate = lrate
        print('building a model')
        model = Sequential()


        num_timesteps = int(padding_size / 10)
        model.add(Embedding(vocab_size + 1, e_size, batch_input_shape=(64, num_timesteps), mask_zero=True))

        for i in range(LAYERS):
            print("adding layer " + str(i))
            #print(len(trainuserids))

            model.add(LSTM(HIDDEN_SIZE, return_sequences=True, dropout_W=0.2, stateful=True))
            print("Added layer")

        # if i == (LAYERS - 1):
        #            model.add(Dropout(.2))
        model.add(TimeDistributed(Dense(vocab_size)))
        model.add(Activation('softmax'))
        opt = opt(lr=lrate)
        if(sw):
            model.compile(loss='categorical_crossentropy', optimizer=opt, sample_weight_mode='temporal', metrics=['accuracy'])
        else:
            model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        if isinstance(opt, Adagrad):
            o = 'adagrad'
        elif isinstance(opt, RMSprop):
            o = 'rmsprop'
        print("SUccess")
        return model

    '''def construct_stateful_sw(self, e_size, hidden_size, layers, lrate, opt, max_len, stateful=True,
                                     window_shifting=False):
        X = self.X
        y = self.y
        vocab_size = self.vocab_size
        max_len = self.max_len
        padding_size = self.padding_size
        padded_y_windows = self.padded_y_windows

        e_size = e_size
        HIDDEN_SIZE = hidden_size
        LAYERS = layers
        lrate = lrate
        print('building a model')
        model = Sequential()

        if (stateful):
            num_timesteps = int(padding_size / 10)
            model.add(Embedding(vocab_size + 1, e_size, batch_input_shape=(64, num_timesteps), mask_zero=True))

        else:
            if window_shifting:
                window_len = max_len / 10
            model.add(Embedding(vocab_size + 1, e_size, mask_zero=True))

        for i in range(LAYERS):
            print("adding layer " + str(i))
            # print(len(trainuserids))
            model.add(LSTM(HIDDEN_SIZE, return_sequences=True, dropout_W=0.2, stateful=True))

        # if i == (LAYERS - 1):
        #            model.add(Dropout(.2))
        model.add(TimeDistributed(Dense(vocab_size)))
        model.add(Activation('softmax'))
        opt = opt(lr=lrate)
        model.compile(loss='categorical_crossentropy', optimizer=opt, sample_weight_mode='temporal', metrics=['categorical_crossentropy'])
        if isinstance(opt, Adagrad):
            o = 'adagrad'
        elif isinstance(opt, RMSprop):
            o = 'rmsprop'

        return model'''



    # STEP 4: Train model
    # May want to output accuracies and other metrics during training
    """
    class WriteResults(Callback):
        def __init__(self, monitor='val_acc',file_to_write, model_params):
            self.file_name = file_to_write
            self.monitor = monitor
            self.params = model_params

        def on_epoch_end(self, epoch, logs={}):
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn("val acc unavailable")

            with open(file_to_write, 'a') as csvfile:
                resultswriter = csv.writer(csvfile)
                towrite = [str(val_acc)] + self.params
                resultswriter.writerow(towrite)
    """
    '''def accuracy_metric(y_true, y_pred):
        if(y_true == 0): '''

    def compute_val_acc(self, model):
        test_x = self.testX
        test_y = self.test_padded_y_windows
        #validation_i = self.validation_i
        vocab_size = self.vocab_size
        max_len = self.max_len
        padding_size = self.padding_size
        #padded_y_windows = self.padded_y_windows


        print("Computing Validation Hill Climbing Accuracy")
        #test_x = testX[validation_i:]
        #test_y = testY[validation_i:]
        predictions = model.predict(test_x, batch_size=8)

        #per_student_accuracies = []

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
                        student_correct[i] +=1
                        #current_student_correct += 1
                        total_correct += 1
                    else:
                        student_incorrect[i] += 1
                        #current_student_incorrect += 1
                        total_incorrect += 1
            '''try:
                acc = float(current_student_correct) / (current_student_incorrect + current_student_correct)
                per_student_accuracies.append(acc)

            except ZeroDivisionError:
                continue'''

        totals = student_correct + student_incorrect  # total per student
        '''zeros = np.where(totals == 0)[0]
        print("For the following sequences of actions zeros were obtained in totals")
        for elem in zeros:
            print("New student")
            print(X[elem])'''
        per_student_accuracies = student_correct / totals
        print("Total: ", totals)
        print("Student Correct: ", student_correct)
        print("Studen Incorrect:", student_incorrect)
        # total_val_acc = total_correct / totals
        per_student_accuracies_copy = per_student_accuracies[np.isfinite(per_student_accuracies)]  # filters out all nans as np.inf is not possibles
        print("Per student accuracy: ", per_student_accuracies)
        # print("Total validation accuracy:", total_val_acc)
        print("Average of student accuracies:", np.mean(per_student_accuracies_copy))
        '''total_val_acc = float(total_correct) / (total_correct + total_incorrect)
            #print("Total validation accuracy:", total_val_acc)
            print("Student Correct ", current_student_correct)
            print("Student Incorrect ", current_student_incorrect)
            print("Average of student accuracies:", np.mean(per_student_accuracies))
            return total_val_acc, per_student_accuracies'''
        return float(total_correct) / (total_correct + total_incorrect), per_student_accuracies_copy

    '''def sample_weights_validation(self):
        #validation_i = self.validation_i

        sample_weights = np.zeros(len(self.X) - validation_i)
        validation_set = self.X[validation_i:]
        for i, x in enumerate(validation_set):
            sample_weights[i] = 1 / np.count_nonzero(x)
        return sample_weights'''

    def compute_val_acc_sw(self, model):
        test_x = self.testX
        test_y = self.test_padded_y_windows
        vocab_size = self.vocab_size
        max_len = self.max_len
        padding_size = self.padding_size
        padded_y_windows = self.padded_y_windows

        print("Computing Validation Hill Climbing Accuracy")
        #test_x = X[validation_i:]
        #test_y = padded_y_windows[validation_i:]
        predictions = model.predict(test_x, batch_size=8)

        per_student_accuracies = []

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
                        current_student_correct += 1
                        total_correct += 1
                    else:
                        current_student_incorrect += 1
                        total_incorrect += 1
            try:
                acc = float(current_student_correct) / (current_student_incorrect + current_student_correct)
            except ZeroDivisionError:
                continue

            per_student_accuracies.append(acc)

        #sample_weights_val = self.sample_weights_validation()
        #weighted_accs = per_student_accuracies * sample_weights_val
        total_val_acc = float(total_correct) / (total_correct + total_incorrect)
        # print("Total validation accuracy:", total_val_acc)
        print("Average of student accuracies:", np.mean(per_student_accuracies))

        return total_val_acc, per_student_accuracies

    def sample_weights(self):
        lengths = np.apply_along_axis(np.count_nonzero, 1, self.X)
        temp = lengths[np.where(lengths > 0)]
        print("temp", temp)
        median_length = np.median(temp)
        print("Median length", median_length)
        print("Max length", np.max(temp))
        print("70 percentile", np.percentile(temp, 70))
        print("90 percentile", np.percentile(temp, 90))
        print("60 percentile", np.percentile(temp, 60))
        print("95 percentile", np.percentile(temp, 95))
        print("100 percentile", np.percentile(temp, 100))
        print("80 percentile", np.percentile(temp, 80))


        sample_weights = np.zeros((len(self.X), self.padding_size))
        for i, x in enumerate(self.X):
            try:
                weight = median_length/np.count_nonzero(x)
                for j in range(len(x)):
                    sample_weights[i][j] = weight
            except ZeroDivisionError:
                continue
        print("Sample weights", sample_weights)
        return sample_weights


    #train sorted non-stateful LSTM
    def train_model_sorted(self, model, modelname, batchsize, epochs, shuffle = True, sw = False):

        X = np.copy(self.X)
        y = np.copy(self.y)
        # padding_size = self.padding_size
        vocab_size = self.vocab_size
        max_len = self.max_len
        padding_size = self.padding_size
        padded_y_windows = self.padded_y_windows
        num_data_points = 0

        modelfolder = '94kerasmodels/'
        # BATCHSIZE = batchsize
        num_timesteps = int(padding_size / 10)
        epochs = epochs
        #    modelname = "directvertical_modelweights_"+str(LAYERS)+'_'+str(HIDDEN_SIZE)+'_'+str(lrate)+'_'+str(e_size)+'_'
        print("base modelname:", modelname)


        if(sw):
            lengths = np.apply_along_axis(np.count_nonzero, 1, X)
            median_length = np.median(lengths)

        batches_x = []
        batches_y = []
        # lengths
        for j in range(0, len(X), batchsize):
            if (j + batchsize > len(X)):
                # print("At this val j greater", j)
                x = X[j:len(X), :]
                small_y = y[j:len(X), :]
                diff = batchsize - len(x)
                # print("Length of x: ", len(x))
                # print("validation_i: ", validation_i)
                # print("j: ", j)
                # shape = x.shape
                temp = np.zeros((diff, padding_size))
                temp_y = np.zeros((diff, padding_size, vocab_size))
                x = np.concatenate((x, temp))
                small_y = np.concatenate((small_y, temp_y))
            else:
                x = X[j:j + batchsize, :]
                small_y = y[j:j + batchsize, :]
            batches_x.append(x)
            batches_y.append(small_y)

        batches_x = np.array(batches_x)
        batches_y = np.array(batches_y)

        stop_training = False
        for i in range(epochs):
            print("epoch: ", i)
            # print()
            # sorting
            if(shuffle):
                # print(X[:2])
                temp = np.random.randint(1, 10000)  # randomly select a value for seeding
                np.random.seed(temp)  # seed
                np.random.shuffle(batches_x)
                np.random.seed(temp)  # set the same seed so that y shuffles in the exact same way as X
                np.random.shuffle(batches_y)
                # print(X[:2])

            # model.fit(X[:validation_i], y[:validation_i], batch_size=64, nb_epoch=10,
            #          validation_data=(X[validation_i:], y[validation_i:]))

            for i in zip(batches_x, batches_y):
                #num_data_points += np.count_nonzero(x)

                x_batch, y_batch = i[0], i[1]

                if(sw):
                    batch_lengths = np.apply_along_axis(np.count_nonzero, 1, x_batch)

                    sampleWeight = self.sample_weights_batch(batchsize, padding_size, batch_lengths, median_length)


                    training_results = model.train_on_batch(x_batch, y_batch, sample_weight=sampleWeight)
                else:
                    training_results = model.train_on_batch(x_batch, y_batch)

                # print(training_results

        acc, per_student_acc = self.compute_val_acc(model)
        # print("Accuracy: ", acc)
        return acc, per_student_acc

        # train sorted non-stateful LSTM with custom sample weights

    '''def sample_weights(self):
        lengths = np.apply_along_axis(np.count_nonzero, 1, self.X)
        min_length = np.argmin(lengths)

        def helper(x):
            try:
                return min_length / np.count_nonzero(x)
            except ZeroDivisionError:
                return 0

        print("SW shape", np.apply_along_axis(helper, 1, self.X).shape)

        return np.reshape(np.apply_along_axis(helper, 1, self.X).shape[0], 1)'''
    def train_model(self, model, modelname, batchsize, epochs, shuffle = True, sw = False):
        X = self.X
        y = self.y
        vocab_size = self.vocab_size
        max_len = self.max_len
        padding_size = self.padding_size
        padded_y_windows = self.padded_y_windows
        modelfolder = '94kerasmodels/'
        BATCHSIZE = batchsize
        epochs = epochs
        #    modelname = "directvertical_modelweights_"+str(LAYERS)+'_'+str(HIDDEN_SIZE)+'_'+str(lrate)+'_'+str(e_size)+'_'
        print("base modelname:", modelname)

        # sampleWeight = sample_weights()
        # print(sampleWeight[:10])

        if(sw):
            sampleWeight = self.sample_weights()
            # print(sampleWeight[:10])
            model.fit(X, y, batch_size=64, nb_epoch=epochs, sample_weight=sampleWeight)
            acc, per_student_acc = self.compute_val_acc_sw(model)

        else:
            model.fit(X, y, batch_size=64, nb_epoch=epochs, shuffle=shuffle)  # , sample_weight = sampleWeight)
        # except IndexError:
        #    temp = np.array(X[:validation_i])
        #    print("Shape of X", temp.shape)

        # history = hist.history

        acc, per_student_acc = self.compute_val_acc(model)
        # print("Accuracy: ", acc)
        return acc, per_student_acc





    '''def train_with_sw(self, model, modelname, batchsize, epochs):
        X = self.X
        y = self.y
        validation_i = self.validation_i
        vocab_size = self.vocab_size
        max_len = self.max_len
        padding_size = self.padding_size
        padded_y_windows = self.padded_y_windows


        sampleWeight = self.sample_weights()
        # print(sampleWeight[:10])
        model.fit(X[:validation_i], y[:validation_i], batch_size=64, nb_epoch=10,
                  validation_data=(X[validation_i:], y[validation_i:]), sample_weight=sampleWeight)
        acc, per_student_acc = self.compute_val_acc_sw(model)
        return acc, per_student_acc'''

    '''def compute_val_acc_stateful(self, model, batchsize, num_timesteps, max_len, validation=True):
        X = self.testX
        y = self.testY

        #test_x = self.testX
        #test_y = self.testY
        validation_i = self.validation_i
        vocab_size = self.vocab_size
        max_len = self.max_len
        padding_size = self.padding_size
        padded_y_windows = self.padded_y_windows



        print("Computing Validation Hill Climbing Accuracy")
        total_correct = 0
        total_incorrect = 0
        if (validation_i):
            per_student_accuracies = np.zeros(len(X) - validation_i)
            student_correct = np.zeros(len(X) - validation_i)
            student_incorrect = np.zeros(len(X) - validation_i)
            for j in range(validation_i, len(X), batchsize):  # go over all batches
                # current_student_correct = 0
                # current_student_incorrect = 0
                for k in range(0, max_len, num_timesteps):  # go over all the sequences

                    # if(k + num_timesteps <= max_len):
                    test_x = X[j:j + batchsize, k:k + num_timesteps]
                    # else:
                    test_y = padded_y_windows[j: j + batchsize, k: k + num_timesteps]

                    if (j + batchsize > validation_i):
                        # print("At this val j greater", j)
                        x = X[j:validation_i, k:k + num_timesteps]
                        small_y = y[j:validation_i, k:k + num_timesteps]
                    if (len(test_x) < 128):
                        # curr =
                        diff = 128 - len(test_x)
                        temp = np.zeros((diff, num_timesteps))
                        test_x = np.concatenate((test_x, temp))
                        test_y = np.concatenate((test_y, temp))

                    predictions = model.predict_on_batch(test_x)  # predicts for batch size of 64
                    for i, x in enumerate(test_x):
                        current_predictions = list(predictions[i])
                        # print("Current predictions", current_predictions[:10])
                        current_answers = list(test_y[i])

                        for l in range(len(x)):
                            current_value = x[l]
                            if current_value == 0:
                                continue
                            else:
                                current_softmax = list(current_predictions[l])
                                # print("Current softmax", current_softmax[:10])
                                # print("index: ", current_softmax.index(max(current_softmax)))
                                best_prediction = current_softmax.index(max(current_softmax))
                                correct_answer = current_answers[l]
                                if best_prediction == correct_answer:
                                    student_correct[i] += 1
                                    total_correct += 1
                                else:
                                    student_incorrect[i] += 1
                                    total_incorrect += 1
                    model.reset_states()

        # acc = round(current_student_correct) / (current_student_incorrect + current_student_correct)
        # per_student_accuracies.append(acc)

        totals = student_correct + student_incorrect
        per_student_accuracies = student_correct / totals
        print("Total: ", totals)
        print("Student Correct: ", student_correct)
        print("Studen Incorrect:", student_incorrect)

        total_val_acc = total_correct / totals
        per_student_accuracies_copy = per_student_accuracies[
            np.isfinite(per_student_accuracies)]  # filters out all nans
        print("Per student accuracy: ", per_student_accuracies_copy)
        # print("Total validation accuracy:", total_val_acc)
        print("Average of student accuracies:", np.mean(per_student_accuracies_copy))

        return float(total_correct) / (total_correct + total_incorrect), per_student_accuracies'''

    def final_accuracy_stateful(self, model, batchsize, num_timesteps, max_len):
        X = self.testX
        y = self.test_padded_y_windows
        vocab_size = self.vocab_size
        max_len = self.max_len
        padding_size = self.padding_size
        #padded_y_windows = self.padded_y_windows

        total_correct = 0
        total_incorrect = 0
        # per_student_accuracies = np.zeros(validation_i)
        student_correct = np.zeros(len(X))
        student_incorrect = np.zeros(len(X))
        per_student_accuracies = []
        for j in range(0, len(X), batchsize):
            for k in range(0, padding_size, num_timesteps):  # go over all the sequences
                test_x = X[j:j + batchsize, k:k + num_timesteps]
                test_y = y[j: j + batchsize, k: k + num_timesteps]

                if (len(test_x) < batchsize):
                    # curr =
                    diff = batchsize - len(test_x)
                    temp = np.zeros((diff, num_timesteps))
                    test_x = np.concatenate((test_x, temp))
                    test_y = np.concatenate((test_y, temp))
                    #continue

                predictions = model.predict_on_batch(test_x)  # predicts for batch size of 64
                for i, x in enumerate(test_x):  # go over each row/student
                    current_predictions = list(predictions[i])
                    current_answers = list(test_y[i])

                    for l in range(len(x)):
                        current_value = x[l]
                        if current_value == 0:
                            continue
                        else:
                            current_softmax = list(current_predictions[l])
                            best_prediction = current_softmax.index(max(current_softmax)) + 1
                            correct_answer = current_answers[l]
                            if best_prediction == correct_answer:
                                student_correct[j + i] += 1 # check this
                                #student_correct[i] += 1
                                total_correct += 1
                            else:
                                student_incorrect[j + i] += 1
                                #student_incorrect[i] += 1
                                total_incorrect += 1


            model.reset_states()
        totals = student_correct + student_incorrect  # total per student
        '''zeros = np.where(totals == 0)[0]
        print("For the following sequences of actions zeros were obtained in totals")
        for elem in zeros:
            print("New student")
            print(X[elem])'''
        per_student_accuracies = student_correct / totals
        print("Total: ", totals)
        print("Student Correct: ", student_correct)
        print("Studen Incorrect:", student_incorrect)
        # total_val_acc = total_correct / totals
        per_student_accuracies_copy = per_student_accuracies[np.isfinite(per_student_accuracies)]  # filters out all nans as np.inf is not possibles
        print("Per student accuracy: ", per_student_accuracies)
        # print("Total validation accuracy:", total_val_acc)
        print("Average of student accuracies:", np.mean(per_student_accuracies_copy))
        return float(total_correct) / (total_correct + total_incorrect), per_student_accuracies_copy

    '''def sort(self, userids):
        userid_len_mapping = {}  # len(elem): elem for elem in list(userids)}
        for elem in list(userids):
            try:
                userid_len_mapping[len(elem)].append(elem)
            except KeyError:
                userid_len_mapping[len(elem)] = [elem]
        # lengths = list(userid_len_mapping.keys())
        # sorted_dict = OrderedDict(sorted(userid_len_mapping.items(), key=lambda t: t[0]))
        result = []
        for key in userid_len_mapping.keys():
            for elem in userid_len_mapping[key]:
                result.append(elem)
        return np.array(result)'''

    def train_stateful_sorted(self, model, modelname, batchsize, epochs, shuffle = True, sw = False):
        '''Assume self.X is sorted'''
        X = np.copy(self.X)
        y = np.copy(self.y)
        #padding_size = self.padding_size
        vocab_size = self.vocab_size
        max_len = self.max_len
        padding_size = self.padding_size
        padded_y_windows = self.padded_y_windows
        num_data_points = 0

        modelfolder = '94kerasmodels/'
        # BATCHSIZE = batchsize
        num_timesteps = int(padding_size / 10)
        epochs = epochs
        #    modelname = "directvertical_modelweights_"+str(LAYERS)+'_'+str(HIDDEN_SIZE)+'_'+str(lrate)+'_'+str(e_size)+'_'
        print("base modelname:", modelname)
        batches_x = []
        batches_y = []

        if (sw):
            lengths = np.apply_along_axis(np.count_nonzero, 1, X)
            median_length = np.median(lengths)
        #lengths
        for j in range(0, len(X), batchsize):
            if (j + batchsize > len(X)):
                # print("At this val j greater", j)
                x = X[j:len(X), :]
                small_y = y[j:len(X), :]
                diff = batchsize - len(x)
                # print("Length of x: ", len(x))
                # print("validation_i: ", validation_i)
                # print("j: ", j)
                # shape = x.shape
                temp = np.zeros((diff, padding_size))
                temp_y = np.zeros((diff, padding_size, vocab_size))
                x = np.concatenate((x, temp))
                small_y = np.concatenate((small_y, temp_y))
            else:
                x = X[j:j + batchsize, :]
                small_y = y[j:j + batchsize, :]
            batches_x.append(x)
            batches_y.append(small_y)

        batches_x = np.array(batches_x)
        batches_y = np.array(batches_y)

        stop_training = False
        for i in range(epochs):
            print("epoch: ", i)
            # print()
            # sorting
            if(shuffle):
                # print(X[:2])
                temp = np.random.randint(1, 10000)  # randomly select a value for seeding
                np.random.seed(temp)  # seed
                np.random.shuffle(batches_x)
                np.random.seed(temp)  # set the same seed so that y shuffles in the exact same way as X
                np.random.shuffle(batches_y)
                # print(X[:2])

            # model.fit(X[:validation_i], y[:validation_i], batch_size=64, nb_epoch=10,
            #          validation_data=(X[validation_i:], y[validation_i:]))

            for i in zip(batches_x, batches_y):
                for k in range(0, padding_size, num_timesteps):  # train on sequences of size n/10

                    # stateful training looks for same user in same position in each epoch
                    # read the FAQ
                    # Now write code to actually train on the batches, and not the entire training data
                    # try:

                    x_batch, y_batch = i[0], i[1]
                    x = x_batch[:, k:k + num_timesteps]
                    small_y = y_batch[:, k:k + num_timesteps]

                    if (np.sum(x) == 0):  # if all 0's in the training data
                        continue
                    # elif(np.count_nonzero(x) < batchsize * num_timesteps * 0.15): #if less than 10% of the batch is not zero
                    #     #less than 50% giving worse results
                    #    continue
                    else:
                        # print(x)
                        # print(np.sum(x))
                        # print("Shape of X", x.shape)
                        # print("Shape of Y", small_y.shape)
                        num_data_points += np.count_nonzero(x)

                        if (sw):
                            batch_lengths = np.apply_along_axis(np.count_nonzero, 1, x_batch)

                            sampleWeight = self.sample_weights_batch(batchsize, num_timesteps, batch_lengths, median_length)

                            training_results = model.train_on_batch(x, small_y, sample_weight=sampleWeight)
                        else:
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
        return self.final_accuracy_stateful(model, batchsize, int(padding_size / 10), max_len)


    def train_stateful(self, model, batchsize, epochs, shuffle = True):
        X = np.copy(self.X) #self.sort(self.X)
        #print(X[:2])
        y = np.copy(self.y)
        vocab_size = self.vocab_size
        max_len = self.max_len
        padding_size = self.padding_size
        padded_y_windows = self.padded_y_windows
        num_data_points = 0

        num_timesteps = int(padding_size / 10)
        epochs = epochs


        length_X = len(X)
        for i in range(epochs):
            print("epoch: ", i)
            #print()
            #sorting
            if(shuffle):
                #print(X[:2])
                temp = np.random.randint(1, 10000) #randomly select a value for seeding
                np.random.seed(temp) #seed
                np.random.shuffle(X)
                np.random.seed(temp) #set the same seed so that y shuffles in the exact same way as X
                np.random.shuffle(y)


            for j in range(0, length_X, batchsize):
                for k in range(0, padding_size, num_timesteps):  # train on sequences of size n/10

                    # stateful training looks for same user in same position in each epoch
                    # read the FAQ
                    # Now write code to actually train on the batches, and not the entire training data
                    # try:
                    if (j + batchsize > length_X):
                        continue
                    else:
                        x = X[j:j + batchsize, k:k + num_timesteps]
                        small_y = y[j:j + batchsize, k:k + num_timesteps]


                    if (len(x) < batchsize):
                        # curr =
                        diff = batchsize - len(x)
                        temp = np.zeros((diff, num_timesteps))
                        temp_y = np.zeros((diff, num_timesteps, vocab_size))
                        x = np.concatenate((x, temp))
                        small_y = np.concatenate((small_y, temp_y))
                        #x = np.concatenate((x, temp))'''
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
                #print(training_results)
                model.reset_states()
        return self.final_accuracy_stateful(model, batchsize, int(padding_size/10), max_len)


    #calculates sample weight for elements in a given batch. parameter 'input' refers to batch
    def sample_weights_batch(self, batchsize, num_timesteps, lengths, median_length):

        sample_weights = np.zeros((batchsize, num_timesteps))
        for i in range(batchsize):
            try:
                weight = median_length/lengths[i]
                for j in range(num_timesteps):
                    sample_weights[i][j] = weight
            except ZeroDivisionError:
                sample_weights[i] = 0
        return sample_weights

    def train_stateful_sw(self, model, batchsize, epochs, shuffle = True):
        X = self.X
        y = self.y

        lengths = np.apply_along_axis(np.count_nonzero, 1, self.X)
        temp = lengths[np.where(lengths > 0)]
        print("temp", temp)
        median_length = np.median(temp)
        print("Median length", median_length)

        #validation_i = self.validation_i
        vocab_size = self.vocab_size
        max_len = self.max_len
        padding_size = self.padding_size
        padded_y_windows = self.padded_y_windows
        num_data_points = 0

        modelfolder = '94kerasmodels/'
        # BATCHSIZE = batchsize

        #this is the window size
        num_timesteps = int(padding_size / 10)
        epochs = epochs
        #    modelname = "directvertical_modelweights_"+str(LAYERS)+'_'+str(HIDDEN_SIZE)+'_'+str(lrate)+'_'+str(e_size)+'_'
        resultscsv = 'stateful_results_2.csv'
        bestresult = '94_best_results.csv'
        #    callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose =0), WriteResults(file_to_write = resultscsv, model_params = [LAYERS, HIDDEN_SIZE, lrate, e_size]
        previous_val_loss = []
        stop_training = False
        for i in range(epochs):
            print("epoch: ", i)
            # print()

            if (shuffle):
                # print(X[:2])
                temp = np.random.randint(1, 10000)  # randomly select a value for seeding
                np.random.seed(temp)  # seed
                np.random.shuffle(X)
                np.random.seed(temp)  # set the same seed so that y shuffles in the exact same way as X
                np.random.shuffle(y)
            # model.fit(X[:validation_i], y[:validation_i], batch_size=64, nb_epoch=10,
            #          validation_data=(X[validation_i:], y[validation_i:]))

            for j in range(0, len(X), batchsize):

                batch_lengths = np.apply_along_axis(np.count_nonzero, 1, X[j:j + batchsize, :])
                sampleWeight = self.sample_weights_batch(batchsize, num_timesteps, batch_lengths, median_length)
                for k in range(0, padding_size, num_timesteps):  # train on sequences of size n/10

                    # stateful training looks for same user in same position in each epoch
                    # read the FAQ
                    # Now write code to actually train on the batches, and not the entire training data
                    # try:
                    if (j + batchsize > len(X)):
                        # print("At this val j greater", j)
                        # x = X[j:validation_i, k:k+num_timesteps]
                        # small_y = y[j:validation_i, k:k+num_timesteps]
                        continue
                    else:
                        x = X[j:j + batchsize, k:k + num_timesteps]
                        small_y = y[j:j + batchsize, k:k + num_timesteps]

                    if (len(x) < batchsize):
                        # curr =
                        diff = batchsize - len(x)
                        # print("Length of x: ", len(x))
                        # print("validation_i: ", validation_i)
                        # print("j: ", j)
                        # shape = x.shape
                        temp = np.zeros((diff, num_timesteps))
                        temp_y = np.zeros((diff, num_timesteps, vocab_size))
                        x = np.concatenate((x, temp))
                        small_y = np.concatenate((small_y, temp_y))
                        # x = np.concatenate((x, temp))'''
                        #   continue

                    if (np.sum(x) == 0):  # if all 0's in the training data
                        continue
                    # elif(np.count_nonzero(x) < batchsize * num_timesteps * 0.15): #if less thaself.sample_weights_batchn 10% of the batch is not zero
                    #     #less than 50% giving worse results
                    #    continue
                    else:
                        # print(x)
                        # print(np.sum(x))
                        # print("Shape of X", x.shape)
                        # print("Shapre of Y", small_y.shape)
                        num_data_points += np.count_nonzero(x)
                        training_results = model.train_on_batch(x, small_y, sample_weight=sampleWeight)
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
        return self.final_accuracy_stateful(model, batchsize, int(padding_size / 10), max_len)
