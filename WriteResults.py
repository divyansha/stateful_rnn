# Class to output accuracies and other metrics during training
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
