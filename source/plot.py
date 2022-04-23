import matplotlib.pyplot as plt
import numpy as np
import time
from adjustText import adjust_text
import wandb

class experiment_plot():
  '''
  Expects results in the form of np array of [[epoch_no, train_loss, train_acc, val_loss, val_acc]]
  Provide a save location for loss/accuracy curves
  True/False to append a timestamp at the end of the figure
  '''
  def __init__(self, results, save_name="plot", save_path = 'experiments/figures/', append_time = True):
    self.epochs = results[:,0]
    self.train_loss = results[:,1]
    self.train_acc = results[:,2]
    self.val_loss = results[:,-2]
    self.val_acc = results[:,-1]
    timestr = time.strftime("%Y%m%d-%H%M%S")
    self.save_path = save_path
    if append_time is True:
      self.save_name = save_name + '_' + timestr + '.png'
    else:
      self.save_name = save_name + '.png'

  def plot(self, title_prepend = "", include_points = True):
    self.plot_loss(title_prepend, include_points)
    self.plot_accuracy(title_prepend, include_points)

  def plot_loss(self, title_prepend="", include_points = True):
    print('epochs: %s \ntrain_loss: %s \nval_loss: %s' % (self.epochs, self.train_loss, self.val_loss))
    plt.plot(self.epochs, self.train_loss, 'g', label='Training Loss')
    plt.plot(self.epochs, self.val_loss, 'b', label='Validation Loss')

    if include_points is True:
      texts = []
      for x,y,z in zip(self.epochs, self.train_loss, self.val_loss):
        labely = "{:.2f}".format(y)
        labelz = "{:.2f}".format(z)

        texts.append(plt.annotate(labely, (x,y), textcoords="offset points", xytext=(0,10),ha='center'))
        texts.append(plt.annotate(labelz, (x,z), textcoords="offset points", xytext=(0,10),ha='center'))
      adjust_text(texts)

    plt.title(title_prepend + '\nTraining and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(self.save_path + 'loss_' + self.save_name)
    plt.show()
    wandb.log({"chartLoss": plt})
    plt.close()

  def plot_accuracy(self, title_prepend="", include_points = True):

    plt.plot(self.epochs, self.train_acc, 'g', label='Training Accuracy')
    plt.plot(self.epochs, self.val_acc, 'b', label='Validation Accuracy')

    if include_points is True:
      texts = []
      for x,y,z in zip(self.epochs, self.train_acc, self.val_acc):
        labely = "{:.2f}".format(y)
        labelz = "{:.2f}".format(z)

        texts.append(plt.annotate(labely, (x,y), textcoords="offset points", xytext=(0,10),ha='center'))
        texts.append(plt.annotate(labelz, (x,z), textcoords="offset points", xytext=(0,10),ha='center'))
      adjust_text(texts)

    plt.title(title_prepend + ' Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(self.save_path + 'accuracy_' + self.save_name)
    plt.show()
    wandb.log({"chartAccuracy": plt})
    plt.close()
