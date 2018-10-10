from IPython.display import clear_output
from keras.callbacks import Callback
class PlotLearning(Callback):

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.loss = []
        self.val_loss = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.loss.append(logs.get('loss'))
        self.acc.append(logs.get('dice_coef'))
        
        self.val_loss.append(logs.get('val_loss'))        
        self.val_acc.append(logs.get('val_dice_coef'))
        
        self.i += 1
        f, ax = plt.subplots(1, 2, figsize=(12,4), sharex=True)
        ax = ax.flatten()
        clear_output(wait=True)
        
        ax[0].plot(self.x, self.loss, label="loss", lw=2)
        ax[0].plot(self.x, self.val_loss, label="val loss")
        #ax[0].set_ylim(bottom=0.)
        ax[0].legend()
        ax[0].grid(True)
        
        ax[1].plot(self.x, self.acc, label="Dice coef", lw=2)
        ax[1].plot(self.x, self.val_acc, label="val Dice coef")
        #ax[1].set_ylim(bottom=0.)
        ax[1].legend()
        ax[1].grid(True)
        
        plt.show();
        
plotLoss = PlotLearning()