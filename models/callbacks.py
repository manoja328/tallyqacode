#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 17:56:22 2018

@author: manoj
"""

class EarlyStopping():
    """
    Early Stopping to terminate training early under certain conditions
    """

    def __init__(self, 
                 monitor='val_loss',
                 min_delta=0,
                 patience=5):
        """
        EarlyStopping callback to exit the training loop if training or
        validation loss does not improve by a certain amount for a certain
        number of epochs
        Arguments
        ---------
        monitor : string in {'val_loss', 'loss'}
            whether to monitor train or val loss
        min_delta : float
            minimum change in monitored value to qualify as improvement.
            This number should be positive.
        patience : integer
            number of epochs to wait for improvment before terminating.
            the counter be reset after each improvment
        """
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.reset()


    def on_train_end(self):
        if self.stopped_epoch > 0:
            print('\nTerminated training for Early Stopping at epoch {}'.format(self.stopped_epoch))


    def reset(self):
        self.wait = 0
        self.best_loss = 1e15
        self.stopped_epoch = 0
        self.stop_training = False

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get(self.monitor)
        if current_loss is None:
            pass
        else:
            if (current_loss - self.best_loss) < -self.min_delta:
                self.best_loss = current_loss
                self.wait = 1
            else:
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.stop_training = True
                    self.on_train_end()
                self.wait += 1








