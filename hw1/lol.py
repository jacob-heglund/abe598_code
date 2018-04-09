# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 21:33:45 2018

@author: Jacob
"""


import keras 
from tensorflow.python.client import device_lib
import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess.run())