''' A simple program to build/plot a convex function and implement the gradient descent algorithm on it.
The function taken here is => g(W) = 10W^6 + W^4 + 10W
First order derivative => g'(W)
'''
import numpy as np
import tensorflow as tf
from sklearn import datasets
import matplotlib.pyplot as plt
print ("Hello, World!")

def convex_function():
	