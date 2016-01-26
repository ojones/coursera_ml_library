import sys
import scipy.optimize, scipy.special
from numpy import *
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D

"""
Math methods:
these are the main methods used to learn basic principles of
machine learning and data science.
"""

def hypothesis( X, theta ):

    return X.dot(theta)


def linear_cost( X, y, theta ):

    m 	 = len(y)
    term = hypothesis(X, theta) - y

    return (term.T.dot(term) / (2 * m))[0, 0]


def gradient_descent( X, y, theta, alpha, iterations ):

    grad = copy(theta)
    m 	 = len(y)

    for counter in range(0, iterations):
        inner_sum = X.T.dot(hypothesis(X, grad) - y)
        grad 	 -= alpha / m * inner_sum

    return grad

	
def sigmoid( z ):

	return scipy.special.expit(z)


def log_reg_cost( theta, X, y ):

	m = shape( X )[0]
	hypo = sigmoid( hypothesis( X, theta ) )
	term1 = log( hypo ).T.dot( -y )
	term2 = log( 1.0 - hypo ).T.dot( 1-y )

	return ((term1 - term2) / m).flatten()


def fmin_theta( theta, X, y ):

	result = scipy.optimize.fmin( log_reg_cost, x0=theta, args=(X, y), maxiter=500, full_output=True )
	
	return result[0], result[1]


def predict( theta, X, binary=True ):

	prob = sigmoid( hypothesis( theta, X ) )
	
	if binary :
		return 1 if prob > 0.5 else 0
	else:
		return prob





