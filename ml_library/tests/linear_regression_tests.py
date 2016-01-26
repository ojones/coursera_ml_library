# external libraries
from nose.tools import *
from numpy import *
import sys
sys.path.append('../')

# project methods
from ml_methods import *

"""
Linear regression tests:
tests for the methods used in linear regression.
Because most of these methods are mathematical functions,
only one test case if often needed.
"""

class linear_regression_tests():

	def __init__(self):
		
		self.PATH_TO_DATA = '../data/'

	def setup(self):

		data = genfromtxt( self.PATH_TO_DATA + "ex1data1.txt", delimiter=',')
		X 	 = data[:, 0] 
		y 	 = data[:, 1]
		m 	 = len(y)

		self.X 			= c_[ones((m, 1)), X]
		self.y 			= y.reshape(m, 1)
		self.theta 		= zeros((2, 1))
		self.iterations = 1500
		self.alpha 		= 0.01

		print "SETUP!"


	def teardown(self):
	    print "TEAR DOWN!"


	@with_setup(setup, teardown)
	def test_hypothesis(self):
		
		result 	 = hypothesis(self.X, self.theta)
		expected = self.X.dot(self.theta)
		
		assert allclose( result, expected )


	@with_setup(setup, teardown)
	def test_linear_cost(self):
		
		result 	 = linear_cost(self.X, self.y, self.theta)
		expected = 32.0727338775
		
		assert allclose( result, expected )


	@with_setup(setup, teardown)
	def test_gradient_descent(self):
		
		result 	 = gradient_descent(self.X, self.y, self.theta, self.alpha, self.iterations)
		expected = array( [[-3.63029144], [ 1.16636235]] )
		
		assert allclose( result, expected )










