# external libraries
from nose.tools import *
from numpy import *
import sys
sys.path.append('../')

# project methods
from ml_methods import *
from ml_plots import lr_plot, lr_boundary

"""
Logistical regression tests:
tests for the methods used in logistical regression.
Because most of these methods are mathematical functions,
only one test case if often needed.
"""

class logistical_regresssion_tests():

	def __init__(self):
		
		self.PATH_TO_DATA = '../data/'

	def setup(self):

		data  = genfromtxt( self.PATH_TO_DATA + "ex2data1.txt", delimiter = ',' )
		m, n  = shape( data )[0], shape(data)[1] - 1

		self.data  = data
		self.X 	   = c_[ ones((m, 1)), data[:, :n] ]
		self.y 	   = data[:, n:n+1]
		self.theta = zeros( (n+1, 1) )

		print "SETUP!"


	def teardown(self):
	    print "TEAR DOWN!"


	@with_setup(setup, teardown)
	def test_sigmoid(self):
		
		result   = sigmoid(.10)
		expected = 0.524979187479

		assert  allclose( result, expected )


	@with_setup(setup, teardown)
	def test_log_reg_cost(self):

		result   = log_reg_cost(self.theta, self.X, self.y)
		expected = array([0.69314718])

		assert  allclose( result, expected )


	@with_setup(setup, teardown)
	def test_log_reg_cost(self):

		result   = log_reg_cost(self.theta, self.X, self.y)
		expected = array([0.69314718])

		assert  allclose( result, expected )


	@with_setup(setup, teardown)
	def test_find_min_theta_cost(self):

		new_theta = self.theta
		new_theta, cost = fmin_theta( self.theta, self.X, self.y )

		result 	 = cost
		expected = 0.20349770159

		assert allclose( result, expected )


	@with_setup(setup, teardown)
	def test_find_min_theta(self):

		new_theta = self.theta
		new_theta, cost = fmin_theta( self.theta, self.X, self.y )
		
		result   = new_theta
		expected = array( [-25.16130062, 0.20623142, 0.20147143] )

		assert allclose( result, expected )


	@with_setup(setup, teardown)
	def test_predict(self):

		test = array([1, 45, 85])

		result 	 = predict( test, self.theta, False )
		expected = array( [0.5] )

		assert allclose( result, expected )


	@with_setup(setup, teardown)
	def test_lr_plot(self):
		
		lr_plot( self.data )
		#pyplot.show()


	@with_setup(setup, teardown)
	def test_plot_decision_boundary(self):

		new_theta = self.theta
		new_theta, cost = fmin_theta( self.theta, self.X, self.y )

		lr_boundary( self.data, self.X, new_theta )
		#pyplot.show()










