from ml_methods import *
from numpy import *

"""
Plot methods:
these procedures create plots and graphs specific to
online machine learning course.
"""

PATH_TO_DATA = 'data/'

def lin_plot(X, y):

    pyplot.plot(X, y, 'rx', markersize=5 )
    pyplot.ylabel('Profit in $10,000s')
    pyplot.xlabel('Population of City in 10,000s')


def lr_plot( data ):

	positives  = data[data[:,2] == 1]
	negatives  = data[data[:,2] == 0]
	
	pyplot.xlabel("Exam 1 score")
	pyplot.ylabel("Exam 2 score")
	pyplot.xlim([25, 115])
	pyplot.ylim([25, 115])

	pyplot.scatter( negatives[:, 0], negatives[:, 1], c='y', marker='o', s=40, linewidths=1, label="Not admitted" )
	pyplot.scatter( positives[:, 0], positives[:, 1], c='b', marker='+', s=40, linewidths=2, label="Admitted" )
	pyplot.legend()

	
def lr_boundary( data, X, theta ):

	lr_plot( data )
	plot_x = array( [min(X[:,1]), max(X[:,1])] )
	plot_y = (-1./ theta[2]) * (theta[1] * plot_x + theta[0])
	pyplot.plot( plot_x, plot_y )


def linear_plot():

    data = genfromtxt( PATH_TO_DATA + "ex1data1.txt", delimiter=',')
    X, y = data[:, 0], data[:, 1]
    m 	 = len(y)
    y 	 = y.reshape(m, 1)

    lin_plot(X, y)
    pyplot.show(block=True)


def linear_regression():

    data = genfromtxt( PATH_TO_DATA + 'ex1data1.txt', delimiter=',')
    X, y = data[:, 0], data[:, 1]
    m 	 = len(y)
    y 	 = y.reshape(m, 1)

    X 			= c_[ones((m, 1)), X]
    theta 		= zeros((2, 1))
    iterations 	= 1500
    alpha 		= 0.01

    cost 	= linear_cost(X, y, theta)  # should be 32.07
    theta 	= gradient_descent(X, y, theta, alpha, iterations)
    #print cost
    #print theta

    predict1 = array([1, 3.5]).dot(theta)
    predict2 = array([1, 7]).dot(theta)
    #print predict1
    #print predict2

    lin_plot(X[:, 1], y)
    pyplot.plot(X[:, 1], X.dot(theta), 'b-')
    pyplot.show(block=True)


def grad_desc_plot():

	data = genfromtxt( PATH_TO_DATA + "ex1data1.txt", delimiter=',')
	X, y = data[:, 0], data[:, 1]
	m 	 = len(y)
	y 	 = y.reshape(m, 1)
	X 	 = c_[ones((m, 1)), X]

	theta0_vals = linspace(-10, 10, 100)
	theta1_vals = linspace(-4, 4, 100)

	J_vals = zeros((len(theta0_vals), len(theta1_vals)), dtype=float64)
	for i, v0 in enumerate(theta0_vals):
	    for j, v1 in enumerate(theta1_vals):
	        theta 		 = array((theta0_vals[i], theta1_vals[j])).reshape(2, 1)
	        J_vals[i, j] = linear_cost(X, y, theta)

	R, P = meshgrid(theta0_vals, theta1_vals)

	fig = pyplot.figure()
	ax 	= fig.gca(projection='3d')
	ax.plot_surface(R, P, J_vals)
	pyplot.show(block=True)

	fig = pyplot.figure()
	pyplot.contourf(R, P, J_vals.T, logspace(-2, 3, 20))
	pyplot.plot(theta[0], theta[1], 'rx', markersize = 10)
	pyplot.show(block=True)


def logistical_plot():
	data = genfromtxt( PATH_TO_DATA + "ex2data1.txt", delimiter = ',' )
	m, n = shape( data )[0], shape(data)[1] - 1
	X 	 = c_[ ones((m, 1)), data[:, :n] ]
	y 	 = data[:, n:n+1]
	
	positives  = data[data[:,2] == 1]
	negatives  = data[data[:,2] == 0]
	
	pyplot.xlabel("Exam 1 score")
	pyplot.ylabel("Exam 2 score")
	pyplot.xlim([25, 115])
	pyplot.ylim([25, 115])

	pyplot.scatter( negatives[:, 0], negatives[:, 1], c='y', marker='o', s=40, linewidths=1, label="Not admitted" )
	pyplot.scatter( positives[:, 0], positives[:, 1], c='b', marker='+', s=40, linewidths=2, label="Admitted" )
	pyplot.legend()

	pyplot.show()
	
def logistical_regression():
	data  = genfromtxt( PATH_TO_DATA + "ex2data1.txt", delimiter = ',' )
	m, n  = shape( data )[0], shape(data)[1] - 1
	X 	  = c_[ ones((m, 1)), data[:, :n] ]
	y 	  = data[:, n:n+1]
	theta = zeros( (n+1, 1) ) 

	#print log_reg_cost(theta, X, y)
	theta, cost = fmin_theta( theta, X, y )	
	
	lr_plot( data )
	plot_x = array( [min(X[:,1]), max(X[:,1])] )
	plot_y = (-1./ theta[2]) * (theta[1] * plot_x + theta[0])
	pyplot.plot( plot_x, plot_y )

	pyplot.show()

	test = array([1, 45, 85])
	#print predict( test, theta, False )







