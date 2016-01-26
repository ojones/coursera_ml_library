# external libraries
from numpy import *

# project methods
from ml_plots import *

"""
The purpose of this project is to create and learn reusable
methods for machine learning.
The main function executes the currently available plots.
"""

def main():

	print "The main function executes the currently available plots."
	print "Just close plot windows as they appear."

	set_printoptions(precision=6, linewidth=200)

	linear_plot()
	linear_regression()
	grad_desc_plot()

	logistical_plot()
	logistical_regression()

	print "Thanks for viewing.  More to come."
	

if __name__ == '__main__':
	main()

