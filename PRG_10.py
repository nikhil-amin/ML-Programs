# (QUESTION: 10)
# Implement the non-parametric Locally Weighted Regression algorithm in order to
# fit data points. Select appropriate data set for your experiment and draw graphs.

import numpy as np
from bokeh.plotting import figure,show
from bokeh.layouts import gridplot

def local_regression(x0,X,Y,tau):
    x0=np.r_[1,x0]
    X=np.c_[np.ones(len(X)),X]
    xw=X.T*radial_kernel(x0,X,tau)
    beta=np.linalg.pinv(xw @ X) @ xw @ Y
    return x0 @ beta

def radial_kernel(x0,X,tau):
    return np.exp(np.sum((X-x0)**2,axis=1)/(-2*tau*tau))

n=1000
X=np.linspace(-3,3,num=n)
print("The Data Set (10 Samples) X:\n",X[1:10])
Y=np.log(np.abs(X**2-1)+.5)
print("\nThe fitting Curve Data Set (10 Samples) Y:\n",Y[1:10])
X+=np.random.normal(scale=.1,size=n)
print("\nNormalised (10 Samples) X:\n",X[1:10])

domain=np.linspace(-3,3,num=300)
print("\nX0 Domain Space (10 Samples):\n",domain[1:10])

def plot_lwr(tau):
    prediction=[local_regression(x0,X,Y,tau) for x0 in domain]
    plot=figure(plot_width=400,plot_height=400)
    plot.title.text='tau=%g'%tau
    plot.scatter(X,Y,alpha=.3)
    plot.line(domain,prediction,line_width=2,color='red')
    return plot

show(gridplot([[plot_lwr(10.0),plot_lwr(1.0)],[plot_lwr(0.1),plot_lwr(0.01)]]))
