import scipy.sparse
from multiprocessing.sharedctypes import Array
from ctypes import c_double
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt
from time import time




d = 100
T_max = 5000
k=10
learning_rate = .1
# T_freq = 10

mean = np.array([0] * d)
alpha = 1.2
cov_diag = [3- 0.01*i for i in range(d)]
# print cov_diag
covariance = np.diag(cov_diag)
truth = np.sum(cov_diag[:k]) 

samples = np.random.multivariate_normal(mean,covariance,T_max)
errors = []
elapsed_times = []
# start_time = time.time()
U = np.random.randn(d,k)
U = np.linalg.qr(U)[0]


# The calculation has been adjusted to allow for mini-batches


def mse_gradient_step(sample):
    global U # Only for instructive purposes!
    # global t
    sample = sample.reshape(d,1)

    U = np.frombuffer(coef_shared)
    U = U.reshape(d,k)
    # print U.shape
    grad = np.dot(sample,np.dot(sample.T,U))
    rate_shared[0] = rate_shared[0]+1
    # print rate_shared[0]

    # Update the nonzero weights one at a time
    U = U + learning_rate/rate_shared[0]*grad
    # coef_shared = U
    for i in range(d):
        for j in range(k):
            coef_shared[j*d+i] = U[i][j]

    U= np.linalg.qr(U)[0]
    error = truth- np.trace(np.dot(np.dot(U.T,covariance),U))
    return [error,time()]



coef_shared = Array(c_double, 
        (np.random.normal(size=(d,k)).flat),
        lock=False) # Hogwild
print coef_shared[d]
rate_shared = Array(c_double, 
        [0],
        lock=False) 

st = time()
p = Pool(10)  
error_n_times = p.map(mse_gradient_step, samples)
errors = [ent[0] for ent in error_n_times]
end_times = [ent[1] for ent in error_n_times]
times = [et - st for et in end_times]
errors = [x for _,x in sorted(zip(times,errors))]
times = sorted(times)
# print errors
# print times
plt.plot(times,errors)
plt.show()
