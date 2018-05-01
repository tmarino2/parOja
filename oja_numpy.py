import numpy as np
import matplotlib.pyplot as plt
import time
np.random.seed(1)
d = 100
T_max = 5000
k=10
eta_0 = 0.1
T_freq = 1

mean = np.array([0] * d)
alpha = 1.2
cov_diag = [3- 0.01*i for i in range(d)]
# print cov_diag
covariance = np.diag(cov_diag)
truth = np.sum(cov_diag[:k]) 

errors = []
elapsed_times = []
start_time = time.time()
U = np.random.randn(d,k)
U = np.linalg.qr(U)[0]

for t in range(1,T_max):
    x = np.random.multivariate_normal(mean,covariance).reshape(d,1)
    U = U + (np.dot(x,np.dot(x.T,U)))*eta_0/t
    if t%T_freq == 0:
        U= np.linalg.qr(U)[0]
        error = truth- np.trace(np.dot(np.dot(U.T,covariance),U))
        errors.append(error)
        elapsed_times.append(time.time() - start_time)


print error
plt.ylabel('Error')
plt.xlabel('Time (secs)')
plt.plot(elapsed_times,errors)
plt.show()   
U_final = np.linalg.qr(U)[0]