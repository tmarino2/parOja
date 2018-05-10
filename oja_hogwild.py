import scipy.sparse
from multiprocessing.sharedctypes import Array
from ctypes import c_double
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt
from time import time
import scipy.io as sio
import sys
# np.random.seed(1)



d = 100
n = 100000
k=10
learning_rate = 0.4
T_freq = 100
num_threads = 1
epochs = 1
Iterations = 10

def getSyntheticData(n,d,k):
    mean = np.array([0] * d)
    alpha = 0.8
    cov_diag = [alpha**i for i in range(d)]
    covariance = np.diag(cov_diag)
    truth = np.sum(cov_diag[:k]) 
    samples = np.random.multivariate_normal(mean,covariance,n)
    return [samples,covariance,truth]


def oja_async(sample):
    # print rate_shared[0]
    sample = sample.reshape(d,1)
    U = np.frombuffer(coef_shared)
    U = U.reshape(d,k)
    grad = np.dot(sample,np.dot(sample.T,U))
    rate_shared[0] = rate_shared[0]+1
    U = U + (learning_rate/rate_shared[0])*grad
    # U = U + (learning_rate/np.sqrt(rate_shared[0]))*grad

    for i in range(d):
        for j in range(k):
            coef_shared[j+i*k] = U[i][j]

    U= np.linalg.qr(U)[0]
    if rate_shared[0]%T_freq ==0:
         error = truth-np.trace(np.dot(np.dot(U.T,covariance),U))
         return [error,time()]
    # else:
    #     return None

def hogwild(samples,k,num_threads):
    n = len(samples)
    d = len(samples[0])

    st = time()
    # print num_threads
    p = Pool(num_threads)  

    error_n_times = p.map(oja_async, samples)
    error_n_times_refined = [e_n_t for e_n_t in error_n_times if e_n_t!= None]
    # print error_n_times_refined;
    errors = [ent[0] for ent in error_n_times_refined]
    end_times = [ent[1] for ent in error_n_times_refined]
    times = [et - st for et in end_times]
    errors = [x for _,x in sorted(zip(times,errors))]
    times = sorted(times)

    n_t_freq = n/T_freq
    return [errors[:n_t_freq],times[:n_t_freq]]
  


def evaluate(model):
    data_train = data["train"]
#     data_test = data["test"]
    covariance_train = np.dot(data_train,data_train.T)/n
#     covariance_test = np.dot(data_test,data_test.T)/n
    truth_train = np.trace(covariance_train)
#     truth_test = np.trace(covariance_test)
#     error_train = np.linalg.norm(data_train - np.dot(np.dot(model,model.T),data_train),"fro")/n
#     error_test = np.linalg.norm(data_test - np.dot(np.dot(model,model.T),data_test),"fro")/n
    error_train = truth_train -  np.trace(np.dot(np.dot(model.T,covariance_train),model))
#     error_test = truth_test -  np.trace(np.dot(np.dot(model.T,covariance_test),model))
#     return error_train, error_test
    return error_train, error_train

def ojaNormal(samples,k):
    errors = []
    elapsed_times = []
    start_time = time()
    U = np.random.randn(d,k)
    # U = np.linalg.qr(U)[0]

    t = 0
    for x in samples:
        t=t+1
        x = x.reshape(d,1)
        U = U + (np.dot(x,np.dot(x.T,U)))*learning_rate/t
        if t%T_freq == 0:
            U_proj= np.linalg.qr(U)[0]
            # U = U_proj
            error = truth- np.trace(np.dot(np.dot(U_proj.T,covariance),U_proj))
            errors.append(error)
            elapsed_times.append(time() - start_time)

    U_final = np.linalg.qr(U)[0]
    return [errors,elapsed_times] 



def plotEverything(errors_oja, times_oja,errors_hogwild_one, times_hogwild_one,errors_hogwild_two, times_hogwild_two,errors_hogwild_four, times_hogwild_four):
    plt.figure(0)
    plt.xlabel('Time (secs)')
    plt.ylabel('Error')
    plt.plot(times_oja,errors_oja)
    plt.plot(times_hogwild_one,errors_hogwild_one)
    plt.plot(times_hogwild_two,errors_hogwild_two)
    plt.plot(times_hogwild_four,errors_hogwild_four)
    plt.legend(("oja","hogwild, 1 process","hogwild 2 processes","hogwild, 4 processes"))
    # plt.legend(("oja","hogwild 2 processes","hogwild, 4 processes"))
    plt.title("k = "+str(k))

    iterations_oja = range(1,len(errors_oja)+1)
    iterations_hogwild_one = range(1,len(errors_hogwild_one)+1)
    iterations_hogwild_two = range(1,len(errors_hogwild_two)+1)
    iterations_hogwild_four = range(1,len(errors_hogwild_four)+1)
    plt.figure(1)
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.plot(iterations_oja,errors_oja)
    plt.plot(iterations_hogwild_one,errors_hogwild_one)
    plt.plot(iterations_hogwild_two,errors_hogwild_two)
    plt.plot(iterations_hogwild_four,errors_hogwild_four)
    plt.legend(("oja","hogwild, 1 process","hogwild 2 processes","hogwild, 4 processes"))
    # plt.legend(("oja","hogwild 2 processes","hogwild, 4 processes"))
    plt.title("k = "+str(k))
    plt.show()


[samples,covariance,truth] = getSyntheticData(n,d,k)
total_samples = []

for i in range(epochs):
    total_samples.extend(samples)

errors_oja_sum = [0]*n
times_oja_sum = [0]*n

errors_hogwild_sum_one = [0]*n
times_hogwild_sum_one = [0]*n


errors_hogwild_sum_two = [0]*n
times_hogwild_sum_two = [0]*n

errors_hogwild_sum_four= [0]*n
times_hogwild_sum_four = [0]*n


for t in range(Iterations):
    [errors_oja, times_oja] = ojaNormal(total_samples,k)

    errors_oja_sum = [e_sum + e for (e_sum,e) in zip(errors_oja_sum,errors_oja)]
    times_oja_sum = [t_sum + t for (t_sum,t) in zip(times_oja_sum,times_oja)]

    coef_shared = Array(c_double, 
        (np.random.randn(d,k).flat),
        lock=False) 
    rate_shared = Array(c_double, 
        [0],
            lock=False) 
    [errors_hogwild_one, times_hogwild_one] = hogwild(total_samples,k,1)

    coef_shared = Array(c_double, 
        (np.random.randn(d,k).flat),
        lock=False) 
    rate_shared = Array(c_double, 
        [0],
            lock=False) 
    [errors_hogwild_two, times_hogwild_two] = hogwild(total_samples,k,2)

    coef_shared = Array(c_double, 
        (np.random.randn(d,k).flat),
        lock=False) 
    rate_shared = Array(c_double, 
        [0],
            lock=False) 
    [errors_hogwild_four, times_hogwild_four] = hogwild(total_samples,k,4)


    errors_hogwild_sum_one = [e_sum + e for (e_sum,e) in zip(errors_hogwild_sum_one,errors_hogwild_one)]
    times_hogwild_sum_one = [t_sum + t for (t_sum,t) in zip(times_hogwild_sum_one,times_hogwild_one)]


    errors_hogwild_sum_two = [e_sum + e for (e_sum,e) in zip(errors_hogwild_sum_two,errors_hogwild_two)]
    times_hogwild_sum_two = [t_sum + t for (t_sum,t) in zip(times_hogwild_sum_two,times_hogwild_two)]

    errors_hogwild_sum_four = [e_sum + e for (e_sum,e) in zip(errors_hogwild_sum_four,errors_hogwild_four)]
    times_hogwild_sum_four = [t_sum + t for (t_sum,t) in zip(times_hogwild_sum_four,times_hogwild_four)]

errors_oja_average = [e/Iterations for e in errors_oja_sum]
times_oja_average = [t/Iterations  for t in times_oja_sum]

times_hogwild_average_one = [t/Iterations  for t in times_hogwild_sum_one]
errors_hogwild_average_one = [e/Iterations  for e in errors_hogwild_sum_one]

times_hogwild_average_two = [t/Iterations  for t in times_hogwild_sum_two]
errors_hogwild_average_two = [e/Iterations  for e in errors_hogwild_sum_two]

times_hogwild_average_four = [t/Iterations  for t in times_hogwild_sum_four]
errors_hogwild_average_four = [e/Iterations  for e in errors_hogwild_sum_four]
plotEverything(errors_oja_average, times_oja_average,errors_hogwild_average_one, times_hogwild_average_one,errors_hogwild_average_two, times_hogwild_average_two,errors_hogwild_average_four, times_hogwild_average_four)

