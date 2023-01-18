import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA

task = 3
# function to perform EM algorithm for GMM
def EM_GMM(data, k_range):
    log_likelihoods = []
    for k in k_range:
        n_samples, n_features = data.shape
        means = data[np.random.choice(n_samples, k, replace=False), :]
        covariances = []
        for i in range(k):
            cov = np.eye(n_features)
            covariances.append(cov)
        weights = np.ones(k) / k
        responsibilities = np.zeros((n_samples, k))
        for i in range(100):
            for j in range(k):
                responsibilities[:, j] = weights[j] * multivariate_normal.pdf(data, mean=means[j], cov=covariances[j])
            responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)
            n_responsibilities = np.sum(responsibilities, axis=0)
            for j in range(k):
                means[j] = 1 / n_responsibilities[j] * np.sum(responsibilities[:, j] * data.T, axis=1).T
                covariances[j] = 1 / n_responsibilities[j] * np.dot((responsibilities[:, j] * (data - means[j]).T), (data - means[j]))
                weights[j] = n_responsibilities[j] / n_samples
        log_likelihood = log_likelihoodd(data, means, covariances, weights)
        log_likelihoods.append(log_likelihood)

    return log_likelihoods

def GMM_kStar(data, k_star):
    log_likelihoods = []
    k=k_star
    n_samples, n_features = data.shape
    means = data[np.random.choice(n_samples, k, replace=False), :]
    covariances = []
    for i in range(k):
        cov = np.eye(n_features)
        covariances.append(cov)
    weights = np.ones(k) / k
    responsibilities = np.zeros((n_samples, k))
    fig, ax = plt.subplots()
    scat = ax.scatter(data[:, 0], data[:, 1], color='b')
    means_scat = ax.scatter(means[:, 0], means[:, 1], color='r')

    #initialize color array of random k color
    colors = ["red", "green", "yellow", "purple", "orange", "black" ]
    color_array = random.sample(colors, k)


    def update(i):
        nonlocal means, covariances, weights, responsibilities
        for j in range(k):
            responsibilities[:, j] = weights[j] * multivariate_normal.pdf(data, mean=means[j], cov=covariances[j])
        responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)
        n_responsibilities = np.sum(responsibilities, axis=0)
        for j in range(k):
            means[j] = 1 / n_responsibilities[j] * np.sum(responsibilities[:, j] * data.T, axis=1).T
            covariances[j] = 1 / n_responsibilities[j] * np.dot((responsibilities[:, j] * (data - means[j]).T), (data - means[j]))
            weights[j] = n_responsibilities[j] / n_samples

        if n_features == 2:
                ax.cla()
                scat = ax.scatter(data[:, 0], data[:, 1], color='b')
                means_scat = ax.scatter(means[:, 0], means[:, 1], color='r')
                ax.set_title('k = ' + str(k) + ' and iteration = ' + str((i+1)))
                for j in range(k):
                    rv = multivariate_normal(mean=means[j], cov=covariances[j])
                    x, y = np.mgrid[min(data[:,0]):max(data[:,0]):.01, min(data[:,1]):max(data[:,1]):.01]
                    pos = np.dstack((x, y))
                    ax.contour(x, y, rv.pdf(pos), colors= color_array[j] )

            
    anim = animation.FuncAnimation(fig, update, frames=100, repeat=False)
    plt.show()
    return log_likelihoods

def GMM_RD(data, k_star):
    log_likelihoods = []
    k=k_star
    n_samples, n_features = data.shape
    means = data[np.random.choice(n_samples, k, replace=False), :]
    covariances = []
    for i in range(k):
        cov = np.eye(n_features)
        covariances.append(cov)
    weights = np.ones(k) / k
    responsibilities = np.zeros((n_samples, k))
    fig, ax = plt.subplots()
    scat = ax.scatter(data[:, 0], data[:, 1], color='b')
    means_scat = ax.scatter(means[:, 0], means[:, 1], color='r')

    #initialize color array of random k color
    colors = ["red", "green", "yellow", "purple", "orange", "black" ]
    color_array = random.sample(colors, k)


    def update(i):
        nonlocal means, covariances, weights, responsibilities
        for j in range(k):
            responsibilities[:, j] = weights[j] * multivariate_normal.pdf(data, mean=means[j], cov=covariances[j])
        responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)
        n_responsibilities = np.sum(responsibilities, axis=0)
        for j in range(k):
            means[j] = 1 / n_responsibilities[j] * np.sum(responsibilities[:, j] * data.T, axis=1).T
            covariances[j] = 1 / n_responsibilities[j] * np.dot((responsibilities[:, j] * (data - means[j]).T), (data - means[j]))
            weights[j] = n_responsibilities[j] / n_samples

        if n_features == 2:
                ax.cla()
                scat = ax.scatter(data[:, 0], data[:, 1], color='b')
                means_scat = ax.scatter(means[:, 0], means[:, 1], color='r')
                ax.set_title('k = ' + str(k) + ' and iteration = ' + str((i+1)))
                for j in range(k):
                    rv = multivariate_normal(mean=means[j], cov=covariances[j])
                    x, y = np.mgrid[min(data[:,0]):max(data[:,0]):.01, min(data[:,1]):max(data[:,1]):.01]
                    pos = np.dstack((x, y))
                    ax.contour(x, y, rv.pdf(pos), colors= color_array[j] )

            
    anim = animation.FuncAnimation(fig, update, frames=100, repeat=False)
    plt.show()
    return log_likelihoods


def log_likelihoodd(data, means, covariances, weights):
    k = len(means)
    n_samples, _ = data.shape
    log_likelihood = 0
    for i in range(n_samples):
        likelihoods = []
        for j in range(k):
            likelihood = weights[j] * multivariate_normal.pdf(data[i], mean=means[j], cov=covariances[j])
            likelihoods.append(likelihood)
        total_likelihood = np.sum(likelihoods)
        log_likelihood += np.log(total_likelihood)
    return log_likelihood




# load data
def load_data(filename) :
    return np.loadtxt(filename)

if __name__ == '__main__':
    #My name and id
    print("\n"+"*"*50)
    print("Name: Fahmid Al Rifat")
    print("Student ID: 1705087")
    print("*"*50+"\n")

    # load data
    data = load_data('data3D.txt')

    # range of k values to try
    k_range = range(1, 10)

    # perform EM algorithm for GMM
    log_likelihoods = EM_GMM(data, k_range)

    print("log_likelihoods : ", log_likelihoods)
    print("*"*50+"\n")

    # Calculate the difference between consecutive log-likelihoods
    differences = np.diff(log_likelihoods)

    print("differences : ", differences)
    print("*"*50+"\n")

    # Set a threshold for the minimum difference
    threshold = 10
    k_star = 0

    # Iterate through the difference values and check if the threshold is met
    for i in range(len(differences)):
        if abs(differences[i]) < threshold:
            k_star = i+1
            break

    print('k* [Elbow Method] = ', k_star)
    print("*"*50+"\n")

    if task == 1:

        plt.plot(k_range, log_likelihoods)
        plt.xlabel('k')
        plt.ylabel('log-likelihood')

        # Plot a vertical line at k*
        plt.axvline(x=k_star, color='red', linestyle='--')
        # Show the plot
        plt.show()

    elif task == 2:
        # Perform EM algorithm for GMM with k=k*
        log_likelihoods = GMM_kStar(data, k_star)

    elif task == 3:
        # Perform EM algorithm for GMM with k=k*
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(data)
        log_likelihoods = GMM_RD(data_pca, k_star)