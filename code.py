#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import random as rd #added random
import timeit
import time

tic = time.clock()

#parameters:
# f1_W = 0 # f1 rule number
# f1_bin = [int(x) for x in np.binary_repr(f1_W, width=8)] # binary representation of f1 rule
# f1 = null rule, so don't need binary vector for it

f2_W = 30 # f2 rule number
f2_bin = [int(x) for x in np.binary_repr(f2_W, width=8)] # binary representation of f2 rule

lbd=0.7 # lambda value

# function which makes the space (cells)- time (iterations) grid for a certain f2 & lambda
# set density_vec to True if you want to keep track of the density for each iteration
def make_space_time_grid(lbd, f2_bin, n, T, density_vec = False):
    rows = T
    columns = n

    grid = np.zeros([rows,columns+2]) # T x n grid (matrix) of zeros (white cells)
    grid[0, :]=rd.choices([0,1], k=(n+2)) # random first row with 50% chance of either 0 or 1
    d_vec = [0]*T # keep track of density per iteration

    # make the space-time grid
    for i in np.arange(0, rows-1): # for each row of the grid (minus first row)
        for j in np.arange(0, columns): # for each column of the grid
            if rd.random() >= lbd:
                # apply null rule so map to 0 (neighbors irrelevant)
                grid[i + 1, j + 1] = 0
            else:   # apply f2 rule so look at neighbors
                #parse neighbors to int:
                k = int("".join(str(int(x)) for x in grid[i, j:j+3]),2)
                grid[i + 1, j + 1] = f2_bin[k]
        if density_vec: # if True, keep track of this iteration's density
            d_vec[i] = np.count_nonzero(grid[i, 1:n+1] == 1)/n
    return grid, d_vec

# make a diploid grid/matrix, and plot it:
def plot_space_time_grid(n, T):

    grid = make_space_time_grid(lbd, f2_bin, n, T)[0]  # take 0th element = grid
    columns = n

    # plot all rows (:) and columns, ignoring the edges (first & last column)
    plt.figure(1)
    plt.imshow(grid[:,1:columns+1], cmap="Greys", interpolation="nearest")
    plt.title("Space-time grid for F2 rule {}, lambda={}".format(f2_W, lbd))
    plt.xlabel("Space (cells)")
    plt.ylabel("Time (iterations)")
    toc = time.clock()
    return toc

# plot time (iterations) against density (per iteration) for different values of lambda in lambda_vec
def plot_time_density(lambda_vec, n, T):
    # make zeros matrix 1 row for each lambda value (each row will contain T density values)
    d = np.zeros([len(lambda_vec), T])

    plt.figure(2)
    for i in range(len(lambda_vec)):
        # take 1st element = density vector and put into i^th column
        d[i,:] = make_space_time_grid(lambda_vec[i], f2_bin, n, T, density_vec=True)[1]
        plt.plot(range(T), d[i,:])

    plt.title("Time-density for lambda's for F2 rule {}".format(f2_W))
    plt.xlabel("Time (iterations)")
    plt.ylabel("Density")
    plt.legend(lambda_vec)
    toc = time.clock()
    return toc

# plot lambda values against final density for a given f2 rule:
def plot_lambda_density(f2_bin, n, T):
    N1=np.arange(0.01,0.05,0.01)
    N2=np.arange(0.05,0.95,0.5)
    N3=np.arange(0.95,0.99,0.01)
    L=np.concatenate((N1,N2,N3),axis=None).tolist()

    d = [0]*len(L)
    for i in range(len(L)):
        grid = make_space_time_grid(L[i], f2_bin, n, T)[0]
        d[i] = np.count_nonzero(grid[T-1, 1:n+1]==1)/n

    plt.figure(3)
    plt.plot(L, d)
    plt.title("Lambda-density for F2 rule {}".format(f2_W))
    plt.xlabel("Lambda")
    plt.ylabel("End density")
    toc = time.clock()
    return toc


#n = int(1e4) # number of columns/cells
n = 500
#T = int(5e3) # number rows/iterations
T = 500

toc = plot_space_time_grid(n, T)
print(f"Computation time making space-time grid: {round(toc - tic, 5)}s")

lambda_vec = [0.2, 0.4, 0.6, 0.8]
toc = plot_time_density(lambda_vec, n, T)
print(f"Computation time making time-density plot: {round(toc - tic, 5)}s")

toc = plot_lambda_density(f2_bin, n, T)
print(f"Computation time making lambda-density plot: {round(toc - tic, 5)}s")

# now show all figures
plt.show()