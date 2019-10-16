import numpy as np
import matplotlib.pyplot as plt
import random as rd #added random

# parameters:
f1_W = 0 # f1 rule number
f1_bin = [int(x) for x in np.binary_repr(f1_W, width=8)] # binary representation of f1 rule

f2_W = 18 # f2 rule number
f2_bin = [int(x) for x in np.binary_repr(f2_W, width=8)] # binary representation of f2 rule

lbd=0.7 # lambda value

n = int(1e4) # number of columns/cells
n = 100
columns = n
T = int(5e3) # number rows/iterations
T = 100
rows = T

input_pattern = np.zeros([8,3])
for i in range(8):
    # input_pattern array of 8 possible input configurations/patterns
    input_pattern[i:] = [int(x) for x in np.binary_repr(7-i, width=3)]

# function which makes the space (cells)- time (iterations) grid for a certain f2 & lambda
# set density_vec to True if you want to keep track of the density for each iteration
def make_space_time_grid(lbd, f2_bin, density_vec = False):
    
    grid = np.zeros([rows,columns+2]) # T x n grid (matrix) of zeros (white cells)
    grid[0, :]=rd.choices([0,1], k=(n+2)) # random first row with 50% chance of either 0 or 1
    d = [0]*T # keep track of density per iteration

    # make the space-time grid
    for i in np.arange(0, rows-1): # for each row of the grid (minus first row)
        for j in np.arange(0, columns): # for each column of the grid
            for k in range(8): # for 8 possible patterns
                # compare input pattern k to grid row i, cell j+1 and its neighbors
                if np.array_equal(input_pattern[k,:], grid[i, j:j+3]):
                    # if they match, set cell j+1, row i+1:
                    if rd.random()<lbd:   # Probabilistic diploid update rule
                        grid[i+1, j+1] = f2_bin[k]
                    else:
                        grid[i+1, j+1] = f1_bin[k]
        if density_vec: # is True, keep track of this iteration's density
            d[i] = np.count_nonzero(grid[i, 1:n+1]==1)/n
    return grid, d

# given a diploid grid/matrix, plots it:
def plot_space_time_grid(grid):
    # plot all rows (:) and columns, ignoring the edges (first & last column)
    plt.imshow(grid[:,1:columns+1], cmap="Greys", interpolation="nearest")
    plt.title("Space-time grid for F2 rule {}, lambda={}".format(f2_W, lbd))
    plt.xlabel("Space (cells)")
    plt.ylabel("Time (iterations)")
    plt.show()

# plot time (iterations) against density (per iteration) for different values of lambda in lambda_vec
def plot_time_density(lambda_vec):
    # make zeros matrix 1 row for each lambda value (each row will contain T density values)
    d = np.zeros([len(lambda_vec), T]) 
    
    for i in range(len(lambda_vec)):
        # take 1st element = density vector and put into i^th column
        d[i,:] = make_space_time_grid(lambda_vec[i], f2_bin, density_vec=True)[1] 
        plt.plot(range(T), d[i,:])
    
    plt.title("Time-density for different values of lambda")
    plt.xlabel("Time (iterations)")
    plt.ylabel("Density")
    plt.legend(lambda_vec)
    plt.show()

# plot lambda values against final density for a given f2 rule:
def plot_lambda_density(f2_bin):
    N1=np.arange(0.01,0.05,0.01)
    N2=np.arange(0.05,0.95,0.5)
    N3=np.arange(0.95,0.99,0.01)
    L=np.concatenate((N1,N2,N3),axis=None).tolist()
    print(L, len(L), type(L))
    
    d = [0]*len(L)
    for i in range(len(L)):
        grid = make_space_time_grid(L[i], f2_bin)[0]
        d[i] = np.count_nonzero(grid[T-1, 1:n+1]==1)/n
        
    plt.plot(L, d)
    plt.title("Lambda-density for F2 rule {}".format(f2_W))
    plt.xlabel("Lambda")
    plt.ylabel("End density")
    plt.show()
        
#grid = make_space_time_grid(lbd, f2_bin)[0] # take 0th element = grid
#plot_space_time_grid(grid)

#lambda_vec = [0.2, 0.4, 0.6, 0.8]
#plot_time_density(lambda_vec)

plot_lambda_density(f2_bin)
