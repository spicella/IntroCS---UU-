import numpy as np
import matplotlib.pyplot as plt
import random as rd 
import timeit
import time
plt.style.use('seaborn-darkgrid')

# Parameters definition:

f2_W = 30 # f2 rule number
f2_bin = [int(x) for x in np.binary_repr(f2_W, width=8)] # binary representation of f2 rule

#All lambdas to simulate are now included here!
N1=np.arange(0.01,0.05,0.005)
N2=np.arange(0.05,0.95,0.05)
N3=np.arange(0.95,0.99,0.005)
lambda_vec=np.concatenate((N1,N2,N3),axis=None).tolist()
#round values for lambdas up to the third digit (useful not to manipulate it later for not having
#huge legends in the time-density plot)
for i in range(0,len(lambda_vec)):
    lambda_vec[i] = round(lambda_vec[i], 3)

    
    
    
#Functions declaration:


# function which makes the space (cells)- time (iterations) grid for a certain f2 & lambda
# set density_vec to True if you want to keep track of the density for each iteration
def make_space_time_grid(lbd, f2_bin, n, T, density_vec = False):
    rows = T
    columns = n

    grid = np.zeros([rows,columns+2]) # T x n grid (matrix) of zeros (white cells)
    grid[0, :]=rd.choices([0,1], k=(n+2)) # random first row with 50% chance of either 0 or 1
    d_vec = [0]*T # keep track of density per iteration

    #Make the space-time grid
    for i in np.arange(0, rows-1): # for each row of the grid (minus first row)
        for j in np.arange(0, columns): # for each column of the grid
            if rd.random() >= lbd:
                # apply null rule so map to 0 (neighbors irrelevant)
                grid[i + 1, j + 1] = 0
            else:   # apply f2 rule so look at neighbors
                #parse neighbors to int:
                k = int("".join(str(int(x)) for x in grid[i, j:j+3]),2)
                grid[i + 1, j + 1] = f2_bin[7-k]
        if density_vec: # if True, keep track of this iteration's density
            d_vec[i] = np.count_nonzero(grid[i, 1:n+1] == 1)/n
    return grid, d_vec

#Make a diploid grid/matrix, and plot it:
def plot_space_time_grid(n, T):

    grid = make_space_time_grid(lbd, f2_bin, n, T)[0]  # take 0th element = grid
    columns = n

    # plot all rows (:) and columns, ignoring the edges (first & last column)
    plt.figure(1,figsize=[8,8])
    plt.imshow(grid[:,1:columns+1], cmap="Greys", interpolation="nearest")
    plt.title("Space-time grid for rule $f_{2}$=%d, $\\lambda$=%.2f"%(f2_W, lbd),fontsize=18)
    plt.xlabel("Space [cell]",fontsize=16)
    plt.ylabel("t [iteration]",fontsize=16)
    toc = time.clock()
    return toc

#Plot time (iterations) against density (per iteration) for different values of lambda in lambda_vec
def plot_time_density(n, T, nruns): 

    # a #lambdas-nsteps-nruns matrix to save all useful informations for density over time,
    #for different lambdas, for different runs
    d = np.zeros([len(lambda_vec),T,nruns])
    plt.figure(2,figsize=[16,9])
    for i in range(len(lambda_vec)):#loop over different lambdas
        print("Evaluating %d/%d lambdas" % (i+1,len(lambda_vec)))
        for k in range(0,nruns): #loop over different runs at fixed lambda
            #print("Evaluating %d/%d runs" % (k+1,nruns))
            # take 1st element = density vector and put into i^th column
            d[i,:,k] = make_space_time_grid(lambda_vec[i], f2_bin, n, T, density_vec=True)[1]

        print("\n")
    #generate arrays avg and std for plotting and plot canvas
    d_avgs = np.zeros([T,len(lambda_vec)])
    d_stds = np.zeros([T,len(lambda_vec)])
    plt.title("Time-density for $\\lambda$-s for rule $f_{2}$=%d\nAverage from %d runs" % (f2_W,nruns),fontsize=18)
    plt.xlabel("t [iteration]",fontsize=16)
    plt.grid(True)
    plt.ylabel("$\\rho(t)$",fontsize=16)
    T_plot = np.linspace(0,T-1,T)
    for i in range(0, len(lambda_vec)):
        for j in range(0,T):
            d_avgs[j,i] = np.mean(d[i,j,:]) #correct!
            d_stds[j,i] = np.std(d[i,j,:])
        plt.errorbar(T_plot[:-1],d_avgs[:-1,i],yerr=d_stds[:-1,i],marker='.',ls='--',elinewidth=1.2,capsize=2)
    
    plt.legend(lambda_vec,ncol=3,fontsize=13)
    toc = time.clock()
    #save output to file 
    np.savetxt("f2=%d_n=%d_t=%d_nruns=%d_dens_avg.csv" % (f2_W,n,T-1,nruns), d_avgs, delimiter=",")
    np.savetxt("f2=%d_n=%d_t=%d_nruns=%d_dens_std.csv"% (f2_W,n,T-1,nruns), d_stds, delimiter=",")

    return toc, d_avgs, d_stds



# plot lambda values against final density for a given f2 rule:
def plot_lambda_density(n, T,nruns,d_avgs,d_stds):

    plt.figure(3,figsize=[16,9])
    plt.errorbar(lambda_vec, d_avgs[-2],yerr=d_stds[-2],marker='.',ls='-.',elinewidth=1.2,capsize=2) #last step (d_avgs[-2]) is always at 0, check
    
    plt.title("$\\rho(\\lambda)$ for rule $f_{2}$=%d,\nAverage from %d runs" % (f2_W,nruns),fontsize=18)
    plt.xlabel("$\\lambda$",fontsize=16)
    plt.ylabel("$\\rho$(T=%d)"%(T-1),fontsize=16)
    toc = time.clock()
    return toc

#Simulation!

#n = int(1e4) # number of columns/cells
n = 1000
#T = int(5e3) +1 # number rows/iterations
T = 1000 +1 #leave the +1 here so that the time density plot "can't see" 
# the last initialized density vector which has 0 value
nruns = 25 #nruns_per lambda

tic = time.clock()
lbd=0.9 # lambda value for one shot grid representation
toc1 = plot_space_time_grid(n, T)
print(f"Computation time making space-time grid: {round(toc1 - tic, 5)}s")

#lambda_vec = [0.2, 0.4, 0.6, 0.8]
tic = time.clock()
toc2 = plot_time_density(n, T,nruns)
print(f"Computation time making time-density plot: {round(toc2[0] - tic, 5)}s")

tic = time.clock()
toc3 = plot_lambda_density(n, T,nruns,toc2[1],toc2[2])
print(f"Computation time making lambda-density plot: {round(toc3 - tic, 5)}s")

# now show all figures
plt.show()
