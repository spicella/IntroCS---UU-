#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import matplotlib.pyplot as plt
import random as rd #added random     


# In[164]:


lbd=0.7


# In[ ]:





# In[165]:


f2_W = 18 # f2 rule number


# In[166]:


f1_W = 0 # f1 rule number


# In[167]:


f2_bin = [int(x) for x in np.binary_repr(f2_W, width=8)]


# In[168]:


f2_bin # binary representation of f2 rule


# In[169]:


f1_bin = [int(x) for x in np.binary_repr(f1_W, width=8)]


# In[170]:


f1_bin # binary representation of f1 rule


# In[171]:


input_pattern = np.zeros([8,3])
for i in range(8):
    input_pattern[i:] = [int(x) for x in np.binary_repr(7-i, width=3)]


# In[172]:


input_pattern # array of 8 possible input configurations/patterns


# In[173]:


n = int(1e4) # number of columns/cells
n = 500
columns = n
T = int(5e3) # number rows/iterations
T = 300
rows = T
grid = np.zeros([rows,columns+2]) # T x n grid (matrix) of zeros (white cells)


# In[174]:


grid[0, :]=rd.choices([0,1], k=(n+2)) # random first row w/ 50% chance of either 0 or 1

#print(grid[0,:]) #Check if it works


# In[175]:


for i in np.arange(0, rows-1): # for each row of the grid (minus first row)
    for j in np.arange(0, columns): # for each column of the grid
        for k in range(8): # for 8 possible patterns
            # compare input pattern k to grid row i, cell j+1 and its neighbors
            if np.array_equal(input_pattern[k,:], grid[i, j:j+3]):
                # if they match, set cell j+1, row i+1:
                if rd.random()<lbd:   #Probabilistic update rule
                    grid[i+1, j+1] = f2_bin[k] 
                else:
                    grid[i+1, j+1] = f1_bin[k] 


# In[177]:


# plot all rows (:) and columns, ignoring the edges (first & last column)
plt.imshow(grid[:,1:columns+1], cmap="Greys", interpolation="nearest")
plt.title("ECA Rule {}".format(f2_W))
plt.show()
