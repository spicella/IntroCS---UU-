#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import matplotlib.pyplot as plt


# In[ ]:





# In[ ]:





# In[28]:


f2_W = 190 # f2 rule number


# In[29]:


f1_W = 0 # f1 rule number


# In[30]:


f2_bin = [int(x) for x in np.binary_repr(f2_W, width=8)]


# In[31]:


f2_bin # binary representation of f2 rule


# In[32]:


f1_bin = [int(x) for x in np.binary_repr(f1_W, width=8)]


# In[33]:


f1_bin


# In[34]:


input_pattern = np.zeros([8,3])
for i in range(8):
    input_pattern[i:] = [int(x) for x in np.binary_repr(7-i, width=3)]


# In[35]:


input_pattern # array of 8 possible input configurations/patterns


# In[36]:


n = int(1e4) # number of columns/cells
n = 100
columns = n
T = int(5e3) # number rows/iterations
T = 100
rows = T
grid = np.zeros([rows,columns+2]) # T x n grid (matrix) of zeros (white cells)


# In[38]:


grid[0, int(columns/2)+1]=1 # set only middle cell to 1
#print(grid[0,:])
#TODO: REPLACE THIS WITH GENERATING RANDOM FIRST ROW


# In[40]:


for i in np.arange(0, rows-1): # for each row of the grid (minus first row)
    for j in np.arange(0, columns): # for each column of the grid
        for k in range(8): # for 8 possible patterns
            # compare input pattern k to grid row i, cell j+1 and its neighbors
            if np.array_equal(input_pattern[k,:], grid[i, j:j+3]):
                # if they match, set cell j+1, row i+1:
                # TODO: REPLACE WITH CHOOSING f1 or f2
                grid[i+1, j+1] = f2_bin[k]


# In[42]:


# plot all rows (:) and columns, ignoring the edges (first & last column)
plt.imshow(grid[:,1:columns+1], cmap="Greys", interpolation="nearest")
plt.title("ECA Rule {}".format(f2_W))
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




