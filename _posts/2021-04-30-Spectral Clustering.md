---
layout: post
title: Spectral Clustering
image: benbrill.github.io\images\ucla-math.png
---
# Creating a Spectral Clustering algorithm

In this problem, we'll study *spectral clustering*. Spectral clustering is an important tool for identifying meaningful parts of data sets with complex structure. To start, let's look at an example where we *don't* need spectral clustering. 


```python
import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt
```


```python
n = 200
np.random.seed(1111)
X, y = datasets.make_blobs(n_samples=n, shuffle=True, random_state=None, centers = 2, cluster_std = 2.0)
plt.scatter(X[:,0], X[:,1])
```




    <matplotlib.collections.PathCollection at 0x2218cea3b20>




    
![svg](blogpost2_files/blogpost2_2_1.svg)
    


*Clustering* refers to the task of separating this data set into the two natural "blobs." K-means is a very common way to achieve this task, which has good performance on circular-ish blobs like these: 


```python
from sklearn.cluster import KMeans
km = KMeans(n_clusters = 2)
km.fit(X)

plt.scatter(X[:,0], X[:,1], c = km.predict(X))
```




    <matplotlib.collections.PathCollection at 0x2218e6f9af0>




    
![svg](blogpost2_files/blogpost2_4_1.svg)
    


### Harder Clustering

That was all well and good, but what if our data is "shaped weird"? 


```python
np.random.seed(1234)
n = 200
X, y = datasets.make_moons(n_samples=n, shuffle=True, noise=0.05, random_state=None)
plt.scatter(X[:,0], X[:,1])
```




    <matplotlib.collections.PathCollection at 0x21136364310>




    
![svg](blogpost2_files/blogpost2_6_1.svg)
    


We can still make out two meaningful clusters in the data, but now they aren't blobs but crescents. As before, the Euclidean coordinates of the data points are contained in the matrix `X`, while the labels of each point are contained in `y`. Now k-means won't work so well, because k-means is, by design, looking for circular clusters. 


```python
km = KMeans(n_clusters = 2)
km.fit(X)
plt.scatter(X[:,0], X[:,1], c = km.predict(X))
```




    <matplotlib.collections.PathCollection at 0x2219003a550>




    
![svg](blogpost2_files/blogpost2_8_1.svg)
    


Whoops! That's not right! 

As we'll see, spectral clustering is able to correctly cluster the two crescents. In the following problems, you will derive and implement spectral clustering. 

## Constructing a similarity matrix

Instead of looking for circles as a method to get clusters, we can use the distance between points to generate our clusters. To do this, we will generate an $ n \times n$ matrix where each row and column corresponds to a point in our matrix $\mathbf{X}$. Each entry $ij$ will correspond to the distance between points $\mathbf{X}_i$ and $\mathbf{X}_j$. 

Once we have a matrix of the pairwise distances, we will see if they lie within a specified distance, called epsilon. If the distance between two points is less than epsilon, we will replace that value with a 1 in a new similarity matrix $\mathbf{A}$, indicating these points have a potential connection. Otherwise, a pair of points that lies outside the distance epsilon will have a value 0. Beacause we do not want to compare the same points to each other, we will place 0's along the digagonal of the matrix. 


```python
from sklearn.metrics import pairwise_distances
def make_simMatrix(X, e):
    # create matrix that includes distances between each pair of points
    r = pairwise_distances(X)
    # square matrix of shape n x n
    new = np.ones((n,n)) 
    # replace entries less greater than epsilon with 0
    new[r > e] = 0 
    # fill the diagonal with 0s
    np.fill_diagonal(new, 0)
    return new
A = make_simMatrix(X, 0.4)
A
```




    array([[0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 1., 0.],
           ...,
           [0., 0., 0., ..., 0., 1., 1.],
           [0., 0., 1., ..., 1., 0., 1.],
           [0., 0., 0., ..., 1., 1., 0.]])



We can see here we have a $ n \times n$ matrix, with each entry corresponding to whether or not there is a connection between two points, signified by the number 1. 

## Part B

The matrix `A` now contains information about which points are near (within distance `epsilon`) which other points. We now pose the task of clustering the data points in `X` as the task of partitioning the rows and columns of `A`. 

Let $d_i = \sum_{j = 1}^n a_{ij}$ be the $i$th row-sum of $\mathbf{A}$, which is also called the *degree* of $i$. Let $C_0$ and $C_1$ be two clusters of the data points. We assume that every data point is in either $C_0$ or $C_1$. The cluster membership as being specified by `y`. We think of `y[i]` as being the label of point `i`. So, if `y[i] = 1`, then point `i` (and therefore row $i$ of $\mathbf{A}$) is an element of cluster $C_1$.  

The *binary norm cut objective* of a matrix $\mathbf{A}$ is the function 

$$N_{\mathbf{A}}(C_0, C_1)\equiv \mathbf{cut}(C_0, C_1)\left(\frac{1}{\mathbf{vol}(C_0)} + \frac{1}{\mathbf{vol}(C_1)}\right)\;.$$

In this expression, 
- $\mathbf{cut}(C_0, C_1) \equiv \sum_{i \in C_0, j \in C_1} a_{ij}$ is the *cut* of the clusters $C_0$ and $C_1$. 
- $\mathbf{vol}(C_0) \equiv \sum_{i \in C_0}d_i$, where $d_i = \sum_{j = 1}^n a_{ij}$ is the *degree* of row $i$ (the total number of all other rows related to row $i$ through $A$). The *volume* of cluster $C_0$ is a measure of the size of the cluster. 

A pair of clusters $C_0$ and $C_1$ is considered to be a "good" partition of the data when $N_{\mathbf{A}}(C_0, C_1)$ is small. To see why, let's look at each of the two factors in this objective function separately. 


#### B.1 The Cut Term

First, the cut term $\mathbf{cut}(C_0, C_1)$ is the number of nonzero entries in $\mathbf{A}$ that relate points in cluster $C_0$ to points in cluster $C_1$. Saying that this term should be small is the same as saying that points in $C_0$ shouldn't usually be very close to points in $C_1$. 

Write a function called `cut(A,y)` to compute the cut term. You can compute it by summing up the entries `A[i,j]` for each pair of points `(i,j)` in different clusters. 

It's ok if you use `for`-loops in this function -- we are going to see a more efficient view of this problem soon. 


```python
def cut(A,y):
    c = 0 # intialize cut
    # iterate through each entry of A
    for row in range(A.shape[0]):
        for col in range(A.shape[1]):
            # each row and column each represent one point
            # check see if points are in same column
            if y[row] != y[col]:
                # check to see if there is a connection that needs to be cut
                if A[row][col] == 1:
                    c +=1
    return c
cut(A,y)
```




    26



Compute the cut objective for the true clusters `y`. Then, generate a random vector of random labels of length `n`, with each label equal to either 0 or 1. Check the cut objective for the random labels. You should find that the cut objective for the true labels is *much* smaller than the cut objective for the random labels. 

This shows that this part of the cut objective indeed favors the true clusters over the random ones. 


```python
newLabs = np.random.randint(2, size = n)
cut(A, newLabs)
```




    2184



#### B.2 The Volume Term 

Now take a look at the second factor in the norm cut objective. This is the *volume term*. As mentioned above, the *volume* of cluster $C_0$ is a measure of how "big" cluster $C_0$ is. If we choose cluster $C_0$ to be small, then $\mathbf{vol}(C_0)$ will be small and $\frac{1}{\mathbf{vol}(C_0)}$ will be large, leading to an undesirable higher objective value. 

Synthesizing, the binary normcut objective asks us to find clusters $C_0$ and $C_1$ such that:

1. There are relatively few entries of $\mathbf{A}$ that join $C_0$ and $C_1$. 
2. Neither $C_0$ and $C_1$ are too small. 

Write a function called `vols(A,y)` which computes the volumes of $C_0$ and $C_1$, returning them as a tuple. For example, `v0, v1 = vols(A,y)` should result in `v0` holding the volume of cluster `0` and `v1` holding the volume of cluster `1`. Then, write a function called `normcut(A,y)` which uses `cut(A,y)` and `vols(A,y)` to compute the binary normalized cut objective of a matrix `A` with clustering vector `y`. 

***Note***: No for-loops in this part. Each of these functions should be implemented in five lines or less. 

- $\mathbf{vol}(C_0) \equiv \sum_{i \in C_0}d_i$, where $d_i = \sum_{j = 1}^n a_{ij}$ is the *degree* of row $i$ (the total number of all other rows related to row $i$ through $A$). The *volume* of cluster $C_0$ is a measure of the size of the cluster.


```python
def vols(A,y):
    d = A.sum(axis = 0) # sum of each row
    c0 = d[y == 0].sum() # get sum of rows in cluster 0
    c1 = d[y == 1].sum() # get sum of rows in cluster 1
    return(c0, c1)
def normcut(A,y):
    volume = vols(A,y)
    # compute normcut according to formula
    return cut(A,y) * ((1/volume[0]) + (1/volume[1]))
```

Now, compare the `normcut` objective using both the true labels `y` and the fake labels you generated above. What do you observe about the normcut for the true labels when compared to the normcut for the fake labels? 


```python
normcut(A,y), normcut(A, newLabs)
```




    (0.02303682466323045, 2.0480047195518316)



## Part C

We have now defined a normalized cut objective which takes small values when the input clusters are (a) joined by relatively few entries in $A$ and (b) not too small. One approach to clustering is to try to find a cluster vector `y` such that `normcut(A,y)` is small. However, this is an NP-hard combinatorial optimization problem, which means that may not be possible to find the best clustering in practical time, even for relatively small data sets. We need a math trick! 

Here's the trick: define a new vector $\mathbf{z} \in \mathbb{R}^n$ such that: 

$$
z_i = 
\begin{cases}
    \frac{1}{\mathbf{vol}(C_0)} &\quad \text{if } y_i = 0 \\ 
    -\frac{1}{\mathbf{vol}(C_1)} &\quad \text{if } y_i = 1 \\ 
\end{cases}
$$


Note that the signs of  the elements of $\mathbf{z}$ contain all the information from $\mathbf{y}$: if $i$ is in cluster $C_0$, then $y_i = 0$ and $z_i > 0$. 

Next, if you like linear algebra, you can show that 

$$\mathbf{N}_{\mathbf{A}}(C_0, C_1) = 2\frac{\mathbf{z}^T (\mathbf{D} - \mathbf{A})\mathbf{z}}{\mathbf{z}^T\mathbf{D}\mathbf{z}}\;,$$

where $\mathbf{D}$ is the diagonal matrix with nonzero entries $d_{ii} = d_i$, and  where $d_i = \sum_{j = 1}^n a_i$ is the degree (row-sum) from before.  

1. Write a function called `transform(A,y)` to compute the appropriate $\mathbf{z}$ vector given `A` and `y`, using the formula above. 
2. Then, check the equation above that relates the matrix product to the normcut objective, by computing each side separately and checking that they are equal. 
3. While you're here, also check the identity $\mathbf{z}^T\mathbf{D}\mathbb{1} = 0$, where $\mathbb{1}$ is the vector of `n` ones (i.e. `np.ones(n)`). This identity effectively says that $\mathbf{z}$ should contain roughly as many positive as negative entries. 

#### Programming Note

You can compute $\mathbf{z}^T\mathbf{D}\mathbf{z}$ as `z@D@z`, provided that you have constructed these objects correctly. 

#### Note

The equation above is exact, but computer arithmetic is not! `np.isclose(a,b)` is a good way to check if `a` is "close" to `b`, in the sense that they differ by less than the smallest amount that the computer is (by default) able to quantify. 

Also, still no for-loops. 


```python
def transform(A,y):
     new = y.copy()
     # assign positive or neg vol according to label
     new[new == 0] = vols(A,y)[0]
     new[new == 1] = -vols(A,y)[1]
     new = 1/new
     return new

```


```python
z = transform(A,y)
z
```




    array([-0.00045106, -0.00045106,  0.00043497,  0.00043497,  0.00043497,
            0.00043497,  0.00043497,  0.00043497, -0.00045106, -0.00045106,
           -0.00045106,  0.00043497, -0.00045106, -0.00045106, -0.00045106,
           -0.00045106, -0.00045106,  0.00043497,  0.00043497,  0.00043497,
           -0.00045106, -0.00045106, -0.00045106,  0.00043497,  0.00043497,
           -0.00045106,  0.00043497, -0.00045106, -0.00045106,  0.00043497,
            0.00043497, -0.00045106, -0.00045106, -0.00045106, -0.00045106,
           -0.00045106,  0.00043497, -0.00045106, -0.00045106,  0.00043497,
           -0.00045106,  0.00043497,  0.00043497,  0.00043497,  0.00043497,
            0.00043497,  0.00043497, -0.00045106, -0.00045106, -0.00045106,
            0.00043497,  0.00043497, -0.00045106, -0.00045106,  0.00043497,
            0.00043497, -0.00045106, -0.00045106, -0.00045106,  0.00043497,
            0.00043497,  0.00043497, -0.00045106,  0.00043497, -0.00045106,
            0.00043497,  0.00043497,  0.00043497,  0.00043497, -0.00045106,
           -0.00045106, -0.00045106, -0.00045106,  0.00043497,  0.00043497,
            0.00043497, -0.00045106,  0.00043497, -0.00045106,  0.00043497,
            0.00043497,  0.00043497,  0.00043497, -0.00045106, -0.00045106,
           -0.00045106, -0.00045106,  0.00043497,  0.00043497,  0.00043497,
            0.00043497, -0.00045106,  0.00043497,  0.00043497, -0.00045106,
           -0.00045106, -0.00045106, -0.00045106,  0.00043497,  0.00043497,
           -0.00045106, -0.00045106, -0.00045106,  0.00043497, -0.00045106,
           -0.00045106,  0.00043497,  0.00043497, -0.00045106, -0.00045106,
            0.00043497,  0.00043497,  0.00043497, -0.00045106,  0.00043497,
            0.00043497,  0.00043497,  0.00043497, -0.00045106, -0.00045106,
           -0.00045106,  0.00043497, -0.00045106, -0.00045106, -0.00045106,
            0.00043497, -0.00045106,  0.00043497, -0.00045106, -0.00045106,
            0.00043497,  0.00043497,  0.00043497,  0.00043497, -0.00045106,
           -0.00045106, -0.00045106, -0.00045106, -0.00045106, -0.00045106,
           -0.00045106, -0.00045106,  0.00043497,  0.00043497, -0.00045106,
            0.00043497,  0.00043497,  0.00043497,  0.00043497,  0.00043497,
            0.00043497,  0.00043497,  0.00043497, -0.00045106, -0.00045106,
            0.00043497, -0.00045106,  0.00043497,  0.00043497,  0.00043497,
           -0.00045106, -0.00045106,  0.00043497,  0.00043497, -0.00045106,
           -0.00045106, -0.00045106,  0.00043497,  0.00043497,  0.00043497,
           -0.00045106,  0.00043497, -0.00045106, -0.00045106, -0.00045106,
            0.00043497,  0.00043497,  0.00043497,  0.00043497, -0.00045106,
           -0.00045106,  0.00043497,  0.00043497, -0.00045106, -0.00045106,
           -0.00045106,  0.00043497, -0.00045106, -0.00045106, -0.00045106,
            0.00043497, -0.00045106,  0.00043497, -0.00045106, -0.00045106,
           -0.00045106, -0.00045106,  0.00043497,  0.00043497,  0.00043497])




```python
D = np.zeros((200,200))
np.fill_diagonal(D, A.sum(axis= 0))
# calculate using formula
rhs = 2*((z.T@(D-A)@z))/(z@D@z)
# check to see if equal using computer precision
np.isclose(normcut(A,y), rhs)
```




    True



## Part D

In the last part, we saw that the problem of minimizing the normcut objective is mathematically related to the problem of minimizing the function 

$$ R_\mathbf{A}(\mathbf{z})\equiv \frac{\mathbf{z}^T (\mathbf{D} - \mathbf{A})\mathbf{z}}{\mathbf{z}^T\mathbf{D}\mathbf{z}} $$

subject to the condition $\mathbf{z}^T\mathbf{D}\mathbb{1} = 0$. It's actually possible to bake this condition into the optimization, by substituting for $\mathbf{z}$ the orthogonal complement of $\mathbf{z}$ relative to $\mathbf{D}\mathbf{1}$. In the code below, I define an `orth_obj` function which handles this for you. 

Use the `minimize` function from `scipy.optimize` to minimize the function `orth_obj` with respect to $\mathbf{z}$. Note that this computation might take a little while. Explicit optimization can be pretty slow! Give the minimizing vector a name `z_`. 


```python
def orth(u, v):
    return (u @ v) / (v @ v) * v

e = np.ones(n) 

d = D @ e

def orth_obj(z):
    z_o = z - orth(z, d)
    return (z_o @ (D - A) @ z_o)/(z_o @ D @ z_o)
```


```python
from scipy.optimize import minimize

z_ = minimize(orth_obj, z).x
z_
```




    array([-1.83358192e-03, -2.41701407e-03, -1.20217848e-03, -1.41748055e-03,
           -9.85566538e-04, -1.23081951e-03, -5.85637745e-04, -8.67448765e-04,
           -2.13354581e-03, -1.82267270e-03, -2.18401808e-03, -1.18174147e-03,
           -1.12567412e-03, -2.27564662e-03, -2.49836000e-03, -2.32669078e-03,
           -2.03861438e-03, -1.27306582e-03, -1.24215828e-03, -1.14279389e-03,
           -2.24028198e-03, -1.12826575e-03, -2.25932974e-03, -1.41888901e-03,
           -8.52792711e-04, -1.94371416e-03, -1.07873687e-03, -2.00277669e-03,
           -2.16749789e-03, -1.20004626e-03, -1.00740381e-03, -2.04776516e-03,
           -2.38485923e-03, -2.15580280e-03, -2.15337541e-03, -2.30809736e-03,
           -8.52792709e-04, -2.46819872e-03, -2.33955477e-03, -1.16161974e-03,
           -2.22245287e-03, -1.27165024e-03, -1.12814187e-03, -3.46791450e-04,
           -9.25811423e-04, -1.41511352e-03, -1.27232761e-03, -2.31470109e-03,
           -2.35157140e-03, -1.11525028e-03, -1.13546644e-03, -1.14953692e-03,
           -2.25932974e-03, -2.21226758e-03, -7.09399119e-04, -1.15736913e-03,
           -2.38244765e-03, -1.43644797e-03, -2.25733591e-03, -1.34595432e-03,
           -4.20084181e-04, -1.11083763e-03, -2.05784603e-03, -1.25847371e-03,
           -2.04517353e-03, -1.08280321e-03, -4.20085215e-04, -8.54285131e-04,
           -1.42133407e-03, -2.33025254e-03, -2.03782912e-03, -2.38888501e-03,
           -2.18983062e-03, -1.34291608e-03, -1.14324611e-03, -1.65971009e-03,
           -1.93553493e-03, -1.27228978e-03, -2.39837146e-03, -1.09185745e-03,
           -1.44533289e-03, -1.05098371e-03, -1.22156255e-03, -2.26382833e-03,
           -2.25739439e-03, -2.33467196e-03, -2.15580280e-03, -1.35423075e-03,
           -1.13220494e-03, -1.27201952e-03, -7.25712474e-04, -2.05784603e-03,
           -1.17085020e-03, -1.13546644e-03, -1.43644756e-03, -1.82267270e-03,
           -2.46819872e-03, -2.40304151e-03, -9.53873197e-04, -1.14953692e-03,
           -2.47013673e-03, -2.35157140e-03, -2.31349781e-03, -6.55809136e-04,
           -2.22984646e-03, -2.33799496e-03, -9.04904381e-04, -1.34564810e-03,
           -2.35511032e-03, -2.26930963e-03, -9.81060423e-04, -1.23216622e-03,
           -1.20415245e-03, -2.33191750e-03, -1.12893706e-03, -7.78376693e-04,
            8.23734807e-05, -5.85638220e-04, -1.43644809e-03, -2.39001730e-03,
           -2.60596367e-03, -1.20834473e-03, -2.22118170e-03, -2.30538524e-03,
           -2.11880220e-03, -1.15342839e-03, -2.37693705e-03, -1.35917291e-03,
           -2.04776517e-03, -1.43644795e-03, -1.00395171e-03, -5.85639986e-04,
           -1.34595432e-03, -1.07238383e-03, -2.38598056e-03, -2.47173902e-03,
           -2.32133494e-03, -2.07601349e-03, -2.22248603e-03, -1.55192706e-03,
           -1.84556225e-03, -2.37802235e-03, -1.21757625e-03, -1.27588063e-03,
           -1.62197782e-03, -1.21523765e-03, -1.03793631e-03, -1.23437384e-03,
           -1.12840906e-03, -1.20795124e-03, -1.12814187e-03, -1.00395171e-03,
           -1.14324610e-03, -1.94371416e-03, -2.50540645e-03, -4.20085283e-04,
           -1.97564367e-03, -1.14953692e-03, -1.21523765e-03, -1.22156255e-03,
           -2.42200051e-03, -2.24456273e-03, -1.08612710e-03, -1.35423075e-03,
           -1.51454801e-03, -1.94371416e-03, -2.33799496e-03, -1.27588063e-03,
           -5.85635734e-04, -4.45230918e-04, -2.53373470e-03, -1.16183500e-03,
           -2.26930963e-03, -2.32100513e-03, -2.33467196e-03, -1.20096473e-03,
           -1.20024879e-03, -1.23437385e-03, -1.41541366e-03, -2.30538524e-03,
           -2.03861438e-03, -1.15075500e-03, -1.11083763e-03, -1.99740084e-03,
           -2.13354581e-03, -2.05040652e-03, -9.86822351e-04, -2.15337541e-03,
           -2.38244765e-03, -2.35157140e-03, -1.20096473e-03, -2.42200050e-03,
           -9.86822352e-04, -2.52226861e-03, -2.27495909e-03, -2.32669078e-03,
           -2.09431749e-03, -1.33500191e-03, -1.49819389e-03, -1.33636813e-03])



**Note**: there's a cheat going on here! We originally specified that the entries of $\mathbf{z}$ should take only one of two values (back in Part C), whereas now we're allowing the entries to have *any* value! This means that we are no longer exactly optimizing the normcut objective, but rather an approximation. This cheat is so common that deserves a name: it is called the *continuous relaxation* of the normcut problem. 

## Part E

Recall that, by design, only the sign of `z_min[i]` actually contains information about the cluster label of data point `i`. Plot the original data, using one color for points such that `z_min[i] < 0` and another color for points such that `z_min[i] >= 0`. 

Does it look like we came close to correctly clustering the data? 


```python
colors = np.zeros(n)
colors[z_ < 0] = 0
colors[z_ > 0] = 1
colors
plt.scatter(X[:,0], X[:,1], c = colors)
```




    <matplotlib.collections.PathCollection at 0x221903ae070>




    
![svg](blogpost2_files/blogpost2_30_1.svg)
    


## Part F

Explicitly optimizing the orthogonal objective is  *way* too slow to be practical. If spectral clustering required that we do this each time, no one would use it. 

The reason that spectral clustering actually matters, and indeed the reason that spectral clustering is called *spectral* clustering, is that we can actually solve the problem from Part E using eigenvalues and eigenvectors of matrices. 

Recall that what we would like to do is minimize the function 

$$ R_\mathbf{A}(\mathbf{z})\equiv \frac{\mathbf{z}^T (\mathbf{D} - \mathbf{A})\mathbf{z}}{\mathbf{z}^T\mathbf{D}\mathbf{z}} $$

with respect to $\mathbf{z}$, subject to the condition $\mathbf{z}^T\mathbf{D}\mathbb{1} = 0$. 

The Rayleigh-Ritz Theorem states that the minimizing $\mathbf{z}$ must be the solution with smallest eigenvalue of the generalized eigenvalue problem 

$$ (\mathbf{D} - \mathbf{A}) \mathbf{z} = \lambda \mathbf{D}\mathbf{z}\;, \quad \mathbf{z}^T\mathbf{D}\mathbb{1} = 0$$

which is equivalent to the standard eigenvalue problem 

$$ \mathbf{D}^{-1}(\mathbf{D} - \mathbf{A}) \mathbf{z} = \lambda \mathbf{z}\;, \quad \mathbf{z}^T\mathbb{1} = 0\;.$$

Why is this helpful? Well, $\mathbb{1}$ is actually the eigenvector with smallest eigenvalue of the matrix $\mathbf{D}^{-1}(\mathbf{D} - \mathbf{A})$. 

> So, the vector $\mathbf{z}$ that we want must be the eigenvector with  the *second*-smallest eigenvalue. 

Construct the matrix $\mathbf{L} = \mathbf{D}^{-1}(\mathbf{D} - \mathbf{A})$, which is often called the (normalized) *Laplacian* matrix of the similarity matrix $\mathbf{A}$. Find the eigenvector corresponding to its second-smallest eigenvalue, and call it `z_eig`. Then, plot the data again, using the sign of `z_eig` as the color. How did we do? 


```python
L = np.linalg.inv(D) @ (D - A)
Lam, U = np.linalg.eig(L)
ix = Lam.argsort()

Lam, U = Lam[ix], U[:,ix]

# 2nd smallest eigenvalue and corresponding eigenvector
z_eig = U[:,1]
colors = np.zeros(n)
colors[z_eig > 0] = 1
plt.scatter(X[:,0], X[:,1], c = colors)
```




    <matplotlib.collections.PathCollection at 0x22194cf1160>




    
![svg](blogpost2_files/blogpost2_32_1.svg)
    


In fact, `z_eig` should be proportional to `z_min`, although this won't be exact because minimization has limited precision by default. 

## Part G

Synthesize your results from the previous parts. In particular, write a function called `spectral_clustering(X, epsilon)` which takes in the input data `X` (in the same format as Part A) and the distance threshold `epsilon` and performs spectral clustering, returning an array of binary labels indicating whether data point `i` is in group `0` or group `1`. Demonstrate your function using the supplied data from the beginning of the problem. 

#### Notes

Despite the fact that this has been a long journey, the final function should be quite short. You should definitely aim to keep your solution under 10, very compact lines. 

**In this part only, please supply an informative docstring!** 

#### Outline

Given data, you need to: 

1. Construct the similarity matrix. 
2. Construct the Laplacian matrix. 
3. Compute the eigenvector with second-smallest eigenvalue of the Laplacian matrix. 
4. Return labels based on this eigenvector. 


```python
def spectral_clustering(X, epsilon): 
    """
    Computes clusters based for a given matrix of points using a 
    spectral clustering algorithm
    ==================================
    Parameters
    ----------------------------------
    X: an (n, 2) array of n points in 2d space

    epsilon: the distance between points to discrimnate different
    clusters
    =================================
    Returns
    labels: an array of the corresponding label of cluster for each
    given point
    """
    # generate similarity matrix
    A = make_simMatrix(X, epsilon)
    # generate matrix D with diagonal containing degree of row
    D = np.zeros((n,n))
    np.fill_diagonal(D, A.sum(axis= 0))
    # compute eigen value
    L = np.linalg.inv(D) @ (D - A)
    Lam, U = np.linalg.eig(L)
    ix = Lam.argsort()
    # get eigenvector corresponding to second lowest eigenvalue
    z_eig = U[:,ix][:,1]
    labels = np.zeros(n)
    labels[z_eig > 0] = 1
    return labels
```


```python
X.shape
```




    (200, 2)




```python
colors = spectral_clustering(X, 0.4)
plt.scatter(X[:,0], X[:,1], c = colors)
```




    <matplotlib.collections.PathCollection at 0x211364ba550>




    
![svg](blogpost2_files/blogpost2_37_1.svg)
    


## Part H

Run a few experiments using your function, by generating different data sets using `make_moons`. What happens when you increase the `noise`? Does spectral clustering still find the two half-moon clusters? For these experiments, you may find it useful to increase `n` to `1000` or so -- we can do this now, because of our fast algorithm! 


```python
n = 1000
X, y = datasets.make_moons(n_samples=n, shuffle=True, noise=0.25, random_state=None)
colors = spectral_clustering(X, 0.4)
plt.scatter(X[:,0], X[:,1], c = colors)
```




    <matplotlib.collections.PathCollection at 0x21136190490>




    
![svg](blogpost2_files/blogpost2_39_1.svg)
    



```python
n = 1000
X, y = datasets.make_moons(n_samples=n, shuffle=True, noise=0.10, random_state=None)
colors = spectral_clustering(X, 0.4)
plt.scatter(X[:,0], X[:,1], c = colors)
```




    <matplotlib.collections.PathCollection at 0x211362685e0>




    
![svg](blogpost2_files/blogpost2_40_1.svg)
    


## Part I

Now try your spectral clustering function on another data set -- the bull's eye! 


```python
n = 1000
X, y = datasets.make_circles(n_samples=n, shuffle=True, noise=0.05, random_state=None, factor = 0.4)
plt.scatter(X[:,0], X[:,1])
```




    <matplotlib.collections.PathCollection at 0x2113632b970>




    
![svg](blogpost2_files/blogpost2_42_1.svg)
    


There are two concentric circles. As before k-means will not do well here at all. 


```python
km = KMeans(n_clusters = 2)
km.fit(X)
plt.scatter(X[:,0], X[:,1], c = km.predict(X))
```




    <matplotlib.collections.PathCollection at 0x7f82dfb0bc18>




    
![png](blogpost2_files/blogpost2_44_1.png)
    


Can your function successfully separate the two circles? Some experimentation here with the value of `epsilon` is likely to be required. Try values of `epsilon` between `0` and `1.0` and describe your findings. For roughly what values of `epsilon` are you able to correctly separate the two rings? 


```python
colors = spectral_clustering(X, 0.4)
plt.scatter(X[:,0], X[:,1], c = colors)
```




    <matplotlib.collections.PathCollection at 0x21136389070>




    
![svg](blogpost2_files/blogpost2_46_1.svg)
    


## Part J

Great work! Turn this notebook into a blog post with plenty of helpful explanation for your reader. 
