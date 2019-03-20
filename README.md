## L1-PCA-Algorithm-Implementation

### PCA yields the directions (principal components) that maximize the variance of the data. 

PCA projects the entire dataset onto a different feature subspace. Often, the desired goal is to reduce the dimensions of a d-dimensional dataset by projecting it onto a k-dimensional subspace (where k<d) in order to increase the computational efficiency while retaining most of the information.

We need to select the hyperplane such that when all the points are projected onto it, they are maximally spread out. In other words, we want the axis of maximal variance. If we pick a line that cuts through our data diagonally, that is the axis where the data would be most spread. This gives the birth to eigenfaces or eigenvector. We then sort these eigen values of these vectors in a descending order and pick the highest value that gives the basic understanding of those subsets of the matrix as vectors that are important components to each image. 
