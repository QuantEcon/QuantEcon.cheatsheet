.. The QuantEcon MATLAB-Python-Julia Cheat Sheet documentation master file, created by
   sphinx-quickstart on Thu Sep  1 18:39:43 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

QuantEcon Cheat Sheet
=========================================================================

This document summarizes commonly-used, equivalent commands across MATLAB, Python, and Julia.



Creating Vectors
----------------

+-----------------------------+--------------------------+---------------------------------------+--------------------------+
| Operation                   |         MATLAB           | Python                                | Julia                    |
+=============================+==========================+=======================================+==========================+
|                             | .. code-block:: matlab   | .. code-block:: python                | .. code-block:: julia    |
|                             |                          |                                       |                          |
| Create a row vector         |     A = [1 2 3]          |  A = np.array([1, 2, 3])              |     A = [1 2 3]          |
+-----------------------------+--------------------------+---------------------------------------+--------------------------+
|                             | .. code-block:: matlab   | .. code-block:: python                | .. code-block:: julia    |
|                             |                          |                                       |                          |
| Create a column vector      |     A = [1; 2; 3]        |  A = np.array([1, 2, 3]).reshape(3,1) |     A = [1; 2; 3]        |
+-----------------------------+--------------------------+---------------------------------------+--------------------------+
|                             | .. code-block:: matlab   | .. code-block:: python                | .. code-block:: julia    |
|                             |                          |                                       |                          |
| Sequence starting at j      |     A = j:k:n            |  A = np.arange(j, n+1, k)             |     A = j:k:n            |
| ending at n, with           |                          |                                       |                          |
| difference k between points |                          |                                       |                          |
+-----------------------------+--------------------------+---------------------------------------+--------------------------+
|                             | .. code-block:: matlab   | .. code-block:: python                | .. code-block:: julia    |
|                             |                          |                                       |                          |
| Linearly spaced vector      |     A = linspace(1, 5, k)|  A = np.linspace(1, 5, k)             |     A = linspace(1, 5, k)|
| of k points                 |                          |                                       |                          |
+-----------------------------+--------------------------+---------------------------------------+--------------------------+



Creating Matrices
-----------------

+--------------------------------+--------------------------+----------------------------------+--------------------------+
| Operation                      |         MATLAB           | Python                           | Julia                    |
+================================+==========================+==================================+==========================+
|                                | .. code-block:: matlab   | .. code-block:: python           | .. code-block:: julia    |
|                                |                          |                                  |                          |
| Create a matrix                |     A = [1 2; 3 4]       |   A = np.array([[1, 2], [3, 4]]) |     A = [1 2; 3 4]       |
+--------------------------------+--------------------------+----------------------------------+--------------------------+
|                                | .. code-block:: matlab   | .. code-block:: python           | .. code-block:: julia    |
|                                |                          |                                  |                          |
| Create a 2 by 2 matrix of zeros|     A = zeros(2, 2)      |   A = np.zeros((2, 2))           |     A = zeros(2, 2)      |
+--------------------------------+--------------------------+----------------------------------+--------------------------+
|                                | .. code-block:: matlab   | .. code-block:: python           | .. code-block:: julia    |
|                                |                          |                                  |                          |
| Create a 2 by 2 matrix of ones |     A = ones(2, 2)       |   A = np.ones((2, 2))            |     A = ones(2, 2)       |
+--------------------------------+--------------------------+----------------------------------+--------------------------+
|                                | .. code-block:: matlab   | .. code-block:: python           | .. code-block:: julia    |
|                                |                          |                                  |                          |
| Create a 2 by 2 identity matrix|     A = eye(2, 2)        |   A = np.eye(2)                  |     A = eye(2, 2)        |
+--------------------------------+--------------------------+----------------------------------+--------------------------+
|                                | .. code-block:: matlab   | .. code-block:: python           | .. code-block:: julia    |
|                                |                          |                                  |                          |
| Create a diagonal matrix       |     A = diag([1 2 3])    |   A = np.diag([1, 2, 3])         |     A = diagm([1; 2; 3]) |
+--------------------------------+--------------------------+----------------------------------+--------------------------+
|                                | .. code-block:: matlab   | .. code-block:: python           | .. code-block:: julia    |
|                                |                          |                                  |                          |
| Matrix of uniformly distributed|     A = rand(2, 2)       |   A = np.random.rand(2,2)        |     A = rand(2, 2)       |
| random numbers                 |                          |                                  |                          |
+--------------------------------+--------------------------+----------------------------------+--------------------------+
|                                | .. code-block:: matlab   | .. code-block:: python           | .. code-block:: julia    |
|                                |                          |                                  |                          |
| Matrix of random numbers drawn |     A = randn(2, 2)      |   A = np.random.randn(2, 2)      |     A = randn(2, 2)      |
| a standard normal              |                          |                                  |                          |
+--------------------------------+--------------------------+----------------------------------+--------------------------+



Manipulating Vectors and Matrices
---------------------------------

+--------------------------------+-------------------------------+---------------------------+---------------------------+
| Operation                      |         MATLAB                | Python                    | Julia                     |
+================================+===============================+===========================+===========================+
|                                | .. code-block:: matlab        | .. code-block:: python    | .. code-block:: julia     |
|                                |                               |                           |                           |
| Transpose                      |     A'                        |   A.T                     |     A'                    |
+--------------------------------+-------------------------------+---------------------------+---------------------------+
|                                | .. code-block:: matlab        | .. code-block:: python    | .. code-block:: julia     |
|                                |                               |                           |                           |
| Concatenate horizontally       |     A = [[1 2] [1 2]]         |    B = np.array([1, 2])   |     A = [[1 2] [1 2]]     |
|                                |                               |    A = np.hstack((B, B))  |                           |
|                                | or                            |                           | or                        |
|                                |                               |                           |                           |
|                                | .. code-block:: matlab        |                           | .. code-block:: julia     |
|                                |                               |                           |                           |
|                                |     A = horzcat([1 2], [1 2]) |                           |    A = hcat([1 2], [1 2]) |
+--------------------------------+-------------------------------+---------------------------+---------------------------+
|                                | .. code-block:: matlab        | .. code-block:: python    | .. code-block:: julia     |
|                                |                               |                           |                           |
| Concatenate vertically         |     A = [[1 2]; [1 2]]        |    B = np.array([1, 2])   |     A = [[1 2]; [1 2]]    |
|                                |                               |    A = np.vstack((B, B))  |                           |
|                                | or                            |                           | or                        |
|                                |                               |                           |                           |
|                                | .. code-block:: matlab        |                           | .. code-block:: julia     |
|                                |                               |                           |                           |
|                                |     A = vertcat([1 2], [1 2]) |                           |    A = vcat([1 2], [1 2]) |
+--------------------------------+-------------------------------+---------------------------+---------------------------+
|                                | .. code-block:: matlab        | .. code-block:: python    | .. code-block:: julia     |
|                                |                               |                           |                           |
| Reshape (to 5 rows, 2 columns) |    A = reshape(1:10, 5, 2)    |    A = A.reshape(5,2)     |    A = reshape(1:10, 5, 2)|
+--------------------------------+-------------------------------+---------------------------+---------------------------+
|                                | .. code-block:: matlab        | .. code-block:: python    | .. code-block:: julia     |
|                                |                               |                           |                           |
| Convert matrix to vector       |    A(:)                       |    A = A.flatten()        |    A[:]                   |
+--------------------------------+-------------------------------+---------------------------+---------------------------+
|                                | .. code-block:: matlab        | .. code-block:: python    | .. code-block:: julia     |
|                                |                               |                           |                           |
| Flip left/right                |    fliplr(A)                  |    np.fliplr(A)           |    flipdim(A, 2)          |
+--------------------------------+-------------------------------+---------------------------+---------------------------+
|                                | .. code-block:: matlab        | .. code-block:: python    | .. code-block:: julia     |
|                                |                               |                           |                           |
| Flip up/down                   |    flipud(A)                  |    np.flipud(A)           |    flipdim(A, 1)          |
+--------------------------------+-------------------------------+---------------------------+---------------------------+
|                                | .. code-block:: matlab        | .. code-block:: python    | .. code-block:: julia     |
|                                |                               |                           |                           |
| Repeat matrix (3 times in the  |    repmat(A, 3, 4)            |    np.tile(A, (4, 3))     |    repmat(A, 3, 4)        |
| row dimension, 4 times in the  |                               |                           |                           |
| column dimension)              |                               |                           |                           |
+--------------------------------+-------------------------------+---------------------------+---------------------------+



Accessing Vector/Matrix Elements
--------------------------------

+--------------------------------+-------------------------------+-------------------------------+---------------------------+
| Operation                      |         MATLAB                | Python                        | Julia                     |
+================================+===============================+===============================+===========================+
|                                | .. code-block:: matlab        | .. code-block:: python        | .. code-block:: julia     |
|                                |                               |                               |                           |
| Access one element             |     A(2, 2)                   |    A[2, 2]                    |     A[2, 2]               |
+--------------------------------+-------------------------------+-------------------------------+---------------------------+
|                                | .. code-block:: matlab        | .. code-block:: python        | .. code-block:: julia     |
|                                |                               |                               |                           |
| Access specific rows           |    A(1:4, :)                  |    A[0:4, :]                  |    A[1:4, :]              |
+--------------------------------+-------------------------------+-------------------------------+---------------------------+
|                                | .. code-block:: matlab        | .. code-block:: python        | .. code-block:: julia     |
|                                |                               |                               |                           |
| Access specific columns        |    A(:, 1:4)                  |    A[:, 0:4]                  |    A[:, 1:4]              |
+--------------------------------+-------------------------------+-------------------------------+---------------------------+
|                                | .. code-block:: matlab        | .. code-block:: python        | .. code-block:: julia     |
|                                |                               |                               |                           |
| Remove a row                   |    A([1 2 4], :)              |    A[[0, 1, 3], :]            |    A[[1, 2, 4], :]        |
+--------------------------------+-------------------------------+-------------------------------+---------------------------+
|                                | .. code-block:: matlab        | .. code-block:: python        | .. code-block:: julia     |
|                                |                               |                               |                           |
| Diagonals of matrix            |    diag(A)                    |    np.diag(A)                 |    diag(A)                |
+--------------------------------+-------------------------------+-------------------------------+---------------------------+
|                                | .. code-block:: matlab        | .. code-block:: python        | .. code-block:: julia     |
|                                |                               |                               |                           |
| Get dimensions of matrix       |    [nrow ncol] = size(A)      |    nrow, ncol = np.shape(A)   |    nrow, ncol = size(A)   |
+--------------------------------+-------------------------------+-------------------------------+---------------------------+



Mathematical Operations
-----------------------

+--------------------------------+-------------------------------+--------------------------------+---------------------------+
| Operation                      |         MATLAB                | Python                         | Julia                     |
+================================+===============================+================================+===========================+
|                                | .. code-block:: matlab        | .. code-block:: python         | .. code-block:: julia     |
|                                |                               |                                |                           |
| Vector dot product             |     dot(A, B)                 |    np.dot(A, B) or A@B         |     dot(A, B)             |
+--------------------------------+-------------------------------+--------------------------------+---------------------------+
|                                | .. code-block:: matlab        | .. code-block:: python         | .. code-block:: julia     |
|                                |                               |                                |                           |
| Matrix multiplication          |     A*B                       |    np.dot(A, B) or A@B         |     A*B                   |
+--------------------------------+-------------------------------+--------------------------------+---------------------------+
|                                | .. code-block:: matlab        | .. code-block:: python         | .. code-block:: julia     |
|                                |                               |                                |                           |
| Element-wise matrix            |     A.*B                      |    A*B                         |     A.*B                  |
| multiplication                 |                               |                                |                           |
+--------------------------------+-------------------------------+--------------------------------+---------------------------+
|                                | .. code-block:: matlab        | .. code-block:: python         | .. code-block:: julia     |
|                                |                               |                                |                           |
| Matrix to a power              |     A^2                       |    np.linalg.matrix_power(A, 2)|     A^2                   |
+--------------------------------+-------------------------------+--------------------------------+---------------------------+
|                                | .. code-block:: matlab        | .. code-block:: python         | .. code-block:: julia     |
|                                |                               |                                |                           |
| Matrix to a power, elementwise |     A.^2                      |    A**2                        |     A.^2                  |
+--------------------------------+-------------------------------+--------------------------------+---------------------------+
|                                | .. code-block:: matlab        | .. code-block:: python         | .. code-block:: julia     |
|                                |                               |                                |                           |
| Inverse of a matrix            |     inv(A)                    |    np.linalg.inv(A)            |     inv(A)                |
|                                |                               |                                |                           |
|                                | or                            |                                | or                        |
|                                |                               |                                |                           |
|                                | .. code-block:: matlab        |                                | .. code-block:: julia     |
|                                |                               |                                |                           |
|                                |     A^(-1)                    |                                |    A^(-1)                 |
+--------------------------------+-------------------------------+--------------------------------+---------------------------+
|                                | .. code-block:: matlab        | .. code-block:: python         | .. code-block:: julia     |
|                                |                               |                                |                           |
| Determinant of a matrix        |     det(A)                    |    np.linalg.det(A)            |     det(A)                |
+--------------------------------+-------------------------------+--------------------------------+---------------------------+
|                                | .. code-block:: matlab        | .. code-block:: python         | .. code-block:: julia     |
|                                |                               |                                |                           |
| Eigenvalues and eigenvectors   |     [vec, val] = eig(A)       |    val, vec = np.linalg.eig(A) |     val, vec = eig(A)     |
+--------------------------------+-------------------------------+--------------------------------+---------------------------+
|                                | .. code-block:: matlab        | .. code-block:: python         | .. code-block:: julia     |
|                                |                               |                                |                           |
| Euclidean norm                 |     norm(A)                   |    np.linalg.norm(A)           |     norm(A)               |
+--------------------------------+-------------------------------+--------------------------------+---------------------------+
|                                | .. code-block:: matlab        | .. code-block:: python         | .. code-block:: julia     |
|                                |                               |                                |                           |
| Solve linear system            |     A\b                       |    np.linalg.solve(A, b)       |     A\b                   |
| :math:`Ax=b`                   |                               |                                |                           |
+--------------------------------+-------------------------------+--------------------------------+---------------------------+



Sum/Maximum/Minimum
-------------------

+--------------------------------+-------------------------------+---------------------------------+---------------------------+
| Operation                      |         MATLAB                | Python                          | Julia                     |
+================================+===============================+=================================+===========================+
|                                | .. code-block:: matlab        | .. code-block:: python          | .. code-block:: julia     |
|                                |                               |                                 |                           |
| Sum/maximum/minimum of         |     sum(A, 1)                 |    sum(A, 0)                    |     sum(A, 1)             |
| each column                    |     max(A, [], 1)             |    np.amax(A, 0)                |     maximum(A, 1)         |
|                                |     min(A, [], 1)             |    np.amin(A, 0)                |     minimum(A, 1)         |
+--------------------------------+-------------------------------+---------------------------------+---------------------------+
|                                | .. code-block:: matlab        | .. code-block:: python          | .. code-block:: julia     |
|                                |                               |                                 |                           |
| Sum/maximum/minimum of         |     sum(A, 2)                 |    sum(A, 1)                    |     sum(A, 2)             |
| each row                       |     max(A, [], 2)             |    np.amax(A, 1)                |     maximum(A, 2)         |
|                                |     min(A, [], 2)             |    np.amin(A, 1)                |     minimum(A, 2)         |
+--------------------------------+-------------------------------+---------------------------------+---------------------------+
|                                | .. code-block:: matlab        | .. code-block:: python          | .. code-block:: julia     |
|                                |                               |                                 |                           |
| Sum/maximum/minimum of         |     sum(A(:))                 |    np.sum(A)                    |     sum(A)                |
| entire matrix                  |     max(A(:))                 |    np.amax(A)                   |     maximum(A)            |
|                                |     min(A(:))                 |    np.amin(A)                   |     minimum(A)            |
+--------------------------------+-------------------------------+---------------------------------+---------------------------+
|                                | .. code-block:: matlab        | .. code-block:: python          | .. code-block:: julia     |
|                                |                               |                                 |                           |
| Cumulative sum/maximum/minimum |     cumsum(A, 1)              |    np.cumsum(A, 0)              |     cumsum(A, 1)          |
| by row                         |     cummax(A, 1)              |    np.maximum.accumulate(A, 0)  |     cummax(A, 1)          |
|                                |     cummin(A, 1)              |    np.minimum.accumulate(A, 0)  |     cummin(A, 1)          |
+--------------------------------+-------------------------------+---------------------------------+---------------------------+
|                                | .. code-block:: matlab        | .. code-block:: python          | .. code-block:: julia     |
|                                |                               |                                 |                           |
| Cumulative sum/maximum/minimum |     cumsum(A, 2)              |    np.cumsum(A, 1)              |     cumsum(A, 2)          |
| by column                      |     cummax(A, 2)              |    np.maximum.accumulate(A, 1)  |     cummax(A, 2)          |
|                                |     cummin(A, 2)              |    np.minimum.accumulate(A, 1)  |     cummin(A, 2)          |
+--------------------------------+-------------------------------+---------------------------------+---------------------------+



Programming
-----------

+------------------------+----------------------------+----------------------------+-------------------------------+
| Operation              |         MATLAB             | Python                     | Julia                         |
+========================+============================+============================+===============================+
|                        | .. code-block:: matlab     | .. code-block:: python     | .. code-block:: julia         |
|                        |                            |                            |                               |
| Comment one line       |     % This is a comment    |    # This is a comment     |     # This is a comment       |
+------------------------+----------------------------+----------------------------+-------------------------------+
|                        | .. code-block:: matlab     | .. code-block:: python     | .. code-block:: julia         |
|                        |                            |                            |                               |
| Comment block          |     %{                     |    # Block                 |     #=                        |
|                        |     Comment block          |    # comment               |     Comment block             |
|                        |     %}                     |    # following PEP8        |     =#                        |
+------------------------+----------------------------+----------------------------+-------------------------------+
|                        | .. code-block:: matlab     | .. code-block:: python     | .. code-block:: julia         |
|                        |                            |                            |                               |
| For loop               |     for i = 1:N            |    for i in range(n):      |     for i = 1:N               |
|                        |        % do something      |        # do something      |        # do something         |
|                        |     end                    |                            |     end                       |
+------------------------+----------------------------+----------------------------+-------------------------------+
|                        | .. code-block:: matlab     | .. code-block:: python     | .. code-block:: julia         |
|                        |                            |                            |                               |
| While loop             |     while i <= N           |    while i <= N:           |     while i <= N              |
|                        |        % do something      |        # do something      |        # do something         |
|                        |     end                    |                            |     end                       |
+------------------------+----------------------------+----------------------------+-------------------------------+
|                        | .. code-block:: matlab     | .. code-block:: python     | .. code-block:: julia         |
|                        |                            |                            |                               |
| If statement           |     if i <= N              |    if i <= N:              |     if i <= N                 |
|                        |        % do something      |       # do something       |        # do something         |
|                        |     end                    |                            |     end                       |
+------------------------+----------------------------+----------------------------+-------------------------------+
|                        | .. code-block:: matlab     | .. code-block:: python     | .. code-block:: julia         |
|                        |                            |                            |                               |
| If/else statement      |     if i <= N              |   if i <= N:               |    if i <= N                  |
|                        |        % do something      |       # do something       |       # do something          |
|                        |     else                   |   else:                    |    else                       |
|                        |        % do something else |       # so something else  |       # do something else     |
|                        |     end                    |                            |    end                        |
+------------------------+----------------------------+----------------------------+-------------------------------+
|                        | .. code-block:: matlab     | .. code-block:: python     | .. code-block:: julia         |
|                        |                            |                            |                               |
| Print text and variable|     x = 10                 |    x = 10                  |    x = 10                     |
| to screen              |     fprintf('The value of  |    print('The value of     |    println("The value of      |
|                        |     x is %d. \n', x)       |    x is {}.'.format(x))    |    x is $(x).")               |
+------------------------+----------------------------+----------------------------+-------------------------------+
|                        | .. code-block:: matlab     | .. code-block:: python     | .. code-block:: julia         |
|                        |                            |                            |                               |
| Function: one line/    |     fun = @(x) x^2         |    fun = lambda x: x**2    |     fun(x) = x^2              |
| anonymous              |                            |                            |                               |
+------------------------+----------------------------+----------------------------+-------------------------------+
|                        | .. code-block:: matlab     | .. code-block:: python     | .. code-block:: julia         |
|                        |                            |                            |                               |
| Function: multiple     |     function out  = fun(x) |    def fun(x):             |     function fun(x)           |
| lines                  |        out = x^2           |        return x**2         |        return x^2             |
|                        |     end                    |                            |     end                       |
+------------------------+----------------------------+----------------------------+-------------------------------+




