.. raw:: html

	<style type="text/css">.menu>li.comparison-on>a {border-color:#444;cursor: default;}</style>

.. toctree::
   :hidden:

   julia-cheatsheet
   python-cheatsheet
   stats-cheatsheet

MATLAB--Python--Julia cheatsheet
===========================================

Dependencies and Setup 
--------------------------

In the Python code we assume that you have already run :code:`import numpy as np`

In the Julia, we assume you are using **v1.0.2 or later** with Compat **v1.3.0 or later** and have run 

.. code-block:: julia 

    using LinearAlgebra, Statistics, Compat

Creating Vectors
----------------

.. container:: multilang-table

    +----------------------------+---------------------------+----------------------------------------+---------------------------+
    |         Operation          |           MATLAB          |                 Python                 |           Julia           |
    +============================+===========================+========================================+===========================+
    |                            | .. code-block:: matlab    | .. code-block:: python                 | .. code-block:: julia     |
    |                            |                           |                                        |                           |
    | Row vector: size (1, n)    |     A = [1 2 3]           |  A = np.array([1, 2, 3]).reshape(1, 3) |     A = [1 2 3]           |
    +----------------------------+---------------------------+----------------------------------------+---------------------------+
    |                            | .. code-block:: matlab    | .. code-block:: python                 | .. code-block:: julia     |
    |                            |                           |                                        |                           |
    | Column vector: size (n, 1) |     A = [1; 2; 3]         |  A = np.array([1, 2, 3]).reshape(3, 1) |     A = [1 2 3]'          |
    +----------------------------+---------------------------+----------------------------------------+---------------------------+
    |                            | Not possible              | .. code-block:: python                 | .. code-block:: julia     |
    |                            |                           |                                        |                           |
    | 1d array: size (n, )       |                           |  A = np.array([1, 2, 3])               |     A = [1; 2; 3]         |
    |                            |                           |                                        |                           |
    |                            |                           |                                        | or                        |
    |                            |                           |                                        |                           |
    |                            |                           |                                        | .. code-block:: julia     |
    |                            |                           |                                        |                           |
    |                            |                           |                                        |     A = [1, 2, 3]         |
    +----------------------------+---------------------------+----------------------------------------+---------------------------+
    |                            | .. code-block:: matlab    | .. code-block:: python                 | .. code-block:: julia     |
    |                            |                           |                                        |                           |
    | Integers from j to n with  |     A = j:k:n             |  A = np.arange(j, n+1, k)              |     A = j:k:n             |
    | step size k                |                           |                                        |                           |
    +----------------------------+---------------------------+----------------------------------------+---------------------------+
    |                            | .. code-block:: matlab    | .. code-block:: python                 | .. code-block:: julia     |
    |                            |                           |                                        |                           |
    | Linearly spaced vector     |     A = linspace(1, 5, k) |  A = np.linspace(1, 5, k)              |     A = range(1, 5,       |
    | of k points                |                           |                                        |     length = k)           |
    +----------------------------+---------------------------+----------------------------------------+---------------------------+



Creating Matrices
-----------------

.. container:: multilang-table

    +--------------------------------+--------------------------+----------------------------------+--------------------------+
    | Operation                      |         MATLAB           | Python                           | Julia                    |
    +================================+==========================+==================================+==========================+
    |                                | .. code-block:: matlab   | .. code-block:: python           | .. code-block:: julia    |
    |                                |                          |                                  |                          |
    | Create a matrix                |     A = [1 2; 3 4]       |   A = np.array([[1, 2], [3, 4]]) |     A = [1 2; 3 4]       |
    +--------------------------------+--------------------------+----------------------------------+--------------------------+
    |                                | .. code-block:: matlab   | .. code-block:: python           | .. code-block:: julia    |
    |                                |                          |                                  |                          |
    | 2 x 2 matrix of zeros          |     A = zeros(2, 2)      |   A = np.zeros((2, 2))           |     A = zeros(2, 2)      |
    +--------------------------------+--------------------------+----------------------------------+--------------------------+
    |                                | .. code-block:: matlab   | .. code-block:: python           | .. code-block:: julia    |
    |                                |                          |                                  |                          |
    | 2 x 2 matrix of ones           |     A = ones(2, 2)       |   A = np.ones((2, 2))            |     A = ones(2, 2)       |
    +--------------------------------+--------------------------+----------------------------------+--------------------------+
    |                                | .. code-block:: matlab   | .. code-block:: python           | .. code-block:: julia    |
    |                                |                          |                                  |                          |
    | 2 x 2 identity matrix          |     A = eye(2, 2)        |   A = np.eye(2)                  |     A = I # will adopt   |
    |                                |                          |                                  | # 2x2 dims if demanded by|
    |                                |                          |                                  | # neighboring matrices   |
    +--------------------------------+--------------------------+----------------------------------+--------------------------+
    |                                | .. code-block:: matlab   | .. code-block:: python           | .. code-block:: julia    |
    |                                |                          |                                  |                          |
    | Diagonal matrix                |     A = diag([1 2 3])    |   A = np.diag([1, 2, 3])         |    A = Diagonal([1, 2,   |
    |                                |                          |                                  |        3])               |
    +--------------------------------+--------------------------+----------------------------------+--------------------------+
    |                                | .. code-block:: matlab   | .. code-block:: python           | .. code-block:: julia    |
    |                                |                          |                                  |                          |
    | Uniform random numbers         |     A = rand(2, 2)       |   A = np.random.rand(2, 2)       |     A = rand(2, 2)       |
    +--------------------------------+--------------------------+----------------------------------+--------------------------+
    |                                | .. code-block:: matlab   | .. code-block:: python           | .. code-block:: julia    |
    |                                |                          |                                  |                          |
    | Normal random numbers          |     A = randn(2, 2)      |   A = np.random.randn(2, 2)      |     A = randn(2, 2)      |
    +--------------------------------+--------------------------+----------------------------------+--------------------------+
    |                                | .. code-block:: matlab   | .. code-block:: python           | .. code-block:: julia    |
    |                                |                          |                                  |                          | 
    | Sparse Matrices                |                          |                                  |     using SparseArrays   | 
    |                                |                          |                                  |     A = spzeros(2, 2)    | 
    |                                |                          |                                  |     A[1, 2] = 4          |
    |                                |                          |                                  |     A[2, 2] = 1          | 
    +--------------------------------+--------------------------+----------------------------------+--------------------------+
    |                                | .. code-block:: matlab   | .. code-block:: python           | .. code-block:: julia    |
    |                                |                          |                                  |                          |
    | Tridiagonal Matrices           |                          |                                  |     x = [1, 2, 3]        |
    |                                |                          |                                  |     y = [4, 5, 6, 7]     |
    |                                |                          |                                  |     z = [8, 9, 10]       |
    |                                |                          |                                  |     Tridiagonal(x, y, z) |
    +--------------------------------+--------------------------+----------------------------------+--------------------------+

Manipulating Vectors and Matrices
---------------------------------

.. container:: multilang-table

    +--------------------------------+-------------------------------+---------------------------+---------------------------+
    | Operation                      |         MATLAB                | Python                    | Julia                     |
    +================================+===============================+===========================+===========================+
    |                                | .. code-block:: matlab        | .. code-block:: python    | .. code-block:: julia     |
    |                                |                               |                           |                           |
    | Transpose                      |     A.'                       |   A.T                     |     transpose(A)          |
    +--------------------------------+-------------------------------+---------------------------+---------------------------+
    |                                | .. code-block:: matlab        | .. code-block:: python    | .. code-block:: julia     |
    | Complex conjugate transpose    |                               |                           |                           |
    | (Adjoint)                      |     A'                        |   A.conj()                |     A'                    |
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
    | Reshape (to 5 rows, 2 columns) |    A = reshape(1:10, 5, 2)    |    A = A.reshape(5, 2)    |    A = reshape(1:10, 5, 2)|
    +--------------------------------+-------------------------------+---------------------------+---------------------------+
    |                                | .. code-block:: matlab        | .. code-block:: python    | .. code-block:: julia     |
    |                                |                               |                           |                           |
    | Convert matrix to vector       |    A(:)                       |    A = A.flatten()        |    A[:]                   |
    +--------------------------------+-------------------------------+---------------------------+---------------------------+
    |                                | .. code-block:: matlab        | .. code-block:: python    | .. code-block:: julia     |
    |                                |                               |                           |                           |
    | Flip left/right                |    fliplr(A)                  |    np.fliplr(A)           |    reverse(A, dims = 2)   |
    +--------------------------------+-------------------------------+---------------------------+---------------------------+
    |                                | .. code-block:: matlab        | .. code-block:: python    | .. code-block:: julia     |
    |                                |                               |                           |                           |
    | Flip up/down                   |    flipud(A)                  |    np.flipud(A)           |    reverse(A, dims = 1)   |
    +--------------------------------+-------------------------------+---------------------------+---------------------------+
    |                                | .. code-block:: matlab        | .. code-block:: python    | .. code-block:: julia     |
    |                                |                               |                           |                           |
    | Repeat matrix (3 times in the  |    repmat(A, 3, 4)            |    np.tile(A, (4, 3))     |    repeat(A, 3, 4)        |
    | row dimension, 4 times in the  |                               |                           |                           |
    | column dimension)              |                               |                           |                           |
    +--------------------------------+-------------------------------+---------------------------+---------------------------+
    |                                |                               |                           |                           | 
    |                                |                               |                           |                           |
    | Preallocating/Similar          |                               |                           | .. code-block:: julia     | 
    |                                |                               |                           |                           | 
    |                                |                               |                           |     y = similar(x)        |
    |                                |                               |                           |     # new dims            |
    |                                |                               |                           |     y = similar(x, R, C)  | 
    +--------------------------------+-------------------------------+---------------------------+---------------------------+
    |                                |                               |                           |                           |
    |                                |                               |                           |                           |
    | Broadcast a function over a    |                               |                           | .. code-block:: julia     |
    | collection/matrix/vector       |                               |                           |                           |
    |                                |                               |                           |     f(x) = x^2            |
    |                                |                               |                           |     g(x, y) = x + 2 + y^2 |
    |                                |                               |                           |     x = 1:10              |
    |                                |                               |                           |     y = 2:11              |
    |                                |                               |                           |     f.(x)                 |
    |                                |                               |                           |     g.(x,y)               |
    +--------------------------------+-------------------------------+---------------------------+---------------------------+


Accessing Vector/Matrix Elements
--------------------------------

.. container:: multilang-table

    +--------------------------------+-------------------------------+-------------------------------+---------------------------+
    | Operation                      |         MATLAB                | Python                        | Julia                     |
    +================================+===============================+===============================+===========================+
    |                                | .. code-block:: matlab        | .. code-block:: python        | .. code-block:: julia     |
    |                                |                               |                               |                           |
    | Access one element             |     A(2, 2)                   |    A[1, 1]                    |     A[2, 2]               |
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

.. container:: multilang-table

    +--------------------------------+-------------------------------+--------------------------------+---------------------------+
    | Operation                      |         MATLAB                | Python                         | Julia                     |
    +================================+===============================+================================+===========================+
    |                                | .. code-block:: matlab        | .. code-block:: python3        | .. code-block:: julia     |
    |                                |                               |                                |                           |
    | Dot product                    |     dot(A, B)                 |    np.dot(A, B) or A @ B       |     dot(A, B)             |
    |                                |                               |                                |                           |
    |                                |                               |                                |     A ⋅ B # \cdot<TAB>    |
    +--------------------------------+-------------------------------+--------------------------------+---------------------------+
    |                                | .. code-block:: matlab        | .. code-block:: python3        | .. code-block:: julia     |
    |                                |                               |                                |                           |
    | Matrix multiplication          |     A * B                     |     A @ B                      |     A * B                 |
    +--------------------------------+-------------------------------+--------------------------------+---------------------------+
    | Inplace matrix multiplication  | .. code-block:: matlab        | .. code-block:: python         | .. code-block:: julia     |
    |                                |                               |                                |                           |
    |                                |                               |                                |     x = [1, 2]            |
    |                                |                               |                                |     A = [1 2; 3 4]        |
    |                                |                               |                                |     y = similar(x)        |
    |                                |                               |                                |     mul!(y, A, x)         |
    +--------------------------------+-------------------------------+--------------------------------+---------------------------+
    |                                | .. code-block:: matlab        | .. code-block:: python         | .. code-block:: julia     |
    |                                |                               |                                |                           |
    | Element-wise multiplication    |     A .* B                    |    A * B                       |     A .* B                |
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
    | Inverse                        |     inv(A)                    |    np.linalg.inv(A)            |     inv(A)                |
    |                                |                               |                                |                           |
    |                                | or                            |                                | or                        |
    |                                |                               |                                |                           |
    |                                | .. code-block:: matlab        |                                | .. code-block:: julia     |
    |                                |                               |                                |                           |
    |                                |     A^(-1)                    |                                |    A^(-1)                 |
    +--------------------------------+-------------------------------+--------------------------------+---------------------------+
    |                                | .. code-block:: matlab        | .. code-block:: python         | .. code-block:: julia     |
    |                                |                               |                                |                           |
    | Determinant                    |     det(A)                    |    np.linalg.det(A)            |     det(A)                |
    +--------------------------------+-------------------------------+--------------------------------+---------------------------+
    |                                | .. code-block:: matlab        | .. code-block:: python         | .. code-block:: julia     |
    |                                |                               |                                |                           |
    | Eigenvalues and eigenvectors   |     [vec, val] = eig(A)       |    val, vec = np.linalg.eig(A) |     val, vec = eigen(A    |
    +--------------------------------+-------------------------------+--------------------------------+---------------------------+
    |                                | .. code-block:: matlab        | .. code-block:: python         | .. code-block:: julia     |
    |                                |                               |                                |                           |
    | Euclidean norm                 |     norm(A)                   |    np.linalg.norm(A)           |     norm(A)               |
    +--------------------------------+-------------------------------+--------------------------------+---------------------------+
    |                                | .. code-block:: matlab        | .. code-block:: python         | .. code-block:: julia     |
    |                                |                               |                                |                           |
    | Solve linear system            |     A\b                       |    np.linalg.solve(A, b)       |     A\b                   |
    | :math:`Ax=b` (when :math:`A`   |                               |                                |                           |
    | is square)                     |                               |                                |                           |
    +--------------------------------+-------------------------------+--------------------------------+---------------------------+
    |                                | .. code-block:: matlab        | .. code-block:: python         | .. code-block:: julia     |
    |                                |                               |                                |                           |
    | Solve least squares problem    |     A\b                       |    np.linalg.lstsq(A, b)       |     A\b                   |
    | :math:`Ax=b` (when :math:`A`   |                               |                                |                           |
    | is rectangular)                |                               |                                |                           |
    +--------------------------------+-------------------------------+--------------------------------+---------------------------+


Sum / max / min
-------------------
    
.. container:: multilang-table

    +--------------------------------+-------------------------------+---------------------------------+---------------------------+
    | Operation                      |         MATLAB                | Python                          | Julia                     |
    +================================+===============================+=================================+===========================+
    |                                | .. code-block:: matlab        | .. code-block:: python          | .. code-block:: julia     |
    |                                |                               |                                 |                           |
    | Sum / max / min of             |     sum(A, 1)                 |    sum(A, 0)                    |     sum(A, dims = 1)      |
    | each column                    |     max(A, [], 1)             |    np.amax(A, 0)                |     maximum(A, dims = 1)  |
    |                                |     min(A, [], 1)             |    np.amin(A, 0)                |     minimum(A, dims = 1)  |
    +--------------------------------+-------------------------------+---------------------------------+---------------------------+
    |                                | .. code-block:: matlab        | .. code-block:: python          | .. code-block:: julia     |
    |                                |                               |                                 |                           |
    | Sum / max / min of each row    |     sum(A, 2)                 |    sum(A, 1)                    |     sum(A, 2)             |
    |                                |     max(A, [], 2)             |    np.amax(A, 1)                |     maximum(A, 2)         |
    |                                |     min(A, [], 2)             |    np.amin(A, 1)                |     minimum(A, 2)         |
    +--------------------------------+-------------------------------+---------------------------------+---------------------------+
    |                                | .. code-block:: matlab        | .. code-block:: python          | .. code-block:: julia     |
    |                                |                               |                                 |                           |
    | Sum / max / min of             |     sum(A(:))                 |    np.sum(A)                    |     sum(A)                |
    | entire matrix                  |     max(A(:))                 |    np.amax(A)                   |     maximum(A)            |
    |                                |     min(A(:))                 |    np.amin(A)                   |     minimum(A)            |
    +--------------------------------+-------------------------------+---------------------------------+---------------------------+
    |                                | .. code-block:: matlab        | .. code-block:: python          | .. code-block:: julia     |
    |                                |                               |                                 |                           |
    | Cumulative sum / max / min     |     cumsum(A, 1)              |    np.cumsum(A, 0)              |     cumsum(A, 1)          |
    | by row                         |     cummax(A, 1)              |    np.maximum.accumulate(A, 0)  |     cummax(A, 1)          |
    |                                |     cummin(A, 1)              |    np.minimum.accumulate(A, 0)  |     cummin(A, 1)          |
    +--------------------------------+-------------------------------+---------------------------------+---------------------------+
    |                                | .. code-block:: matlab        | .. code-block:: python          | .. code-block:: julia     |
    |                                |                               |                                 |                           |
    | Cumulative sum / max / min     |     cumsum(A, 2)              |    np.cumsum(A, 1)              |     cumsum(A, 2)          |
    | by column                      |     cummax(A, 2)              |    np.maximum.accumulate(A, 1)  |     cummax(A, 2)          |
    |                                |     cummin(A, 2)              |    np.minimum.accumulate(A, 1)  |     cummin(A, 2)          |
    +--------------------------------+-------------------------------+---------------------------------+---------------------------+



Programming
-----------

.. container:: multilang-table

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
    | For loop               |     for i = 1:N            |    for i in range(n):      |     for i in 1:N              |
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
    | If                     |     if i <= N              |    if i <= N:              |     if i <= N                 |
    |                        |        % do something      |       # do something       |        # do something         |
    |                        |     end                    |                            |     end                       |
    +------------------------+----------------------------+----------------------------+-------------------------------+
    |                        | .. code-block:: matlab     | .. code-block:: python     | .. code-block:: julia         |
    |                        |                            |                            |                               |
    | If / else              |     if i <= N              |   if i <= N:               |    if i <= N                  |
    |                        |        % do something      |       # do something       |       # do something          |
    |                        |     else                   |   else:                    |    else                       |
    |                        |        % do something else |       # so something else  |       # do something else     |
    |                        |     end                    |                            |    end                        |
    +------------------------+----------------------------+----------------------------+-------------------------------+
    |                        | .. code-block:: matlab     | .. code-block:: python     | .. code-block:: julia         |
    |                        |                            |                            |                               |
    | Print text and variable|     x = 10                 |   x = 10                   |    x = 10                     |
    |                        |     fprintf('x = %d \n', x)|   print(f'x = {x}')        |    println("x = $x")          |
    +------------------------+----------------------------+----------------------------+-------------------------------+
    |                        | .. code-block:: matlab     | .. code-block:: python     | .. code-block:: julia         |
    |                        |                            |                            |                               |
    | Function: anonymous    |     f = @(x) x^2           |    f = lambda x: x**2      |     f = x -> x^2              |
    |                        |                            |                            |     # can be rebound          |
    +------------------------+----------------------------+----------------------------+-------------------------------+
    |                        | .. code-block:: matlab     | .. code-block:: python     | .. code-block:: julia         |
    |                        |                            |                            |                               |
    | Function               |     function out  = f(x)   |    def f(x):               |     function f(x)             |
    |                        |        out = x^2           |        return x**2         |        return x^2             |
    |                        |     end                    |                            |     end                       |
    |                        |                            |                            |                               |
    |                        |                            |                            |     f(x) = x^2 # not anon!    |
    +------------------------+----------------------------+----------------------------+-------------------------------+
    |                        |                            |                            |                               | 
    |                        |                            |                            |                               | 
    | Tuples                 |     N/A                    | {PYTHON}                   | .. code-block:: julia         | 
    |                        |                            |                            |                               |
    |                        |                            |                            |    (1, 2, 3)                  |
    +------------------------+----------------------------+----------------------------+-------------------------------+
    |                        |                            |                            |                               |
    |                        |                            |                            |                               | 
    | Named Tuples/          |                            |                            | .. code-block:: julia         |
    | Anonymous Structures   |                            |                            |                               |
    |                        |                            |                            |    # vanilla                  |
    |                        |                            |                            |    m = (x = 1, y = 2)         |
    |                        |                            |                            |                               |
    |                        |                            |                            |    # constructor              |
    |                        |                            |                            |    using Parameters           |
    |                        |                            |                            |    func = @with_kw (x=1, y=2) |
    |                        |                            |                            |    m = func() # same as above |
    |                        |                            |                            |    m = func(x = 3)            | 
    +------------------------+----------------------------+----------------------------+-------------------------------+
    |                        |                            |                            |                               |
    |                        |                            |                            |                               |
    | Closures               |                            |                            | .. code-block:: julia         |
    |                        |                            |                            |                               |
    |                        |                            |                            |        a = 2.0                |
    |                        |                            |                            |        f(x) = a + x           |
    |                        |                            |                            |        f(1.0)                 |
    +------------------------+----------------------------+----------------------------+-------------------------------+
    |                        | .. code-block:: matlab     | .. code-block:: python     | .. code-block:: julia         |
    |                        |                            |                            |                               |
    | Inplace Modification   |                            |                            |    function f!(out, x)        |
    |                        |                            |                            |        out .= x.^2            |
    |                        |                            |                            |    end                        |
    +------------------------+----------------------------+----------------------------+-------------------------------+