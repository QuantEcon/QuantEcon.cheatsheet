.. The QuantEcon MATLAB-Python-Julia Cheat Sheet documentation master file, created by
   sphinx-quickstart on Thu Sep  1 18:39:43 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

The QuantEcon MATLAB-Python-Julia Cheat Sheet
=========================================================================

This document summarizes commonly-used, equivalent commands across MATLAB, Python, and Julia.



Creating Vectors
----------------

+-----------------------------+--------------------------+-------------------------+--------------------------+
| Operation                   |         MATLAB           | Python                  | Julia                    |
+=============================+==========================+=========================+==========================+
|                             | .. code-block:: matlab   | .. code-block:: python  | .. code-block:: julia    |
|                             |                          |                         |                          |
| Create a row vector         |     A = [1 2 3]          |    A = np.array([1 2 3])|     A = [1 2 3]          |
+-----------------------------+--------------------------+-------------------------+--------------------------+
|                             | .. code-block:: matlab   | .. code-block:: python  | .. code-block:: julia    |
|                             |                          |                         |                          |
| Create a column vector      |     A = [1; 2; 3]        |                         |     A = [1; 2; 3]        |
+-----------------------------+--------------------------+-------------------------+--------------------------+
|                             | .. code-block:: matlab   | .. code-block:: python  | .. code-block:: julia    |
|                             |                          |                         |                          |
| Sequence starting at j      |     A = j:k:n            |                         |     A = j:k:n            |
| ending at n, with           |                          |                         |                          |
| difference k between points |                          |                         |                          |
+-----------------------------+--------------------------+-------------------------+--------------------------+
|                             | .. code-block:: matlab   | .. code-block:: python  | .. code-block:: julia    |
|                             |                          |                         |                          |
| Linearly spaced vector      |     A = linspace(1, 5, k)|                         |     A = linspace(1, 5, k)|
| of k points                 |                          |                         |                          |
+-----------------------------+--------------------------+-------------------------+--------------------------+



Creating Matrices
-----------------

+--------------------------------+--------------------------+-------------------------+--------------------------+
| Operation                      |         MATLAB           | Python                  | Julia                    |
+================================+==========================+=========================+==========================+
|                                | .. code-block:: matlab   | .. code-block:: python  | .. code-block:: julia    |
|                                |                          |                         |                          |
| Create a matrix                |     A = [1 2; 3 4]       |                         |     A = [1 2; 3 4]       |
+--------------------------------+--------------------------+-------------------------+--------------------------+
|                                | .. code-block:: matlab   | .. code-block:: python  | .. code-block:: julia    |
|                                |                          |                         |                          |
| Create a 2 by 2 matrix of zeros|     A = zeros(2, 2)      |                         |     A = zeros(2, 2)      |
+--------------------------------+--------------------------+-------------------------+--------------------------+
|                                | .. code-block:: matlab   | .. code-block:: python  | .. code-block:: julia    |
|                                |                          |                         |                          |
| Create a 2 by 2 matrix of ones |     A = ones(2, 2)       |                         |     A = ones(2, 2)       |
+--------------------------------+--------------------------+-------------------------+--------------------------+
|                                | .. code-block:: matlab   | .. code-block:: python  | .. code-block:: julia    |
|                                |                          |                         |                          |
| Create a 2 by 2 identity matrix|     A = eye(2, 2)        |                         |     A = eye(2, 2)        |
+--------------------------------+--------------------------+-------------------------+--------------------------+
|                                | .. code-block:: matlab   | .. code-block:: python  | .. code-block:: julia    |
|                                |                          |                         |                          |
| Create a diagonal matrix       |     A = diag([1 2 3])    |                         |     A = diagm([1; 2; 3]) |
+--------------------------------+--------------------------+-------------------------+--------------------------+
|                                | .. code-block:: matlab   | .. code-block:: python  | .. code-block:: julia    |
|                                |                          |                         |                          |
| Matrix of uniformly distributed|     A = rand(2, 2)       |                         |     A = rand(2, 2)       |
| random numbers                 |                          |                         |                          |
+--------------------------------+--------------------------+-------------------------+--------------------------+
|                                | .. code-block:: matlab   | .. code-block:: python  | .. code-block:: julia    |
|                                |                          |                         |                          |
| Matrix of random numbers drawn |     A = randn(2, 2)      |                         |     A = randn(2, 2)      |
| a standard normal              |                          |                         |                          |
+--------------------------------+--------------------------+-------------------------+--------------------------+



Manipulating Vectors and Matrices
---------------------------------

+--------------------------------+-------------------------------+--------------------------+---------------------------+
| Operation                      |         MATLAB                | Python                   | Julia                     |
+================================+===============================+==========================+===========================+
|                                | .. code-block:: matlab        | .. code-block:: python   | .. code-block:: julia     |
|                                |                               |                          |                           |
| Transpose                      |     A'                        |                          |     A'                    |
+--------------------------------+-------------------------------+--------------------------+---------------------------+
|                                | .. code-block:: matlab        | .. code-block:: python   | .. code-block:: julia     |
|                                |                               |                          |                           |
| Concatenate horizontally       |     A = [[1 2] [1 2]]         |                          |     A = [[1 2] [1 2]]     |
|                                |                               |                          |                           |
|                                | or                            |                          | or                        |
|                                |                               |                          |                           |
|                                | .. code-block:: matlab        | .. code-block:: python   | .. code-block:: julia     |
|                                |                               |                          |                           |
|                                |     A = horzcat([1 2], [1 2]) |                          |    A = hcat([1 2], [1 2]) |
+--------------------------------+-------------------------------+--------------------------+---------------------------+
|                                | .. code-block:: matlab        | .. code-block:: python   | .. code-block:: julia     |
|                                |                               |                          |                           |
| Concatenate vertically         |     A = [[1 2]; [1 2]]        |                          |     A = [[1 2]; [1 2]]    |
|                                |                               |                          |                           |
|                                | or                            |                          | or                        |
|                                |                               |                          |                           |
|                                | .. code-block:: matlab        | .. code-block:: python   | .. code-block:: julia     |
|                                |                               |                          |                           |
|                                |     A = vertcat([1 2], [1 2]) |                          |    A = vcat([1 2], [1 2]) |
+--------------------------------+-------------------------------+--------------------------+---------------------------+
|                                | .. code-block:: matlab        | .. code-block:: python   | .. code-block:: julia     |
|                                |                               |                          |                           |
| Reshape (to 5 rows, 2 columns) |    A = reshape(1:10, 5, 2)    |                          |    A = reshape(1:10, 5, 2)|
+--------------------------------+-------------------------------+--------------------------+---------------------------+
|                                | .. code-block:: matlab        | .. code-block:: python   | .. code-block:: julia     |
|                                |                               |                          |                           |
| Convert matrix to vector       |    A(:)                       |                          |    A[:]                   |
+--------------------------------+-------------------------------+--------------------------+---------------------------+
|                                | .. code-block:: matlab        | .. code-block:: python   | .. code-block:: julia     |
|                                |                               |                          |                           |
| Flip left/right                |    fliplr(A)                  |                          |    flipdim(A, 2)          |
+--------------------------------+-------------------------------+--------------------------+---------------------------+
|                                | .. code-block:: matlab        | .. code-block:: python   | .. code-block:: julia     |
|                                |                               |                          |                           |
| Flip up/down                   |    flipud(A)                  |                          |    flipdim(A, 1)          |
+--------------------------------+-------------------------------+--------------------------+---------------------------+
|                                | .. code-block:: matlab        | .. code-block:: python   | .. code-block:: julia     |
|                                |                               |                          |                           |
| Repeat matrix (3 times in the  |    repmat(A, 3, 4)            |                          |    repmat(A, 3, 4)        |
| row dimension, 4 times in the  |                               |                          |                           |
| column dimension)              |                               |                          |                           |
+--------------------------------+-------------------------------+--------------------------+---------------------------+



Accessing Vector/Matrix Elements
--------------------------------

+--------------------------------+-------------------------------+--------------------------+---------------------------+
| Operation                      |         MATLAB                | Python                   | Julia                     |
+================================+===============================+==========================+===========================+
|                                | .. code-block:: matlab        | .. code-block:: python   | .. code-block:: julia     |
|                                |                               |                          |                           |
| Access one element             |     A(2, 2)                   |                          |     A[2, 2]               |
+--------------------------------+-------------------------------+--------------------------+---------------------------+
|                                | .. code-block:: matlab        | .. code-block:: python   | .. code-block:: julia     |
|                                |                               |                          |                           |
| Access specific rows           |    A(1:4, :)                  |                          |    A[1:4, :]              |
+--------------------------------+-------------------------------+--------------------------+---------------------------+
|                                | .. code-block:: matlab        | .. code-block:: python   | .. code-block:: julia     |
|                                |                               |                          |                           |
| Access specific columns        |    A(:, 1:4)                  |                          |    A[:, 1:4]              |
+--------------------------------+-------------------------------+--------------------------+---------------------------+
|                                | .. code-block:: matlab        | .. code-block:: python   | .. code-block:: julia     |
|                                |                               |                          |                           |
| Remove a row                   |    A([1 2 4], :)              |                          |    A[[1, 2, 4], :]        |
+--------------------------------+-------------------------------+--------------------------+---------------------------+
|                                | .. code-block:: matlab        | .. code-block:: python   | .. code-block:: julia     |
|                                |                               |                          |                           |
| Diagonals of matrix            |    diag(A)                    |                          |    diag(A)                |
+--------------------------------+-------------------------------+--------------------------+---------------------------+
|                                | .. code-block:: matlab        | .. code-block:: python   | .. code-block:: julia     |
|                                |                               |                          |                           |
| Get dimensions of matrix       |    [nrow ncol] = size(A)      |                          |    nrow, ncol = size(A)   |
+--------------------------------+-------------------------------+--------------------------+---------------------------+



Mathematical Operations
-----------------------

+--------------------------------+-------------------------------+--------------------------+---------------------------+
| Operation                      |         MATLAB                | Python                   | Julia                     |
+================================+===============================+==========================+===========================+
|                                | .. code-block:: matlab        | .. code-block:: python   | .. code-block:: julia     |
|                                |                               |                          |                           |
| Vector dot product             |     dot(A, B)                 |                          |     dot(A, B)             |
+--------------------------------+-------------------------------+--------------------------+---------------------------+
|                                | .. code-block:: matlab        | .. code-block:: python   | .. code-block:: julia     |
|                                |                               |                          |                           |
| Matrix multiplication          |     A*B                       |                          |     A*B                   |
+--------------------------------+-------------------------------+--------------------------+---------------------------+
|                                | .. code-block:: matlab        | .. code-block:: python   | .. code-block:: julia     |
|                                |                               |                          |                           |
| Element-wise matrix            |     A.*B                      |                          |     A.*B                  |
| multiplication                 |                               |                          |                           |
+--------------------------------+-------------------------------+--------------------------+---------------------------+
|                                | .. code-block:: matlab        | .. code-block:: python   | .. code-block:: julia     |
|                                |                               |                          |                           |
| Matrix to a power              |     A^2                       |                          |     A^2                   |
+--------------------------------+-------------------------------+--------------------------+---------------------------+
|                                | .. code-block:: matlab        | .. code-block:: python   | .. code-block:: julia     |
|                                |                               |                          |                           |
| Matrix to a power, elementwise |     A.^2                      |                          |     A.^2                  |
+--------------------------------+-------------------------------+--------------------------+---------------------------+
|                                | .. code-block:: matlab        | .. code-block:: python   | .. code-block:: julia     |
|                                |                               |                          |                           |
| Inverse of a matrix            |     inv(A)                    |                          |     inv(A)                |
|                                |                               |                          |                           |
|                                | or                            |                          | or                        |
|                                |                               |                          |                           |
|                                | .. code-block:: matlab        | .. code-block:: python   | .. code-block:: julia     |
|                                |                               |                          |                           |
|                                |     A^(-1)                    |                          |    A^(-1)                 |
+--------------------------------+-------------------------------+--------------------------+---------------------------+
|                                | .. code-block:: matlab        | .. code-block:: python   | .. code-block:: julia     |
|                                |                               |                          |                           |
| Determinant of a matrix        |     det(A)                    |                          |     det(A)                |
+--------------------------------+-------------------------------+--------------------------+---------------------------+
|                                | .. code-block:: matlab        | .. code-block:: python   | .. code-block:: julia     |
|                                |                               |                          |                           |
| Eigenvalues and eigenvectors   |     [vec, val] = eig(A)       |                          |     val, vec = eig(A)     |
+--------------------------------+-------------------------------+--------------------------+---------------------------+
|                                | .. code-block:: matlab        | .. code-block:: python   | .. code-block:: julia     |
|                                |                               |                          |                           |
| Euclidean norm                 |     norm(A)                   |                          |     norm(A)               |
+--------------------------------+-------------------------------+--------------------------+---------------------------+
|                                | .. code-block:: matlab        | .. code-block:: python   | .. code-block:: julia     |
|                                |                               |                          |                           |
| Solve linear system            |     A\b                       |                          |     A\b                   |
| :math:`Ax=b`                   |                               |                          |                           |
+--------------------------------+-------------------------------+--------------------------+---------------------------+



Sum/Maximum/Minimum
-------------------

+--------------------------------+-------------------------------+--------------------------+---------------------------+
| Operation                      |         MATLAB                | Python                   | Julia                     |
+================================+===============================+==========================+===========================+
|                                | .. code-block:: matlab        | .. code-block:: python   | .. code-block:: julia     |
|                                |                               |                          |                           |
| Sum/maximum/minimum of         |     sum(A, 1)                 |                          |     sum(A, 1)             |
| each column                    |     max(A, [], 1)             |                          |     maximum(A, 1)         |
|                                |     min(A, [], 1)             |                          |     minimum(A, 1)         |
+--------------------------------+-------------------------------+--------------------------+---------------------------+
|                                | .. code-block:: matlab        | .. code-block:: python   | .. code-block:: julia     |
|                                |                               |                          |                           |
| Sum/maximum/minimum of         |     sum(A, 2)                 |                          |     sum(A, 2)             |
| each row                       |     max(A, [], 2)             |                          |     maximum(A, 2)         |
|                                |     min(A, [], 2)             |                          |     minimum(A, 2)         |
+--------------------------------+-------------------------------+--------------------------+---------------------------+
|                                | .. code-block:: matlab        | .. code-block:: python   | .. code-block:: julia     |
|                                |                               |                          |                           |
| Sum/maximum/minimum of         |     sum(A(:))                 |                          |     sum(A)                |
| entire matrix                  |     max(A(:))                 |                          |     maximum(A)            |
|                                |     min(A(:))                 |                          |     minimum(A)            |
+--------------------------------+-------------------------------+--------------------------+---------------------------+
|                                | .. code-block:: matlab        | .. code-block:: python   | .. code-block:: julia     |
|                                |                               |                          |                           |
| Cumulative sum/maximum/minimum |     cumsum(A, 1)              |                          |     cumsum(A, 1)          |
| by row                         |     cummax(A, 1)              |                          |     cummax(A, 1)          |
|                                |     cummin(A, 1)              |                          |     cummin(A, 1)          |
+--------------------------------+-------------------------------+--------------------------+---------------------------+
|                                | .. code-block:: matlab        | .. code-block:: python   | .. code-block:: julia     |
|                                |                               |                          |                           |
| Cumulative sum/maximum/minimum |     cumsum(A, 2)              |                          |     cumsum(A, 2)          |
| by column                      |     cummax(A, 2)              |                          |     cummax(A, 2)          |
|                                |     cummin(A, 2)              |                          |     cummin(A, 2)          |
+--------------------------------+-------------------------------+--------------------------+---------------------------+



Programming
-----------

+------------------------+----------------------------+------------------------+-------------------------------+
| Operation              |         MATLAB             | Python                 | Julia                         |
+========================+============================+========================+===============================+
|                        | .. code-block:: matlab     | .. code-block:: python | .. code-block:: julia         |
|                        |                            |                        |                               |
| Comment one line       |     % This is a comment    |                        |     # This is a comment       |
+------------------------+----------------------------+------------------------+-------------------------------+
|                        | .. code-block:: matlab     | .. code-block:: python | .. code-block:: julia         |
|                        |                            |                        |                               |
| Comment block          |     %{                     |                        |     #=                        |
|                        |     Comment block          |                        |     Comment block             |
|                        |     %}                     |                        |     =#                        |
+------------------------+----------------------------+------------------------+-------------------------------+
|                        | .. code-block:: matlab     | .. code-block:: python | .. code-block:: julia         |
|                        |                            |                        |                               |
| For loop               |     for i = 1:N            |                        |     for i = 1:N               |
|                        |        % do something      |                        |        # do something         |
|                        |     end                    |                        |     end                       |
+------------------------+----------------------------+------------------------+-------------------------------+
|                        | .. code-block:: matlab     | .. code-block:: python | .. code-block:: julia         |
|                        |                            |                        |                               |
| While loop             |     while i <= N           |                        |     while i <= N              |
|                        |        % do something      |                        |        # do something         |
|                        |     end                    |                        |     end                       |
+------------------------+----------------------------+------------------------+-------------------------------+
|                        | .. code-block:: matlab     | .. code-block:: python | .. code-block:: julia         |
|                        |                            |                        |                               |
| If statement           |     if i <= N              |                        |     if i <= N                 |
|                        |        % do something      |                        |        # do something         |
|                        |     end                    |                        |     end                       |
+------------------------+----------------------------+------------------------+-------------------------------+
|                        | .. code-block:: matlab     | .. code-block:: python | .. code-block:: julia         |
|                        |                            |                        |                               |
| If/else statement      |     if i <= N              |                        |    if i <= N                  |
|                        |        % do something      |                        |       # do something          |
|                        |     else                   |                        |    else                       |
|                        |        % do something else |                        |       # do something else     |
|                        |     end                    |                        |    end                        |
+------------------------+----------------------------+------------------------+-------------------------------+
|                        | .. code-block:: matlab     | .. code-block:: python | .. code-block:: julia         |
|                        |                            |                        |                               |
| Print text and variable|     x = 10                 |                        |    x = 10                     |
| to screen              |     fprintf('The value of  |                        |    println("The value of      |
|                        |     x is %d. \n', x)       |                        |    x is $(x).")               |
+------------------------+----------------------------+------------------------+-------------------------------+
|                        | .. code-block:: matlab     | .. code-block:: python | .. code-block:: julia         |
|                        |                            |                        |                               |
| Function: one line/    |     fun = @(x) x^2         |                        |     fun(x) = x^2              |
| anonymous              |                            |                        |                               |
+------------------------+----------------------------+------------------------+-------------------------------+
|                        | .. code-block:: matlab     | .. code-block:: python | .. code-block:: julia         |
|                        |                            |                        |                               |
| Function: multiple     |     function out  = fun(x) |                        |     function fun(x)           |
| lines                  |        out = x^2           |                        |        return x^2             |
|                        |     end                    |                        |     end                       |
+------------------------+----------------------------+------------------------+-------------------------------+




