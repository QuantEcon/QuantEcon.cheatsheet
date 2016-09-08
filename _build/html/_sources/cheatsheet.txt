.. The QuantEcon MATLAB-Python-Julia Cheat Sheet documentation master file, created by
   sphinx-quickstart on Thu Sep  1 18:39:43 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

The QuantEcon MATLAB-Python-Julia Cheat Sheet
=========================================================================

This document summarizes commonly-used, equivalent commands across MATLAB, Python, and Julia



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
|                             | .. code-block:: matlab   | .. code-block:: python  | .. code-block:: julia    |
|                             |                          |                         |                          |
| Vector dot product          |     dot(A, B)            |                         |     dot(A, B)            |
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



Manipulating Vectors and Matrices
---------------------------------

+--------------------------------+-------------------------------+--------------------------+--------------------------+
| Operation                      |         MATLAB                | Python                   | Julia                    |
+================================+===============================+==========================+==========================+
|                                | .. code-block:: matlab        | .. code-block:: python   | .. code-block:: julia    |
|                                |                               |                          |                          |
| Transpose                      |     A'                        |                          |     A'                   |
+--------------------------------+-------------------------------+--------------------------+--------------------------+
|                                | .. code-block:: matlab        | .. code-block:: python   | .. code-block:: julia    |
|                                |                               |                          |                          |
| Concatenate horizontally       |     A = [[1 2] [1 2]]         |                          |     A = [[1 2] [1 2]]    |
|                                | .. code-block:: matlab        | .. code-block:: python   | .. code-block:: julia    |
|                                |                               |                          |                          |
|                                |     A = horzcat([1 2], [1 2]) |                          |    A = hcat([1 2], [1 2])|
+--------------------------------+-------------------------------+--------------------------+--------------------------+
|                                | .. code-block:: matlab        | .. code-block:: python   | .. code-block:: julia    |
|                                |                               |                          |                          |
| Concatenate vertically         |     A = [[1 2]; [1 2]]        |                          |     A = [[1 2]; [1 2]]   |
|                                | .. code-block:: matlab        | .. code-block:: python   | .. code-block:: julia    |
|                                |                               |                          |                          |
|                                |     A = vertcat([1 2], [1 2]) |                          |    A = vcat([1 2], [1 2])|
+--------------------------------+-------------------------------+--------------------------+--------------------------+


Input and Output
----------------

+------------------------+------------------------+------------------------+-----------------------+
| Operation              |         MATLAB         | Python                 | Julia                 |
+========================+========================+========================+=======================+
|                        | .. code-block:: matlab | .. code-block:: python | .. code-block:: julia |
|                        |                        |                        |                       |
| Opening a file         |     fopen('file')      |    open('file')        |     open('file')      |
+------------------------+------------------------+------------------------+-----------------------+
