.. _julia-cheatsheet:

QuantEcon Julia Cheat Sheet
===========================

Variables
---------

Here are a few examples of basic kinds of variables we might be interested in creating.

+---------------------------+---------------------------------------------------------------------------------------------------------------+
| Command                   |                                             Description                                                       |
+===========================+===============================================================================================================+
| .. code-block:: julia     | How to **create a scalar, a vector, or a matrix**. Here, each example will result in a slightly different form|
|                           | of output. ``A`` is a scalar, ``B`` is a flat array with 3 elements, ``C`` is a 1 by 3 vector, ``D`` is a 3 by|
|    A = 4.1                | 1 vector, and ``E`` is a 2 by 2 matrix.                                                                       |
|    B = [1, 2, 3]          |                                                                                                               |
|    C = [1.1 2.2 3.3]      |                                                                                                               |
|    D = [1 2 3]'           |                                                                                                               |
|    E = [1 2; 3 4]         |                                                                                                               |
+---------------------------+---------------------------------------------------------------------------------------------------------------+
| .. code-block:: julia     | A **string** variable                                                                                         |
|                           |                                                                                                               |
|    s = "This is a string" |                                                                                                               |
+---------------------------+---------------------------------------------------------------------------------------------------------------+
| .. code-block:: julia     | A **Boolean** variable                                                                                        |
|                           |                                                                                                               |
|    x = true               |                                                                                                               |
+---------------------------+---------------------------------------------------------------------------------------------------------------+  

Vectors and Matrices
--------------------
These are a few kinds of special vectors/matrices we can create and some things we can do with them.

+---------------------------+--------------------------------------------------------------------------------------------------------+
| Command                   |      Description                                                                                       |
+===========================+========================================================================================================+
| .. code-block:: julia     | Creates a **matrix of all zeros** of size ``m`` by ``n``. We can also do the following:                |
|                           |  .. code-block:: julia                                                                                 |
|    A = zeros(m, n)        |                                                                                                        |
|                           |     A = zeros(B)                                                                                       |
|                           | which will create a matrix of all zeros with the same dimensions as matrix or vector ``B``.            | 
+---------------------------+--------------------------------------------------------------------------------------------------------+
| .. code-block:: julia     | Creates a **matrix of all ones** of size ``m`` by ``n``. We can also do the following:                 |
|                           |  .. code-block:: julia                                                                                 |
|    A = ones(m, n)         |                                                                                                        |
|                           |     A = ones(B)                                                                                        |
|                           | which will create a matrix of all ones with the same dimensions as matrix or vector ``B``.             | 
+---------------------------+--------------------------------------------------------------------------------------------------------+
| .. code-block:: julia     | Creates an ``n`` by ``n`` **identity matrix**. For example, ``eye(3)`` will return                     |
|                           |  .. math::                                                                                             |
|    A = eye(n)             |                                                                                                        |
|                           |     \begin{pmatrix}                                                                                    |
|                           |     1 & 0 & 0\\                                                                                        |
|                           |     0 & 1 & 0\\                                                                                        |
|                           |     0 & 0 & 1                                                                                          |
|                           |     \end{pmatrix}                                                                                      |
+---------------------------+--------------------------------------------------------------------------------------------------------+
| .. code-block:: julia     | This will create a **sequence** starting at ``j``, ending at ``n``, with difference                    |
|                           | ``k`` between points. For example, ``A = 2:4:10`` will create the sequence ``2, 6, 10``                |
|    A = j:k:n              | To convert the output to an array, use ``collect(A)``.                                                 |
+---------------------------+--------------------------------------------------------------------------------------------------------+
| .. code-block:: julia     | This will create a **sequence** of ``m`` points starting at ``j``, ending at ``n``. For example,       |
|                           | ``A = linspace(2, 10, 3)`` will create the sequence ``2.0, 6.0, 10.0``. To convert the output to an    |
|    A = linspace(j, n, m)  | array, use ``collect(A)``.                                                                             |         
+---------------------------+--------------------------------------------------------------------------------------------------------+
| .. code-block:: julia     | Creates a **diagonal matrix** using the elements in ``x``.  For example if ``x = [1, 2, 3]``,          |
|                           |  ``diagm(x)`` will return                                                                              |
|    A = diagm(x)           |  .. math::                                                                                             |
|                           |                                                                                                        |
|                           |     \begin{pmatrix}                                                                                    |
|                           |     1 & 0 & 0\\                                                                                        |
|                           |     0 & 2 & 0\\                                                                                        |
|                           |     0 & 0 & 3                                                                                          |
|                           |     \end{pmatrix}                                                                                      |
+---------------------------+--------------------------------------------------------------------------------------------------------+
| .. code-block:: julia     | Creates an ``m`` by ``n`` **matrix of random numbers** drawn from a **uniform distribution** on        |
|                           | :math:`[0, 1]`. Alternatively, ``rand`` can be used to draw random elements from a set ``X``. For      |
|    A = rand(m, n)         | example, if ``X = [1, 2, 3]``, ``rand(X)`` will return either ``1``, ``2``, or ``3``.                  |    
+---------------------------+--------------------------------------------------------------------------------------------------------+
| .. code-block:: julia     | Creates an ``m`` by ``n`` **matrix of random numbers** drawn from a **standard normal distribution**.  |
|                           |                                                                                                        |
|    A = randn(m, n)        |                                                                                                        |
+---------------------------+--------------------------------------------------------------------------------------------------------+
| .. code-block:: julia     | This is the general syntax for **accessing elements** of an array or matrix, where ``m`` and ``n`` are |
|                           | integers. The example here returns the element in the second row and third column.                     |
|                           |                                                                                                        |
|    A[m, n]                | * We can also use ranges (like ``1:3``) in place of single numbers to extract multiple rows or columns |
|                           |                                                                                                        |
|                           | * A colon, ``:``, by itself indicates all rows or columns                                              |
|                           |                                                                                                        |
|                           | * The word ``end`` can also be used to indicate the last row or column                                 |
+---------------------------+--------------------------------------------------------------------------------------------------------+
| .. code-block:: julia     | **Returns the number of rows and columns** in a matrix. Alternatively, we can do                       |
|                           |  .. code-block:: julia                                                                                 |
|    nrow, ncol = size(A)   |                                                                                                        |
|                           |    nrow = size(A, 1)                                                                                   |
|                           |                                                                                                        |
|                           | and                                                                                                    |
|                           |  .. code-block:: julia                                                                                 |
|                           |                                                                                                        |
|                           |     ncol = size(A, 2)                                                                                  |
|                           |                                                                                                        |
+---------------------------+--------------------------------------------------------------------------------------------------------+
| .. code-block:: julia     | This function returns a vector of the **diagonal elements** of ``A``                                   |
|                           | (i.e., ``A[1, 1], A[2, 2]``, etc...).                                                                  |
|    diag(A)                |                                                                                                        |
+---------------------------+--------------------------------------------------------------------------------------------------------+
| .. code-block:: julia     | **Horizontally concatenates** two matrices or vectors. The example here would return                   |
|                           |  .. math::                                                                                             |
|    A = hcat([1 2], [3 4]) |                                                                                                        |
|                           |     \begin{pmatrix}                                                                                    |
|                           |     1 & 2 & 3 & 4                                                                                      |
|                           |     \end{pmatrix}                                                                                      |
|                           |                                                                                                        |
|                           | An alternative syntax is:                                                                              |
|                           |  .. code-block:: julia                                                                                 |
|                           |                                                                                                        |
|                           |     A = [[1 2] [3 4]]                                                                                  |
|                           |                                                                                                        |
|                           | For either of these commands to work, both matrices or vectors must have the same number of rows.      |
+---------------------------+--------------------------------------------------------------------------------------------------------+
| .. code-block:: julia     | **Vertically concatenates** two matrices or vectors. The example here would return                     |
|                           |  .. math::                                                                                             |
|    A = vcat([1 2], [3 4]) |                                                                                                        |
|                           |     \begin{pmatrix}                                                                                    |
|                           |     1 & 2 \\                                                                                           |
|                           |     3 & 4                                                                                              |
|                           |     \end{pmatrix}                                                                                      |
|                           |                                                                                                        |
|                           | An alternative syntax is:                                                                              |
|                           |  .. code-block:: julia                                                                                 |
|                           |                                                                                                        |
|                           |     A = [[1 2]; [3 4]]                                                                                 |
|                           |                                                                                                        |
|                           | For either of these commands to work, both matrices or vectors must have the same number of columns.   |
+---------------------------+--------------------------------------------------------------------------------------------------------+
| .. code-block:: julia     | **Reshapes** matrix or vector ``a`` into a new matrix or vector, ``A`` with ``m`` rows                 |
|                           | and ``n`` columns. For example ``A = reshape(1:10, 5, 2)`` would return                                |
|                           |                                                                                                        |
|    A = reshape(a, m, n)   |  .. math::                                                                                             |
|                           |                                                                                                        |
|                           |    \begin{pmatrix}                                                                                     |
|                           |    1 & 6 \\                                                                                            |
|                           |    2 & 7 \\                                                                                            |
|                           |    3 & 8 \\                                                                                            |
|                           |    4 & 9 \\                                                                                            |
|                           |    5 & 10                                                                                              |
|                           |    \end{pmatrix}                                                                                       |
|                           |                                                                                                        |
|                           | For this to work, the number  of elements in ``a`` (number of rows times number of columns) must       |
|                           | equal ``m * n``.                                                                                       |
+---------------------------+--------------------------------------------------------------------------------------------------------+
| .. code-block:: julia     | **Converts matrix A to a vector.** For example, if ``A = [1 2; 3 4]``, then ``A[:]`` will return       |
|                           |                                                                                                        |
|    A[:]                   |                                                                                                        |
|                           |  .. math::                                                                                             |
|                           |                                                                                                        |
|                           |    \begin{pmatrix}                                                                                     |
|                           |    1 \\                                                                                                |
|                           |    2 \\                                                                                                |
|                           |    3 \\                                                                                                |
|                           |    4                                                                                                   |
|                           |    \end{pmatrix}                                                                                       |
+---------------------------+--------------------------------------------------------------------------------------------------------+
| .. code-block:: julia     | Reverses the vector or matrix ``A`` along dimension ``d``. For example, if ``A = [1 2 3; 4 5 6]``,     |
|                           |  ``flipdim(A, 1)}``, will reverse the rows of ``A`` and return                                         |
|    flipdim(A, d)          |  .. math::                                                                                             |
|                           |                                                                                                        |
|                           |     \begin{pmatrix}                                                                                    |
|                           |     4 & 5 & 6 \\                                                                                       |
|                           |     1 & 2 & 3                                                                                          |
|                           |     \end{pmatrix}                                                                                      |
|                           |                                                                                                        |
|                           |  ``flipdim(A, 2)`` will reverse the columns of ``A`` and return                                        |
|                           |  .. math::                                                                                             |
|                           |                                                                                                        |
|                           |     \begin{pmatrix}                                                                                    |
|                           |     3 & 2 & 1 \\                                                                                       |
|                           |     6 & 5 & 4                                                                                          |
|                           |     \end{pmatrix}                                                                                      |
+---------------------------+--------------------------------------------------------------------------------------------------------+
| .. code-block:: julia     | **Repeats matrix** ``A``, ``m`` times in the row direction and ``n`` in the column direction.          |
|                           | For example, if ``A = [1 2; 3 4]``, ``repmat(A, 2, 3)`` will return                                    |
|    repmat(A, m, n)        |  .. math::                                                                                             |
|                           |                                                                                                        |
|                           |     \begin{pmatrix}                                                                                    |
|                           |     1 & 2 & 1 & 2 & 1 & 2 \\                                                                           |
|                           |     3 & 4 & 3 & 4 & 3 & 4 \\                                                                           |
|                           |     1 & 2 & 1 & 2 & 1 & 2 \\                                                                           |
|                           |     3 & 4 & 3 & 4 & 3 & 4                                                                              |
|                           |     \end{pmatrix}                                                                                      |
|                           |                                                                                                        |
+---------------------------+--------------------------------------------------------------------------------------------------------+


