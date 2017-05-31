.. raw:: html

	<style type="text/css">.menu>li.python-on>a {border-color:#444;cursor: default;}</style>

.. _python-cheatsheet:

.. role:: python(code)
   :language: python

Python cheatsheet
=================

Operators
---------

.. container:: singlelang-table python-table

    +---------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
    | Command                         | Description                                                                                                                                        |
    +=================================+====================================================================================================================================================+
    | :python:`*`                     | multiplication operation: :python:`2*3` returns ``6``                                                                                              |
    +---------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
    | :python:`**`                    | power operation: :python:`2**3` returns ``8``                                                                                                      |
    +---------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
    | :python:`@`                     | matrix multiplication:                                                                                                                             |
    |                                 |                                                                                                                                                    |
    |                                 | .. code-block:: python3                                                                                                                            |
    |                                 |                                                                                                                                                    |
    |                                 |     import numpy as np                                                                                                                             |
    |                                 |     A = np.array([[1,2,3]])                                                                                                                        |
    |                                 |     B = np.array([[3],[2],[1]])                                                                                                                    |
    |                                 |     A @ B                                                                                                                                          |
    |                                 |                                                                                                                                                    |
    |                                 | returns                                                                                                                                            |
    |                                 |                                                                                                                                                    |
    |                                 | .. code-block:: python                                                                                                                             |
    |                                 |                                                                                                                                                    |
    |                                 |     array([[10]])                                                                                                                                  |
    |                                 |                                                                                                                                                    |
    +---------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+

Data Types
----------------------

.. container:: singlelang-table python-table

    +---------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
    | Command                         | Description                                                                                                                                        |
    +=================================+====================================================================================================================================================+
    | :python:`l = [a1,a2,...,an]`    | Constructs a list containing the objects :math:`a1,a2,...,an`.  You can append to the list using :python:`l.append()`.                             |
    |                                 | The :math:`ith` element of :math:`l` can be accessed using :python:`l[i]`                                                                          |
    +---------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
    | :python:`t =(a1,a2,...,an)`     | Constructs a tuple containing the objects :math:`a1,a2,...,an`.  The :math:`ith` element of :math:`t` can be accessed using :python:`t[i]`         |
    +---------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+

Built-In Functions
----------------------

.. container:: singlelang-table python-table

    +---------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
    | Command                         | Description                                                                                                                                        |
    +=================================+====================================================================================================================================================+
    | :python:`len(iterable)`         | :python:`len` is a function that takes an iterable, such as a list, tuple or numpy array and returns the number of items in that object.           |
    |                                 | For a numpy array, :python:`len` returns the length of the outermost dimension                                                                     |
    |                                 |                                                                                                                                                    |
    |                                 | .. code-block:: python                                                                                                                             |
    |                                 |                                                                                                                                                    |
    |                                 |     len(np.zeros((5,4)))                                                                                                                           |
    |                                 |                                                                                                                                                    |
    |                                 | returns ``5``.                                                                                                                                     |
    +---------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
    | :python:`zip`                   | Make an iterator that aggregates elements from each of the iterables.                                                                              |
    |                                 |                                                                                                                                                    |
    |                                 | .. code-block:: python                                                                                                                             |
    |                                 |                                                                                                                                                    |
    |                                 |     x = [1, 2, 3]                                                                                                                                  |
    |                                 |     y = [4, 5, 6]                                                                                                                                  |
    |                                 |     zipped = zip(x, y)                                                                                                                             |
    |                                 |     list(zipped)                                                                                                                                   |
    |                                 |                                                                                                                                                    |
    |                                 | returns :python:`[(1, 4), (2, 5), (3, 6)]`                                                                                                         |
    +---------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+

Iterating
----------------------

.. container:: singlelang-table python-table

    +---------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
    | Command                         | Description                                                                                                                                        |
    +=================================+====================================================================================================================================================+
    | :python:`for a in iterable:`    | For loop used to perform a sequence of commands (denoted using tabs) for each element in an iterable object such as a list, tuple, or numpy array. |
    |                                 | An example code is                                                                                                                                 |
    |                                 |                                                                                                                                                    |
    |                                 | .. code-block:: python                                                                                                                             |
    |                                 |                                                                                                                                                    |
    |                                 |     l  = []                                                                                                                                        |
    |                                 |     for i in [1,2,3]:                                                                                                                              |
    |                                 |         l.append(i**2)                                                                                                                             |
    |                                 |     print(l)                                                                                                                                       |
    |                                 |                                                                                                                                                    |
    |                                 | prints :python:`[1,4,9]`                                                                                                                           |
    |                                 |                                                                                                                                                    |
    +---------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+

Comparisons and Logical Operators
---------------------------------

.. container:: singlelang-table python-table

    +---------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
    | Command                         | Description                                                                                                                                        |
    +=================================+====================================================================================================================================================+
    | :python:`if condition:`         | Performs code if a condition is met (using tabs). For example                                                                                      |
    |                                 |                                                                                                                                                    |
    |                                 | .. code-block:: python                                                                                                                             |
    |                                 |                                                                                                                                                    |
    |                                 |     if x == 5:                                                                                                                                     |
    |                                 |         x = x**2                                                                                                                                   |
    |                                 |     else:                                                                                                                                          |
    |                                 |         x = x**3                                                                                                                                   |
    |                                 |                                                                                                                                                    |
    |                                 | squares :math:`x` if :math:`x` is :math:`5`, otherwise cubes it.                                                                                   |
    +---------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+

User-Defined Functions
----------------------

.. container:: singlelang-table python-table

    +---------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
    | Command                         | Description                                                                                                                                        |
    +=================================+====================================================================================================================================================+
    | :python:`lambda`                | Used for create anonymous one line functions of the form:                                                                                          |
    |                                 |                                                                                                                                                    |
    |                                 | .. code-block:: python                                                                                                                             |
    |                                 |                                                                                                                                                    |
    |                                 |     f = lambda x,y: 5*x+y                                                                                                                          |
    |                                 |                                                                                                                                                    |
    |                                 | The code after the lambda but before variables specifies the parameters. The code after the colon tells python what object to return.              |
    +---------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
    | :python:`def`                   | The def command is used to create functions of more than one line:                                                                                 |
    |                                 |                                                                                                                                                    |
    |                                 | .. code-block:: python                                                                                                                             |
    |                                 |                                                                                                                                                    |
    |                                 |     def g(x,y):                                                                                                                                    |
    |                                 |         """                                                                                                                                        |
    |                                 |         Docstring                                                                                                                                  |
    |                                 |         """                                                                                                                                        |
    |                                 |         ret = sin(x)                                                                                                                               |
    |                                 |         return ret + y                                                                                                                             |
    |                                 |                                                                                                                                                    |
    |                                 | The code immediately following :python:`def` names the function, in this example ``g`` .                                                           |
    |                                 | The variables in the parenthesis are the parameters of the function.  The remaining lines of the function are denoted by tab indents.              |
    |                                 | The return statement specifies the object to be returned.                                                                                          |
    +---------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+

Numpy
------------

.. container:: singlelang-table python-table

    +---------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------+
    | Command                                     | Description                                                                                                                                           |
    +=============================================+=======================================================================================================================================================+
    | :python:`np.array(object,dtype = None)`     | :python:`np.array` constructs a numpy array from an object, such as a list or a list of lists.                                                        |
    |                                             | :python:`dtype` allows you to specify the type of object the array is holding.                                                                        |
    |                                             | You will generally note need to specify the :python:`dtype`.                                                                                          |
    |                                             | Examples:                                                                                                                                             |
    |                                             |                                                                                                                                                       |
    |                                             | .. code-block:: python                                                                                                                                |
    |                                             |                                                                                                                                                       |
    |                                             |     np.array([1, 2, 3]) #creates 1 dim array of ints                                                                                                  |
    |                                             |     np.array( [1,2,3.0] )#creates 1 dim array of floats                                                                                               |
    |                                             |     np.array( [ [1,2],[3,4] ]) #creates a 2 dim array                                                                                                 |
    |                                             |                                                                                                                                                       |
    +---------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------+
    | :python:`A[i1,i2,...,in]`                   | Access a the element in numpy array A in with index i1 in dimension 1, i2 in dimension 2, etc.                                                        |
    |                                             | Can use ``:`` to access a range of indices, where ``imin:imax`` represents all :math:`i` such that :math:`imin \leq i < imax`.                        |
    |                                             | Always returns an object of minimal dimension.                                                                                                        |
    |                                             | For example,                                                                                                                                          |
    |                                             |                                                                                                                                                       |
    |                                             | :python:`A[:,2]`                                                                                                                                      |
    |                                             |                                                                                                                                                       |
    |                                             | returns the 2nd column (counting from 0) of A as a 1 dimensional array and                                                                            |
    |                                             |                                                                                                                                                       |
    |                                             | :python:`A[0:2,:]`                                                                                                                                    |
    |                                             |                                                                                                                                                       |
    |                                             | returns the 0th and 1st rows in a 2 dimensional array.                                                                                                |
    +---------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------+
    | :python:`np.zeros(shape)`                   | Constructs numpy array of shape shape.  Here shape is an integer of sequence of integers.  Such as 3, (1,2),(2,1), or (5,5).  Thus                    |
    |                                             |                                                                                                                                                       |
    |                                             | :python:`np.zeros((5,5))`                                                                                                                             |
    |                                             |                                                                                                                                                       |
    |                                             | Constructs an :math:`5\times 5` array while                                                                                                           |
    |                                             |                                                                                                                                                       |
    |                                             | :python:`np.zeros(5,5)`                                                                                                                               |
    |                                             |                                                                                                                                                       |
    |                                             | will throw an error.                                                                                                                                  |
    +---------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------+
    | :python:`np.ones(shape)`                    | Same as :python:`np.zeros` but produces an array of ones                                                                                              |
    +---------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------+
    | :python:`np.linspace(a,b,n)`                | Returns a numpy array with :math:`n` linearly spaced points between :math:`a` and :math:`b`.  For example                                             |
    |                                             |                                                                                                                                                       |
    |                                             | :python:`np.linspace(1,2,10)`                                                                                                                         |
    |                                             |                                                                                                                                                       |
    |                                             | returns                                                                                                                                               |
    |                                             |                                                                                                                                                       |
    |                                             | .. code-block:: python                                                                                                                                |
    |                                             |                                                                                                                                                       |
    |                                             |     array([ 1.        ,  1.11111111,  1.22222222,  1.33333333,                                                                                        |
    |                                             |     1.44444444, 1.55555556,  1.66666667,  1.77777778,                                                                                                 |
    |                                             |     1.88888889,  2.        ])                                                                                                                         |
    |                                             |                                                                                                                                                       |
    +---------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------+
    | :python:`np.eye(N)`                         | Constructs the identity matrix of size :math:`N`.  For example                                                                                        |
    |                                             |                                                                                                                                                       |
    |                                             | :python:`np.eye(3)`                                                                                                                                   |
    |                                             |                                                                                                                                                       |
    |                                             | returns the :math:`3\times 3` identity matrix:                                                                                                        |
    |                                             |                                                                                                                                                       |
    |                                             | .. math::                                                                                                                                             |
    |                                             |                                                                                                                                                       |
    |                                             |     \left(\begin{matrix}1&0&0\\0&1&0\\ 0&0&1\end{matrix}\right)                                                                                       |
    |                                             |                                                                                                                                                       |
    +---------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------+
    | :python:`np.diag(a)`                        | :python:`np.diag` has 2 uses.  First if :python:`a` is a 2 dimensional array then :python:`np.diag` returns the principle diagonal of the matrix.     |
    |                                             | Thus                                                                                                                                                  |
    |                                             |                                                                                                                                                       |
    |                                             | :python:`np.diag( [ [1,3], [5,6] ])`                                                                                                                  |
    |                                             |                                                                                                                                                       |
    |                                             | returns :python:`[1,6]`.                                                                                                                              |
    |                                             |                                                                                                                                                       |
    |                                             | If :math:`a` is a 1 dimensional array then :python:`np.diag` constructs an array with $a$ as the principle diagonal.  Thus,                           |
    |                                             |                                                                                                                                                       |
    |                                             | :python:`np.diag([1,2])`                                                                                                                              |
    |                                             |                                                                                                                                                       |
    |                                             | returns                                                                                                                                               |
    |                                             |                                                                                                                                                       |
    |                                             | .. math::                                                                                                                                             |
    |                                             |                                                                                                                                                       |
    |                                             |     \left(\begin{matrix}1&0\\0&2\end{matrix}\right)                                                                                                   |
    |                                             |                                                                                                                                                       |
    +---------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------+
    | :python:`np.random.rand(d0, d1, ..., dn)`   | Constructs a numpy array of shape :python:`(d0,d1,...,dn)` filled with random numbers drawn from a uniform distribution between :math`(0,1)`.         |
    |                                             | For example, :python:`np.random.rand(2,3)` returns                                                                                                    |
    |                                             |                                                                                                                                                       |
    |                                             | .. code-block:: python                                                                                                                                |
    |                                             |                                                                                                                                                       |
    |                                             |     array([[ 0.69060674,  0.38943021,  0.19128955],                                                                                                   |
    |                                             |     [ 0.5419038 ,  0.66963507,  0.78687237]])                                                                                                         |
    |                                             |                                                                                                                                                       |
    +---------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------+
    | :python:`np.random.randn(d0, d1, ..., dn)`  | Same as :python:`np.random.rand(d0, d1, ..., dn)` except that it draws from the standard normal distribution :math:`\mathcal N(0,1)`                  |
    |                                             | rather than the uniform distribution.                                                                                                                 |
    +---------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------+
    | :python:`A.T`                               | Reverses the dimensions of an array (transpose).                                                                                                      |
    |                                             | For example,                                                                                                                                          |
    |                                             | if :math:`x = \left(\begin{matrix} 1& 2\\3&4\end{matrix}\right)` then :python:`x.T` returns :math:`\left(\begin{matrix} 1& 3\\2&4\end{matrix}\right)` |
    +---------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------+
    | :python:`np.hstack(tuple)`                  | Take a sequence of arrays and stack them horizontally to make a single array.  For example                                                            |
    |                                             |                                                                                                                                                       |
    |                                             | .. code-block:: python                                                                                                                                |
    |                                             |                                                                                                                                                       |
    |                                             |     a = np.array(( [1,2,3] )                                                                                                                          |
    |                                             |     b = np.array( [2,3,4] )                                                                                                                           |
    |                                             |     np.hstack( (a,b) )                                                                                                                                |
    |                                             |                                                                                                                                                       |
    |                                             | returns :python:`[1,2,3,2,3,4]` while                                                                                                                 |
    |                                             |                                                                                                                                                       |
    |                                             | .. code-block:: python                                                                                                                                |
    |                                             |                                                                                                                                                       |
    |                                             |     a = np.array( [[1],[2],[3]] )                                                                                                                     |
    |                                             |     b = np.array( [[2],[3],[4]] )                                                                                                                     |
    |                                             |     np.hstack((a,b))                                                                                                                                  |
    |                                             |                                                                                                                                                       |
    |                                             | returns :math:`\left( \begin{matrix} 1&2\\2&3\\ 3&4 \end{matrix}\right)`                                                                              |
    +---------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------+
    | :python:`np.vstack(tuple)`                  | Like :python:`np.hstack`.  Takes a sequence of arrays and stack them vertically to make a single array.  For example                                  |
    |                                             |                                                                                                                                                       |
    |                                             | .. code-block:: python                                                                                                                                |
    |                                             |                                                                                                                                                       |
    |                                             |     a = np.array( [1,2,3] )                                                                                                                           |
    |                                             |     b = np.array( [2,3,4] )                                                                                                                           |
    |                                             |     np.hstack( (a,b) )                                                                                                                                |
    |                                             |                                                                                                                                                       |
    |                                             | returns                                                                                                                                               |
    |                                             |                                                                                                                                                       |
    |                                             | .. code-block:: python                                                                                                                                |
    |                                             |                                                                                                                                                       |
    |                                             |     array( [ [1,2,3],                                                                                                                                 |
    |                                             |     [2,3,4] ] )                                                                                                                                       |
    |                                             |                                                                                                                                                       |
    +---------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------+
    | :python:`np.amax(a, axis = None)`           | By default :python:`np.amax(a)` finds the maximum of all elements in the array :math:`a`.                                                             |
    |                                             | Can specify maximization along a particular dimension with axis.                                                                                      |
    |                                             | If                                                                                                                                                    |
    |                                             |                                                                                                                                                       |
    |                                             | :python:`a = np.array( [ [2,1], [3,4] ]) #creates a 2 dim array`                                                                                      |
    |                                             |                                                                                                                                                       |
    |                                             | then                                                                                                                                                  |
    |                                             |                                                                                                                                                       |
    |                                             | :python:`np.amax(a,axis = 0) #maximization along row (dim 0)`                                                                                         |
    |                                             |                                                                                                                                                       |
    |                                             | returns :python:`array([3,4])`  and                                                                                                                   |
    |                                             |                                                                                                                                                       |
    |                                             | :python:`np.amax(a, axis = 1) #maximization along column (dim 1)`                                                                                     |
    |                                             |                                                                                                                                                       |
    |                                             | returns :python:`array([2,4])`                                                                                                                        |
    |                                             |                                                                                                                                                       |
    +---------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------+
    | :python:`np.amin(a, axis = None)`           | Same as :python:`np.amax` except returns minimum element.                                                                                             |
    +---------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------+
    | :python:`np.argmax(a, axis = None)`         | Performs similar function to np.amax except returns index of maximal element.                                                                         |
    |                                             | By default gives index of flattened array, otherwise can use axis to specify dimension.                                                               |
    |                                             | From the example for np.amax                                                                                                                          |
    |                                             |                                                                                                                                                       |
    |                                             | .. code-block:: python                                                                                                                                |
    |                                             |                                                                                                                                                       | 
    |                                             |     np.amax(a,axis = 0) #maximization along row (dim 0)                                                                                               |
    |                                             |                                                                                                                                                       |
    |                                             | returns :python:`array([1,1])` and                                                                                                                    |
    |                                             |                                                                                                                                                       |
    |                                             | .. code-block:: python                                                                                                                                |
    |                                             |                                                                                                                                                       |
    |                                             |     np.amax(a, axis = 1) #maximization along column (dim 1)                                                                                           |
    |                                             |                                                                                                                                                       |
    |                                             | returns :python:`array([0,1])`                                                                                                                        |
    +---------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------+
    | :python:`np.argmin(a, axis =None)`          | Same as :python:`np.argmax` except finds minimal index.                                                                                               |
    +---------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------+
    | :python:`np.dot(a,b)` or :python:`a.dot(b)` | Returns an array equal to the dot product of :math:`a` and :math:`b`.                                                                                 |
    |                                             | For this operation to work the innermost dimension of :math:`a` must be equal to the outermost dimension of :math:`b`.                                |
    |                                             | If :math:`a` is a :math:`(3,2)` array and :math:`b` is a :math:`(2)` array then :python:`np.dot(a,b)` is valid.                                       |
    |                                             | If :math:`b` is a :math:`(1,2)` array then the operation will return an error.                                                                        |
    +---------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------+


numpy.linalg 
-------------

.. container:: singlelang-table python-table

    +--------------------------------+----------------------------------------------------------------------------------------------------------------------------------+
    | Command                        | Description                                                                                                                      |
    +================================+==================================================================================================================================+
    | :python:`np.linalg.inv(A)`     | For a 2-dimensional array :math:`A`. :python:`np.linalg.inv` returns the inverse of :math:`A`.                                   |
    |                                | For example, for a :math:`(2,2)` array :math:`A`                                                                                 |
    |                                |                                                                                                                                  |
    |                                | .. code-block:: python                                                                                                           |
    |                                |                                                                                                                                  |
    |                                |      np.linalg.inv(A).dot(A)                                                                                                     |
    |                                |                                                                                                                                  |
    |                                | returns                                                                                                                          |
    |                                |                                                                                                                                  |
    |                                | .. code-block:: python                                                                                                           |
    |                                |                                                                                                                                  |
    |                                |      np.array( [1,0],                                                                                                            |
    |                                |      [0,1] ])                                                                                                                    |
    |                                |                                                                                                                                  |
    +--------------------------------+----------------------------------------------------------------------------------------------------------------------------------+
    | :python:`np.linalg.eig(A)`     | Returns a 1-dimensional array with all the eigenvalues of $A$ as well as a 2-dimensional array with the eigenvectors as columns. |
    |                                | For example,                                                                                                                     |
    |                                |                                                                                                                                  |
    |                                | :python:`eigvals,eigvecs = np.linalg.eig(A)`                                                                                     |
    |                                |                                                                                                                                  |
    |                                | returns the eigenvalues in :python:`eigvals` and the eigenvectors in :python:`eigvecs`.                                          |
    |                                | :python:`eigvecs[:,i]` is the eigenvector of :math:`A`  with eigenvalue of :python:`eigval[i]`.                                  |
    +--------------------------------+----------------------------------------------------------------------------------------------------------------------------------+
    | :python:`np.linalg.solve(A,b)` | Constructs array :math:`x` such that :python:`A.dot(x)` is equal to :math:`b`.  Theoretically should give the same answer as     |
    |                                |                                                                                                                                  |
    |                                | .. code-block:: python                                                                                                           |
    |                                |                                                                                                                                  |
    |                                |      Ainv = np.linalg.inv(A)                                                                                                     |
    |                                |      x = Ainv.dot(b)                                                                                                             |
    |                                |                                                                                                                                  |
    |                                | but numerically more stable.                                                                                                     |
    +--------------------------------+----------------------------------------------------------------------------------------------------------------------------------+
    
Pandas
------

.. container:: singlelang-table python-table

    +----------------+-----------------------------------------------------------------------------------------------+
    | Command        | Description                                                                                   |
    +================+===============================================================================================+
    | pd.Series()    | Constructs a Pandas Series Object from some specified data and/or index                       |
    |                |                                                                                               |
    |                | .. code-block:: python                                                                        |
    |                |                                                                                               |
    |                |      s1 = pd.Series([1,2,3])                                                                  |
    |                |      s2 = pd.Series([1,2,3], index=['a','b','c'])                                             |
    |                |                                                                                               |
    +----------------+-----------------------------------------------------------------------------------------------+
    | pd.DataFrame() | Constructs a Pandas DataFrame object from some specified data and/or index, column names etc. |
    |                |                                                                                               |
    |                | .. code-block:: python                                                                        |
    |                |                                                                                               |
    |                |      d = {'a' : [1,2,3], 'b' : [4,5,6]}                                                       |
    |                |      df = pd.DataFrame(d)                                                                     |
    |                |                                                                                               |
    |                | or alternatively,                                                                             |
    |                |                                                                                               |
    |                | .. code-block:: python                                                                        |
    |                |                                                                                               |
    |                |      a = [1,2,3]                                                                              |
    |                |      b = [4,5,6]                                                                              |
    |                |      df = pd.DataFrame(list(zip(a,b)), columns=['a','b'])                                     |
    |                |                                                                                               |
    +----------------+-----------------------------------------------------------------------------------------------+

Plotting
---------------------------------

.. container:: singlelang-table python-table

    +---------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
    | Command                         | Description                                                                                                                                        |
    +=================================+====================================================================================================================================================+
    | :python:`plt.plot(x,y,s =None)` | The plot command is included in :python:`matplotlib.pyplot`.                                                                                       |
    |                                 | The plot command is used to plot :math:`x` versus :math:`y` where :math:`x` and :math:`y` are iterables of the same length.                        |
    |                                 | By default the plot command draws a line, using the :math:`s` argument you can specify type of line and color.                                     |
    |                                 | For example '-','- -',':','o','x', and '-o' reprent line, dashed line, dotted line, circles, x's, and circle with line through it respectively.    |
    |                                 | Color can be changed by appending 'b','k','g' or 'r', to get a blue, black, green or red plot respectively.                                        |
    |                                 | For example,                                                                                                                                       |
    |                                 |                                                                                                                                                    |
    |                                 | .. code-block:: python                                                                                                                             |
    |                                 |                                                                                                                                                    |
    |                                 |     import numpy as np                                                                                                                             |
    |                                 |     import matplotlib.pyplot as plt                                                                                                                |
    |                                 |     x=np.linspace(0,10,100)                                                                                                                        |
    |                                 |     N=len(x)                                                                                                                                       |
    |                                 |     v= np.cos(x)                                                                                                                                   |
    |                                 |     plt.figure(1)                                                                                                                                  |
    |                                 |     plt.plot(x,v,'-og')                                                                                                                            |
    |                                 |     plt.show()                                                                                                                                     |
    |                                 |     plt.savefig('tom_test.eps')                                                                                                                    |
    |                                 |                                                                                                                                                    |
    |                                 | plots the cosine function on the domain (0,10) with a green line with circles at the points :math:`x,v`                                            |
    +---------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
    
