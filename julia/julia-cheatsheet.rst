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