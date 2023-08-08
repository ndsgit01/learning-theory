import numpy as np
import itertools


def rad_set(A):
    """
    Compute the Rademacher complexity of a finite set 
    A \\subseteq \\doubleR^m and |A| = n

    Parameters
    ----------
    A: numpy array
       set represented as a numpy array of shape (m, n)

    Returns
    -------
    rad: float
         Rademacher complexity of set A
    """

    m = A.shape[0]
    all_sigma = itertools.product((1, -1), repeat=m)
    rad = 0
    for sigma in all_sigma:
        rad += np.max(np.array(sigma) @ A)
    prob_sigma = 1 / 2 ** m
    rad *= prob_sigma
    rad /= m

    return rad


def func_composition(F, S):
    """
    Compute the composition of a function class F on sample S

    Parameters
    ----------
    F: iterable
       function class where |F| = n
    S: numpy array
       m sample points of dimension n represented as a numpy
       array of shape (m, *)
    Returns
    -------
    FoS: numpy array
         function composition {(f(S)) | f \\in F} represented as
         a numpy array of shape (m, n)
    """

    FoS = np.vstack(
        tuple(
            np.apply_along_axis(f, 1, S) for f in F
        )
    ).T

    return FoS


def rad_func_class(F, S):
    """
    Compute the empirical Rademacher complexity of a function class 
    F given S

    Parameters
    ----------
    F: iterable
       class of functions where |F| = n
    S: numpy array
       m sample points of dimension n represented as a numpy
       array of shape (m, *)

    Returns
    -------
    rad: float
         Rademacher complexity of F given S
    """

    rad = rad_set(func_composition(F, S))

    return rad


if __name__ == "__main__":
    # Find Rademacher complexity of set A
    A_ = np.array([[1, 0, 1, -1], [0, -1, 1, 0]])
    print(rad_set(A_) == 5 / 8)

    # Find Rademacher complexity of a function class
    def loss_always_positive(s):
        return 1 == s[-1]

    def loss_always_negative(s):
        return -1 == s[-1]

    F_ = (loss_always_positive, loss_always_negative)
    S_ = np.array([[0, 1],
                  [1, -1]])

    print(rad_func_class(F_, S_) == 1/4)
