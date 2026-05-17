import numpy as np
# import matplotlib.pyplot as plt

def my_cross(u, v):
    # Levi-Civita symbol
    epsilon = np.zeros((3, 3, 3))
    epsilon[0, 1, 2] = 1
    epsilon[0, 2, 1] = -1
    epsilon[1, 0, 2] = -1
    epsilon[1, 2, 0] = 1
    epsilon[2, 0, 1] = 1
    epsilon[2, 1, 0] = -1
    return np.einsum("ijk,j,k->i", epsilon, u, v)

def solution1():
    """ Altenative to
    x = np.array([7, 8, 9])
    W = np.array([[1, 2, 3], [4, 5, 6]])
    
    """
    x = np.arange(7, 10, 1)
    W = np.arange(1, 7, 1).reshape((2, 3))
    """ Altenative to
    [[ 1.  0. -1.]
    [ 4.  3.  2.]]
    """
    W = np.concatenate((np.arange(1, -2, 1),np.arange(4, 1, -1)), axis=0)
    # problem 3
    u = np.array([2,3,4])
    v = np.array([1,2,5])
    
    print(my_cross(u, v))
    print(np.cross(u, v))

def main():
    solution1()

if __name__ == "__main__":
    main()