from annealing import simulated_annealing
from numpy import testing, array

def test_annealing():
    def f(x, y):
        y = (x-1)**2 + (y+1)**2
        return y
    
    res = simulated_annealing(f, [4, 4])

    s = array([1, -1])
    
    testing.assert_array_almost_equal(res['x'], s, decimal=1)