def simulated_annealing(f, x0, T0=1., C=10., sigma=1., eps=1e-6, 
                        burn=20, max_count=1000, verbose=False):
    """
    Simulaled annealing algorithm.
     
    Simulaled annealing with exponential temperature decay. 
    A metaheuristic algorithm commonly used for optimization 
    problems of black-box functions with large search spaces. 

    Parameters
    ----------
    f: function
        User-supplied external function which describes the 
        so-called energy function of the system being optimized.
    x0: np.array
        Initial values for the parameters of the energy function.
    T0: float, default=1.
        Initial temperature value.
    C: float, default=10.
        Constant decay value of the temperature scheduling.
        Temperature after k-th iteration is determined from 
        the following relation: Tk = T0*exp(-k/C).
    sigma: float, default=1.
        Standard deviation of the Normal distribution for the
        random walk by which new candidates are generated. 
        Namely, new candidates (i.e. x-values) are obtained from:
        x_new = x + N(0,sigma), where N(0,sigma) are the random
        numbers from the Normal distribution with zero mean and
        standard deviation of `sigma`.
    eps: float, default=1e-6
        Temperature value at which the algorithm is stopped.
    burn: int, default=20
        Number of iterations with the original step size of
        the random walk from N(0,sigma), after which the step
        size is reduced by a factor of 10, i.e. N(0,0.1*sigma).
    max_count: int, default=1000
        Maximum loop counter for stopping the iterations. It is
        assumed that there is no convergence if the main loop
        exceeds this counter number and exception is raised.
    verbose: bool, default=False
        Indicator for printing (on stdout) internal messages.

    Returns
    -------
    x: np.array
        Coordinate values (i.e. parameters) of the energy 
        function's optimum point.
    E: float
        Energy function's optimal value.
    
    Raises
    ------
    StopIteration
        Exit the main loop after the `max_count` counter is 
        exceeded. This signals that the algorithm is not
        converging.
    
    Important
    ---------
    There is a relationship in the cooling schedule between the
    initial temperature, decay value and stopping temperature,
    which is closely coorelated with the burn iterations and
    maximum counter value. The user should plot the cooling
    schedule that he/she intends to use and only then decide on
    the concrete `eps`, `burn` and `sigma` parameter values.
    Defaults are provided for orientation only, and may not be
    suited for all applications. As a rule of thumb, the `burn`
    parameter can be set near the iteration number around which
    the cooling schedule curve exhibits a knee-point.
    
    Notes
    -----
    Algorithm consists of four parts: (1) energy function that is
    being optimized, (2) perturbation function that is used for
    generating candidate solutions, (3) acceptance criterion, and 
    (4) temperature schedule (i.e. cooling).
    Perturbation function is a random walk in multi-dimensional
    space, with random samples drawn from the Normal distribution 
    N(0,sigma). Step size of the walk is reduced after the burn-in.
    Acceptance criterion is according to the Boltzman probability
    and Metropolis algorithm. Temperature schedule is exponential
    cooling.
    """
    from numpy import exp
    from scipy import stats

    # Dimension of the search space.
    N = x0.size
    # Initial values.
    x = x0
    T = T0
    # Initial energy.
    E = f(*x)
    
    # Initialize best values.
    x_top = x
    E_top = E
    
    k = 0
    while T > eps:
        # Generate coordinates from random walk.
        if k < burn:
            # Original step size during burn-in.
            walk = stats.norm(scale=sigma).rvs(N)
        else:
            # Reduce step size after temperature cools down.
            walk = stats.norm(scale=0.1*sigma).rvs(N)
        x_new = x + walk

        # Compute energy function at new coords.
        E_new = f(*x_new)
        # Energy difference.
        Delta_E = E_new - E

        # Metropolis acceptance criterion.
        if Delta_E < 0.:
            x = x_new
            E = E_new
        else:
            r = stats.uniform.rvs()
            # Boltzman probability.
            alpha = exp(-Delta_E/T)
            # Stochastic acceptance.
            if r < alpha:
                x = x_new
                E = E_new
        
        # Save the best solution.
        if E_new < E_top:
            x_top = x_new
            E_top = E_new

        # Temperature schedule.
        T = T0*exp(-k/C)

        k += 1

        # Early stopping.
        if abs(E_new) < eps:
            print(f'Early stopping after {k} iterations.')
            break

        # Loop counter exceeded without convergence.
        if k > max_count:
            raise StopIteration(
                f'No convergence after {max_count} iterations.')
    
    if verbose:
        print(f'Final temperature: {T:.3e} after {k} iterations.')
    
    print('Optimization successful.')

    return x_top, E_top


if __name__ == "__main__":
    """ Test with a simple function. """
    import numpy as np

    def f(x, y):
        y = (x-1)**2 + (y+1)**2
        return y

    x0 = np.array([4, 4])
    x, E = simulated_annealing(f, x0)
    print(f'Coordinates: {x[0]:.4f}, {x[1]:.4f}')
    print(f'Energy func.: {E:.4e}')