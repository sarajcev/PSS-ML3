def simulated_annealing(f, x0, T0=1., C=10., sigma=1., eps=1e-8, 
                        burn=20, fs=0.5, verbose=False):
    """
    Simulaled annealing algorithm.
     
    Simulaled annealing with exponential temperature decay. 
    A metaheuristic algorithm commonly used for optimization 
    problems of black-box functions (or systems) with large 
    search spaces.

    Parameters
    ----------
    f: function
        User-supplied external function f(x) which describes the 
        so-called energy function of the system being optimized,
        where x is a vector. This can also be a black-box func-
        tion or even a simulation process that takes inputs and 
        returns a state of the system.
    x0: np.array
        Initial values for the parameters of the energy function.
    T0: float, default=1.
        Initial temperature value.
    C: float, default=10.
        Constant decay value of the temperature scheduling.
        Temperature after k-th iteration is determined from 
        the following relation: Tk = T0*exp(-k/C).
    sigma: float, default=1.
        Standard deviation of a statistical distribution for
        the random walk by which new candidates are generated. 
        Random numbers are initially drawn from the Student's
        t-distribution with a low degree of freedom, zero mean 
        and standard deviation of `sigma`. After the burn-in 
        period, random numbers are drawn from the Normal 
        distribution with much lower standard deviation, i.e.
        N(0, fs*sigma), where `fs` is the factor by which
        `sigma` is reduced.
    eps: float, default=1e-6
        Temperature value at which the algorithm is stopped.
    burn: int, default=20
        Number of iterations with the original step size of
        the random walk from the Student's t distribution, after
        which the step size is reduced approximately by a factor
        of 10 by switching over to the Normal distribution with
        a lower standard deviation (see parameter `sigma` above).
    fs: float, default=0.1
        Factor for reducing the standard deviation of a statisti-
        cal distributin used for the random walk (see parameter
        `sigma` above).
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
    space, with random samples drawn from the Student's t distri-
    bution with low degrees of freedom. Step size of the walk is
    reduced after the burn-in period by switching to the Normal
    distribution with a lower standard deviation. Acceptance cri-
    terion is according to the Boltzman probability and Metropolis
    algorithm. Temperature schedule is exponential cooling.
    """
    from numpy import exp, pi
    from scipy import stats

    # Testing input parameters.
    if T0 <= 0.:
        raise ValueError('Initial temperature must be positive.')
    if fs <= 0. or fs > 1.:
        raise ValueError('Parameter "fs" must be between 0 and 1.')

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
    while T >= eps:
        # Generate coordinates for the random walk.
        if k < burn:
            # Original step size during the burn-in, from
            # the Student's t distribution with a low degree
            # of freedom.
            walk = stats.t(df=10, loc=0, scale=sigma).rvs(N)
        else:
            # Reduce the step size after temperature cools down
            # by switching to the Normal distribution with a 
            # lower standard deviation.
            walk = stats.norm(loc=0, scale=sigma*fs).rvs(N)

        # Random walk.
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
            alpha = (2*pi*T)**(-N/2) * exp(-abs(Delta_E)**2/(2*T))
            # Stochastic acceptance.
            if r < alpha:
                x = x_new
                E = E_new

        # Temperature schedule (cooling).
        T = T0*exp(-k/C)

        # Save the best solution found thus far.
        if E < E_top:
            x_top = x_new
            E_top = E_new

        k += 1
    
    if verbose:
        print('Final temperature: {:.3e} after {:d} iterations.'
              .format(T, k))
    
    print('Optimization successful.')

    return x_top, E_top


if __name__ == "__main__":
    """ Test with simple benchmark functions. """
    import numpy as np

    def rosenbrock(x, y):
        # Rosenbrock's function.
        # Minimum is at [1., 1.].
        f = (1. - x)**2 + 100.*(y - x**2)**2
        return f

    def beale(x, y):
        # Beale's function.
        # Minimum is at [3, 0.5].
        f = (1.5 - x + x*y)**2 + (2.25 - x + x*y**2) + (2.625 - x + x*y**3)**2
        return f

    def booth(x, y):
        # Booth's function.
        # minimum is at [1, 3].
        f = (x + 2.*y - 7.)**2 + (2.*x + y - 5.)**2
        return f
    
    def rosenbrock_constrained(x, y):
        # Rosenbrock function constrained to a disk.
        if x**2 + y**2 <= 2.:
            f = rosenbrock(x, y)
        else:
            f = 1000.
        return f
    
    # Initial point.
    x0 = np.array([2., 2.])
    # Find minimum of the Rosenbrock's function.
    x, E = simulated_annealing(rosenbrock, x0, 
                               T0=1000., C=20, eps=1e-20, burn=50,
                               verbose=True)
    print('Rosenbrock function [1, 1]:')
    print(f'Coordinates: {x[0]:.4f}, {x[1]:.4f}')
    print(f'Energy func.: {E:.4e}\n')

    # Initial point.
    x0 = np.array([0., 0.])
    # Find minimum of the Beale's function.
    x, E = simulated_annealing(beale, x0, 
                               T0=1000., C=20., eps=1e-24, 
                               burn=200, sigma=0.6, verbose=True)
    print('Beale function [3, 0.5]:')
    print(f'Coordinates: {x[0]:.4f}, {x[1]:.4f}')
    print(f'Energy func.: {E:.4e}\n')

    # Initial point.
    x0 = np.array([0., 0.])
    # Find minimum of the Booth's function.
    x, E = simulated_annealing(booth, x0, 
                               T0=1000., C=20., eps=1e-18, burn=50)
    print('Booth function [1, 3]:')
    print(f'Coordinates: {x[0]:.4f}, {x[1]:.4f}')
    print(f'Energy func.: {E:.4e}\n')

    # Initial point.
    x0 = np.array([0., 1.])
    # Find minimum of the Rosenbrock's function,
    # subject to constraint (circle): x**2 + y**2 <= 2.
    x, E = simulated_annealing(rosenbrock_constrained, x0, 
                               T0=1000., C=20., eps=1e-18, 
                               burn=100, sigma=0.6, verbose=True)
    print('Rosenbrock (constrained) function [1, 1]:')
    print(f'Coordinates: {x[0]:.4f}, {x[1]:.4f}')
    print(f'Energy func.: {E:.4e}')
