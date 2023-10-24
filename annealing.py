def simulated_annealing(f, x0, bounds=None, T0=1., C=10., sigma=1., 
                        fs=0.5, burn=20, eps=1e-8, tol=1e-3, wait=50, 
                        verbose=False):
    """
    Simulaled annealing algorithm.
     
    Simulaled annealing with exponential temperature decay. 
    A metaheuristic algorithm commonly used for optimization 
    problems of black-box functions (and systems) with large 
    search spaces. This algorithm works in the domain of real 
    valued arguments, and uses certain enhancements on top of 
    the classical approach.

    Parameters
    ----------
    f : function
        User-supplied external function f(x) which describes the 
        so-called energy function of the system being optimized,
        where x is a vector. This can also be a black-box func-
        tion or even a simulation process that takes inputs and 
        returns a state of the system.
    x0 : np.array or list
        Initial values for the variables of the energy function.
        For example, if the energy functions is f(x,y), then
        initial values are given as [x0, y0], where x0 is the 
        initial value for the variable x and y0 for the variable
        y. Order of array elements is important!
    bounds : list of tuples or None, default=None
        List of two-element tuples which define bounds on energy 
        function variables. The number and order of list elements
        is the same as for the array `x0` for the initial values.
        For example, if the energy function is f(x,y), then 
        bounds are defined as follows: [(xl,xu), (yl,yu)], where
        xl, xu are, respectively, lower and upper bounds for the
        variable x, and yl, yu represnt the same limits for the
        variable y. Order of tuples in the list is important!
    T0 : float, default=1.
        Initial temperature value.
    C : float, default=10.
        Constant decay value of the temperature scheduling.
        Temperature after k-th iteration is determined from 
        the following relation: Tk = T0*exp(-k/C).
    sigma : float or list, default=1.
        Standard deviation of a statistical distribution for
        the random walk by which new candidates are generated. 
        If scalar value is given, it represents a unique stan-
        dard deviation for the multivariate distribution. If
        list is given, it represents different standard devi-
        ations, one for each of the variables of the energy
        function (i.e. each variable of the search space). In
        this case, different steps are taken in different dir-
        ections of the multi-dimensional search space. For
        example, if sigma=[s1, s2] then the covariance matrix
        of the associated multivariate distribution is as
        follows: cov = [[s1^2, 0], [0, s2^2]]. 
        Random numbers are initially drawn from the multivariate
        Student's t-distribution with a low degree of freedom. 
        After the burn-in period, random numbers are drawn from 
        the multivariate Normal distribution with much lower 
        standard deviation, i.e. MVN([0], fs*[cov]), where `fs` 
        is the factor by which the standard deviations from the
        covariance matrix are reduced. Multivariate random 
        samples are not statistically correlated.
    fs : float, default=0.5
        Factor for reducing the standard deviation of a statisti-
        cal distribution used for the random walk (see parameter
        `sigma` above for more information). Default value halves
        the `sigma` after the burn-in.
    burn : int, default=20
        Number of iterations with the original step size of
        the random walk from the Student's t distribution, after
        which the step size is reduced by switching over to the 
        Normal distribution with a lower standard deviation (see 
        parameter `sigma` above).
    eps : float, default=1e-6
        Temperature value at which the algorithm is stopped.
    tol : float, default=1e-3
        Tolerance for considering that absolute difference between 
        two successive energy function evaluations is small enough
        that it has not improved further.
    wait : int, default=50
        Wait period (iterations) before early stopping due to the 
        lack of improvement. Counter is incremented each time the
        absolute difference between two successive energy function 
        evaluations is below the `tol` level. If this counter ever
        exceeds the `wait` value, the optimization is terminated.
    verbose : bool, default=False
        Indicator for printing (on stdout) internal messages.

    Returns
    -------
    results : dictionary
        Result is a dictionary holding the following keys:
        'x' : np.array
            Coordinate values (i.e. parameters) of the energy 
            function's optimum point.
        'E' : float
            Energy function's optimal value.
        'x_all' : np.array
            Return the list of all x-values.
        'E_all' : np.array
            Return the list of all E-values.
    
    Raises
    ------
    ValueError
        Checking input parameter's value and raising error
        if it falls outside the valid range.
    
    Important
    ---------
    There is a relationship in the cooling schedule between the
    initial temperature, decay value and stopping temperature. 
    The user should plot the cooling schedule that he/she intends 
    to use and only then decide on the concrete `eps`, `burn` and 
    `sigma` parameter values. Defaults are provided for orientation 
    only, and may not be suited for all applications. As a rule of 
    thumb the `burn` parameter may be set near the iteration number 
    around which the cooling schedule curve exhibits a knee-point.
    If the bounds are set on the energy function's variables these 
    should be wide enough to allow the exploration of the search-
    space by the random walk.
    
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
    Early stopping is implemented by monitoring absolute value of
    the energy difference between any two succesive iterations and
    a waiting period.
    """
    from numpy import exp, pi, sqrt
    from numpy import array, identity, zeros, repeat, atleast_1d
    from scipy import stats

    # Checking input parameters for validity.
    if T0 <= 0.:
        raise ValueError('Initial temperature must be positive.')
    if fs <= 0. or fs > 1.:
        raise ValueError('Parameter "fs" must be between 0 and 1.')
    if C <= 0.:
        raise ValueError('Parameter "C" must be positive number.')
    if type(x0) is list:
        # Turn list into a numpy array.
        x0 = array(x0)

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

    if type(sigma) is list:
        # Turn list into a numpy array.
        sigma = array(sigma)**2
    else:
        # Turn scalar into a numpy array.
        sigma = repeat(sigma**2, repeats=N)

    # Calculate number of steps after the burn-in.
    j = 0
    temperature = T0
    while temperature > eps:
        temperature = T0 * exp(-j/C)
        j += 1
    n_steps = j - burn

    # Degrees of freedom for the Chi2 distr.
    nu = 10
    # Mean values vector and covariance matrix
    # for the Multivariate Normal distribution.
    mu = zeros(N)
    eye = identity(N)
    cov = sigma * eye

    # Random samples are pre-generated outside the main loop for the reasons
    # of speeding-up the code execution (see the results of code profiling).
    # Pre-generated random samples from the Chi2 distribution with "nu" 
    # degrees of freedom.
    u = stats.chi2(df=nu).rvs(size=burn)
    # Pre-generate random samples from the Multivariate Normal distribution
    # for drawing from the Multivariate Student's t distribution during the
    # burn-in period.
    z = stats.multivariate_normal(mean=mu, cov=cov).rvs(size=burn)
    # Pre-generate random samples from the Multivariate Normal distribution
    # for the rest of the optimization process.
    w = stats.multivariate_normal(mean=mu, cov=fs*cov).rvs(size=n_steps)

    k = 0
    early_stop = 0
    x_all = []; E_all = []
    while T > eps:
        # Generate coordinates for the random walk.
        if k < burn:
            # Original step size during the burn-in, from the multivariate 
            # Student's t distribution with low degrees of freedom.
            walk = sqrt(nu/u[k]) * z[k]  # step size
        else:
            # Reduce the step size after the temperature cools down by
            # switching to the multivariate Normal distribution with a
            # lower standard deviation.
            j = k - burn
            walk = w[j]  # step size

        # Variable "walk" must be 1d-vector and not a scalar.
        walk = atleast_1d(walk)
        # Start from the best known position after the burn-in.
        if k == burn:
            x = x_top      
        
        # Random walk in N dimensions.
        x_new = x + walk

        # Checking bounds on the function's variables.
        # If the variable is found outside the bounds <l_bound, u_bound> 
        # it is re-translated in the opposite direction, using half the
        # step from the current random walk.
        if bounds is not None:
            for i, bound in enumerate(bounds):
                # Check for each variable:
                if x_new[i] < bound[0]:
                    # Coord. is below the lower bound.
                    # Move the new coord. to the right.
                    x_new[i] = x[i] + abs(walk[i]/2)
                    if verbose:
                        print(f'iter. {k} (var. {i}): l_bound')
                elif x_new[i] > bound[1]:
                    # Coord. is above the upper bound.
                    # Move the new coord. to the left.
                    x_new[i] = x[i] - abs(walk[i]/2)
                    if verbose:
                        print(f'iter. {k} (var. {i}): u_bound')

        # Compute energy function at new coords.
        E_new = f(*x_new)
        # Energy difference.
        Delta_E = E_new - E

        # Metropolis acceptance criterion.
        if Delta_E <= 0.:
            x = x_new
            E = E_new
        else:
            r = stats.uniform.rvs()
            # Boltzman probability.
            alpha = (2*pi*T)**(-N/2.) * exp(-abs(Delta_E)**2/(2*T))
            # Stochastic acceptance.
            if r < alpha:
                x = x_new
                E = E_new

        # Save the best solution found thus far.
        if E < E_top:
            x_top = x_new
            E_top = E_new

        x_all.append(x)
        E_all.append(E)

        # Temperature schedule (cooling).
        T = T0 * exp(-k/C)

        # Early stopping.
        if abs(Delta_E) <= tol: 
            early_stop += 1
        if early_stop >= wait:
            print(f'Early stopping after {k} iterations.')
            break

        k += 1
    
    if verbose:
        print('Final temperature: {:.3e} after {:d} iterations.'
              .format(T, k))
    
    # Format results as a dictionary.
    results = {
        'x': x_top,      # Best x-values array.
        'E': E_top,      # Best energy func. value.
        'x_all': x_all,  # x-values arrays for all iterations.
        'E_all': E_all   # Energy func. values for all iterations
    }
    
    return results


if __name__ == "__main__":
    """ Test with simple benchmark functions. """
    import numpy as np
    import matplotlib.pyplot as plt

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
            f = 10 * (x**2 + y**2) * rosenbrock(x, y)
        return f
    

    # Initial point.
    x0 = [0., 2.]
    # Find minimum of the Rosenbrock's function.
    res = simulated_annealing(
        rosenbrock, x0, 
        bounds=[(-2,5), (-2,5)], fs=0.1,
        T0=1000., C=20, sigma=[0.4, 0.8], 
        eps=1e-20, burn=50, verbose=True)
    print('Rosenbrock function [1, 1]:')
    print(f"Coordinates: {res['x'][0]:.4f}, {res['x'][1]:.4f}")
    print(f"Energy func.: {res['E']:.4e}\n")
    plt.plot(res['x_all'])
    plt.xlabel('Iterations')
    plt.ylabel('x-values')
    plt.show()
    plt.semilogy(res['E_all'])
    plt.xlabel('Iterations')
    plt.ylabel('Energy func. value')
    plt.show()

    # Initial point.
    x0 = np.array([0., 0.])
    # Find minimum of the Beale's function.
    res = simulated_annealing(beale, x0, bounds=[(-3,6), (-3,5)],
                              T0=1000., C=20., sigma=[0.8, 0.6], 
                              eps=1e-24, fs=0.1, burn=200)
    print('Beale function [3, 0.5]:')
    print(f"Coordinates: {res['x'][0]:.4f}, {res['x'][1]:.4f}")
    print(f"Energy func.: {res['E']:.4e}\n")

    # Initial point.
    x0 = np.array([0., 0.])
    # Find minimum of the Booth's function.
    res = simulated_annealing(booth, x0, 
                              T0=1000., C=20., eps=1e-18, burn=50)
    print('Booth function [1, 3]:')
    print(f"Coordinates: {res['x'][0]:.4f}, {res['x'][1]:.4f}")
    print(f"Energy func.: {res['E']:.4e}\n")

    # Initial point.
    x0 = np.array([0., 0.])
    # Find minimum of the Rosenbrock's function,
    # subject to constraint (circle): x**2 + y**2 <= 2.
    res = simulated_annealing(rosenbrock_constrained, x0, 
                              bounds=[(-2,4), (-2,4)],
                              T0=1000., C=20., eps=1e-18, 
                              burn=100, sigma=0.5)
    print('Rosenbrock (constrained) function [1, 1]:')
    print(f"Coordinates: {res['x'][0]:.4f}, {res['x'][1]:.4f}")
    print(f"Energy func.: {res['E']:.4e}")
