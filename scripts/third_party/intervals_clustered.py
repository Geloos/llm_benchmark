import numpy as np
import scipy.stats as stats
import statsmodels.stats.proportion as sm_proportion
import pickle

__all__ = ['clt_subtask_confidence_interval',
           'bootstrap_subtask_confidence_interval',

           'bayes_subtask_credible_interval_IS',
           'bayes_subtask_credible_interval_IS_parallel',]


################################################################
## Frequentist (Confidence Intervals)

### Clustered Standard Error (CLT-based)
def clt_subtask_confidence_interval(subtask_data, alpha):
    '''
    Compute confidence intervals in the subtask setting according to clustered standard error formulation.
    (See Miller, 2024: https://arxiv.org/abs/2411.00640 for details.)

    Parameters
    ----------    
    subtask_data: array
        list containing T np.ndarrays of sampled data
    alpha: float, or list of floats
        confidence level
    
    Returns
    -------
    interval: np.ndarray
        Lower and upper bounds of the confidence interval (as a (..., 2) array)
    '''
    if isinstance(alpha, float):
        alpha = np.array([alpha])
    else:
        alpha = np.array(alpha)

    T = len(subtask_data)
    n_per_task = [x.shape[-1] for x in subtask_data]
    shapes_per_task = [x.shape[:-1] for x in subtask_data]

    assert all(shapes_per_task[0] == shape for shape in shapes_per_task)
    data_shape = shapes_per_task[0]
    assert len(data_shape) in (0,1)
    if len(data_shape) == 0:
        no_repeats = True
        data_shape = 1
        subtask_data = [x[None,:] for x in subtask_data]
    else:
        data_shape = data_shape[0] 
        no_repeats = False

    # Flatten the data
    subtask_data_flat = np.concatenate(subtask_data, axis=-1)

    # Get X_bar
    X_bar = subtask_data_flat.mean(-1)
    assert X_bar.shape == (data_shape,)

    # Get SE_CLT
    N = sum(n_per_task)
    assert subtask_data_flat.shape[-1] == N

    SE_CLT = (np.sqrt(X_bar * (1 - X_bar) / N))[None,...] # shape == (1, *data.shape[:-1])

    # Do the cluster adjustment
    cluster_adjustment = np.zeros((data_shape,))
    for t in range(T):
        residual = subtask_data[t] - X_bar[:, None]
        assert residual.shape == (data_shape, n_per_task[t])

        residual_cov = np.matmul(residual[:, :, None], residual[:, None, :])
        assert residual_cov.shape == (data_shape, n_per_task[t], n_per_task[t])

        residual_cov_diag = residual_cov.diagonal(axis1=-2, axis2=-1)
        assert residual_cov_diag.shape == (data_shape, n_per_task[t])

        cluster_adjustment += residual_cov.sum((-1,-2)) - residual_cov_diag.sum(-1)
    cluster_adjustment /= (N**2)
    
    # Get the final SE as sqrt(SE_CLT**2 + cluster_adjustment)
    # NOTE: In some cases, we may have 0 as final SE, i.e.
    #           SE_CLT **2 == - cluster_adjustment 
    # which is fine, however, we need to ensure numerical stability by taking the absolute value.
    # Although we'll shout at the user if we see any cases where
    #           SE_CLT **2 < - cluster_adjustment
    # occurs and it doesn't seem to be due to numerical instability (though this should never happen)
    var_clustered = (SE_CLT ** 2) + cluster_adjustment[None,...]
    if np.any(var_clustered < -1e-12):
        print("!!! WARNING: SE_CLT ** 2 < - cluster_adjustment !!!")
    
    SE_clustered = np.abs(var_clustered)**0.5

    # Get the confidence intervals
    z = stats.norm.ppf(1 - alpha / 2)  # shape == (len(alpha),)
    SE_alpha_scaled = z[:, None] * SE_clustered

    assert SE_alpha_scaled.shape == (len(alpha), data_shape)

    intervals = np.array([X_bar - SE_alpha_scaled, X_bar + SE_alpha_scaled])
    assert intervals.shape == (2, len(alpha), data_shape)

    # Transpose to get the desired shape
    intervals = intervals.transpose(1, 2, 0)
    assert intervals.shape == (len(alpha), data_shape, 2)

    if len(alpha) == 1:
        intervals = intervals[0]
        if no_repeats:
            intervals = intervals[0]
    else:
        if no_repeats:
            intervals = intervals[:,0]
    return intervals



### Clustered Bootstrap
def bootstrap_subtask_confidence_interval(subtask_data, alpha, K=100, sample_length=None, max_parallel_repeats=1000):
    '''
    Compute the confidence interval for the parameter of a Bernoulli distribution using bootstrapping in the subtask setting.

    Parameters
    ----------
    subtask_data: array
        list containing T np.ndarrays of sampled data
    alpha: float, or list of floats
        Significance level
    K: int
        Number of bootstrap samples

    Returns
    -------
    confidence_interval: tuple
        Lower and upper bounds of the confidence interval
    '''
    if isinstance(alpha, float):
        alpha = np.array([alpha])
    else:
        alpha = np.array(alpha)

    no_repeat_dim = len(subtask_data[0].shape) == 1

    num_repeats = subtask_data[0].shape[0]
    if num_repeats > max_parallel_repeats:
        interval = np.zeros((len(alpha), num_repeats, 2))
        for i in range(num_repeats // max_parallel_repeats):
            interval[:, i*max_parallel_repeats:(i+1)*max_parallel_repeats, :] = bootstrap_subtask_confidence_interval([x[i*max_parallel_repeats:(i+1)*max_parallel_repeats, :] for x in subtask_data], alpha, K, sample_length, max_parallel_repeats)

        if (interval[-1,...] == 0).all():
            print(interval)
            print("ooh!")
    else:
    
        theta_t_hats = np.zeros((len(subtask_data), *subtask_data[0].shape[:-1], K))


        for t, data in enumerate(subtask_data):
            no_repeat_dim = len(data.shape) == 1
            if no_repeat_dim:
                data = data[None,:]

            n = data.shape[-1]
            if not isinstance(sample_length, int):
                sample_length = n
        
            indices = np.random.choice(n, (sample_length, K), replace=True)
            theta_t_bootstrapped = data[..., indices]

            theta_t_hats[t] = theta_t_bootstrapped.mean(axis=-2)  # shape == data.shape[:-1] + (K,)

            # theta_t_hats[t] = np.stack([data.mean(axis=-1)]*K, axis=-1)  # shape == data.shape[:-1] + (K,)
            

        theta_hats = theta_t_hats.mean(axis=0)  # shape == data.shape[:-1] + (K,)
        assert theta_hats.shape == (*subtask_data[0].shape[:-1], K)

        # indices = np.random.choice(K, (K,), replace=True)
        # theta_hats_bootstrapped = theta_hats[..., indices]  # shape == data.shape[:-1] + (K,)

        theta_hats_bootstrapped = theta_hats

        assert theta_hats_bootstrapped.shape == (*subtask_data[0].shape[:-1], K)

        percentiles = np.array([100 * alpha / 2, 100 * (1 - alpha / 2)]).transpose(1,0) # shape == (len(alpha), 2)

        if no_repeat_dim:
            theta_hats = theta_hats[None,...]
        
        interval = np.percentile(theta_hats, percentiles, axis=-1).transpose(0,2,1) # shape == (len(alpha), *data.shape[:-1], 2)
    
    if len(alpha) == 1:
        if no_repeat_dim:
            interval = interval[0]
        interval = interval[0]
    elif no_repeat_dim:
        interval = interval[:,0]
    
    return interval


################################################################
## Bayes (Credible Intervals)

### Importance Sampling
def bayes_subtask_credible_interval_IS(subtask_data, alpha, num_samples=10_000, max_parallel_repeats=1000):
    '''
    Compute the credible interval for the parameter of a subtask model using importance sampling.

    Parameters
    ----------
    subtask_data: list of np.ndarray
        List of arrays, where each array contains {X_{t,i}} for a given t.
    alpha: float
        Significance level
    num_samples: int
        Number of importance samples

    Returns
    -------
    interval: np.ndarray
        Lower and upper bounds of the credible interval
    '''
    if isinstance(alpha, float):
        alpha = np.array([alpha])
    else:
        alpha = np.array(alpha)

    if len(subtask_data[0].shape) == 1:
        no_repeats = True
        subtask_data = [x[None,:] for x in subtask_data]  # add in dummy repeats dimension
    else:
        no_repeats = False

    num_repeats = subtask_data[0].shape[0]
    if num_repeats > max_parallel_repeats:
        interval = np.zeros((len(alpha), num_repeats, 2))
        for i in range(num_repeats // max_parallel_repeats):
            interval[:, i*max_parallel_repeats:(i+1)*max_parallel_repeats, :] = bayes_subtask_credible_interval_IS([x[i*max_parallel_repeats:(i+1)*max_parallel_repeats, :] for x in subtask_data], alpha, num_samples, max_parallel_repeats)

        if (interval[-1,...] == 0).all():
            print(interval)
            print("ooh!")
    else:

        repeats = subtask_data[0].shape[0]
        n_per_task = [x.shape[-1] for x in subtask_data]
        T = len(subtask_data)

        assert all([X_t.shape[0] == repeats for X_t in subtask_data])

        # Convert data_list to np.ndarray of just NumSuccess and NumTrials
        data_compact = np.array([[x.sum(-1), [x.shape[-1] for _ in range(x.shape[0])]] for x in subtask_data]).transpose(2,0,1)
        assert data_compact.shape == (repeats, T, 2)

        # Sample a bunch of theta and ds
        thetas = np.random.beta(1, 1, size=(repeats, num_samples))
        ds = np.random.gamma(1, 1, size=(repeats, num_samples))

        # Get their likelihoods
        likelihoods = np.zeros((repeats, num_samples))
        for t in range(T):
            likelihoods += stats.betabinom(data_compact[:, None, t, 1],
                                        ds * thetas,
                                        ds * (1 - thetas)).logpmf(data_compact[:, None, t, 0])
        
        # Get the prior pdf
        # prior = stats.beta(1, 1).logpdf(thetas) + stats.gamma(1).logpdf(ds)
        # assert prior.shape == (repeats, num_samples)

        # Get the importance weights
        log_weights = likelihoods

        # Normalise the weights
        max_log_weights = log_weights.max(axis=-1, keepdims=True)
        weights = np.exp(log_weights - max_log_weights)
        weights /= weights.sum(axis=-1, keepdims=True)

        # Resample the thetas
        thetas_post = np.zeros((repeats, num_samples))
        for r in range(repeats):
            thetas_post[r] = thetas[r, np.random.choice(num_samples, size=num_samples, replace=True, p=weights[r])]
        assert thetas_post.shape == (repeats, num_samples)

        # Get the credible intervals
        percentiles = np.array([100 * alpha / 2, 100 * (1 - alpha / 2)]).transpose(1,0) # shape == (len(alpha), 2)S
        interval = np.percentile(thetas_post, percentiles, axis=-1).transpose(0,2,1) # shape == (len(alpha), *data.shape[:-1], 2)

    if len(alpha) == 1:
        interval = interval[0]
        if no_repeats:
            interval = interval[0]
    else:
        if no_repeats:
            interval = interval[:,0]
    return interval

def bayes_subtask_credible_interval_IS_parallel(subtask_data_collection, alpha, theta_vals, n_vals, num_samples=10_000):
    '''
    Compute the credible interval for the parameter of a subtask model using importance sampling
    IN PARALLEL for a variety of theta values AND n values.

    Parameters
    ----------
    subtask_data_collection: dict[int, np.ndarray]
        Dictionary with 
            keys: n
            values: len(n_vals)-length list containing len(theta_vals) lists of length T containing numpy arrays of shape (repeats, n_per_task[t])
                    (note: each n may have a different value for T, but the same repeats value)
    alpha: float
        Significance level
    num_samples: int
        Number of importance samples

    Returns
    -------
    interval: np.ndarray
        Lower and upper bounds of the credible interval
    '''
    if isinstance(alpha, float):
        alpha = np.array([alpha])
    else:
        alpha = np.array(alpha)

    repeats = subtask_data_collection[n_vals[0]][0][0].shape[0]
    T_per_n = [len(subtask_data_collection[n_val][0]) for n_val in n_vals]
    max_T = max(T_per_n)

    for n_idx, n_val in enumerate(n_vals):
        assert all([len(X_t) == T_per_n[n_idx] for X_t in subtask_data_collection[n_val]])

        for theta_idx in range(len(theta_vals)):
            assert sum([X_t.shape[-1] for X_t in subtask_data_collection[n_val][theta_idx]]) == n_val 

            for t in range(len(subtask_data_collection[n_val][theta_idx])):
                assert subtask_data_collection[n_val][theta_idx][t].shape[0] == repeats
                assert len(subtask_data_collection[n_val][theta_idx][t].shape) == 2
            
    # Convert subtask_data_collection to np.ndarray of just NumSuccess and NumTrials
    num_success = np.array([[[X_t.sum(-1)   for X_t in subtask_data_collection[n_val][theta_idx]] + [np.zeros((repeats,))]*(max_T - len(subtask_data_collection[n_val][theta_idx])) for theta_idx in range(len(theta_vals))] for n_val in n_vals])
    num_trials  = np.array([[[[X_t.shape[-1]] * repeats for X_t in subtask_data_collection[n_val][theta_idx]] + [np.zeros((repeats,))]*(max_T - len(subtask_data_collection[n_val][theta_idx])) for theta_idx in range(len(theta_vals))] for n_val in n_vals])

    assert num_success.shape == (len(n_vals), len(theta_vals), max_T, repeats)
    assert num_trials.shape  == (len(n_vals), len(theta_vals), max_T, repeats)

    # data_compact = np.array([[x.sum(-1), [x.shape[-1] for _ in range(x.shape[0])]] for x in subtask_data_collection]).transpose(2,0,1)
    # assert data_compact.shape == (repeats, T, 2)

    # Sample a bunch of theta and ds
    thetas =  np.random.beta(1, 1, size=(len(n_vals), len(theta_vals), repeats, num_samples))
    ds     = np.random.gamma(1, 1, size=(len(n_vals), len(theta_vals), repeats, num_samples))

    # Get their likelihoods
    likelihoods = np.zeros((len(n_vals), len(theta_vals), repeats, num_samples))
    for t in range(max_T):
        likelihoods += stats.betabinom(num_trials[:, :, t, :, None],
                                       ds * thetas,
                                       ds * (1 - thetas)).logpmf(num_success[:, :, t, :, None])
    
    # Get the prior pdf
    # prior = stats.beta(1, 1).logpdf(thetas) + stats.gamma(1).logpdf(ds)
    # assert prior.shape == (repeats, num_samples)

    # Get the importance weights
    log_weights = likelihoods

    # Normalise the weights
    max_log_weights = log_weights.max(axis=-1, keepdims=True)
    weights = np.exp(log_weights - max_log_weights)
    weights /= weights.sum(axis=-1, keepdims=True)

    # Resample the thetas
    thetas_post = np.zeros((len(n_vals), len(theta_vals), repeats, num_samples))
    for n_idx in range(len(n_vals)):
        for theta_idx in range(len(theta_vals)):
            for r in range(repeats):
                thetas_post[n_idx, theta_idx, r] = thetas[n_idx, theta_idx, r, np.random.choice(num_samples, size=num_samples, replace=True, p=weights[n_idx, theta_idx, r])]

    # Get the credible intervals
    percentiles = np.array([100 * alpha / 2, 100 * (1 - alpha / 2)]).transpose(1,0) # shape == (len(alpha), 2)S
    interval = np.percentile(thetas_post, percentiles, axis=-1)#.transpose(0,2,1) # shape == (len(alpha), *data.shape[:-1], 2)

    assert interval.shape == (len(alpha), 2, len(n_vals), len(theta_vals), repeats)
    # put len-2 dimension at the end, repeats at start and n_vals second
    interval = interval.transpose(4, 2, 0, 3, 1)
    assert interval.shape == (repeats, len(n_vals), len(alpha), len(theta_vals), 2)
    # print(thetas_post.shape, percentiles.shape, interval.shape)
    
    if len(alpha) == 1:
        return interval[0]
    return interval

################################################################
## Testing

if __name__ == "__main__":
    from utils import sample_subtasks, flatten_subtask_data
    from intervals import *

    np.random.seed(0)

    for repeats in [1,3]:
        print(f"REPEATS = {repeats}")
        theta = 0.6
        T = 5

        n_per_task = np.random.randint(3, 10, size=T)
        
        print("Data:")
        subtask_data = sample_subtasks(n_per_task, theta, shape=(repeats,))
        for x in subtask_data:
            print(x, x.shape)

        subtask_data_flat = flatten_subtask_data(subtask_data)
        print()
        print("Flattened data:")
        print(subtask_data_flat, subtask_data_flat.shape)


        print()

        print("Intervals with alpha = 0.05")
        alpha = 0.05

        bayes_simple = bayes_credible_interval(subtask_data_flat, alpha, prior=(1,1))
        freq_simple  = freq_confidence_interval(subtask_data_flat, alpha)
        wils_simple  = freq_confidence_interval_better(subtask_data_flat, alpha, method='wilson')
        clop_simple  = freq_confidence_interval_better(subtask_data_flat, alpha, method='beta')
        boot_simple  = bootstrap_confidence_interval(subtask_data_flat, alpha)
        
        clt_subtask = clt_subtask_confidence_interval(subtask_data, alpha)
        IS_subtask   = bayes_subtask_credible_interval_IS(subtask_data, alpha)
        boot_subtask = bootstrap_subtask_confidence_interval(subtask_data, alpha)


        intervals = {'bayes': bayes_simple, 'freq': freq_simple, 'wils': wils_simple, 'clop': clop_simple, 'boot': boot_simple,
                    'IS': IS_subtask,
                    'clt': clt_subtask,
                    'boot_subtask': boot_subtask}

        for x in intervals:
            print(x)
            print(intervals[x], intervals[x].shape)

        
        print("Intervals with alpha = [0.05, 0.1]")
        alpha = [0.05, 0.1]

        bayes_simple = bayes_credible_interval(subtask_data_flat, alpha, prior=(1,1))
        freq_simple  = freq_confidence_interval(subtask_data_flat, alpha)
        wils_simple  = freq_confidence_interval_better(subtask_data_flat, alpha, method='wilson')
        clop_simple  = freq_confidence_interval_better(subtask_data_flat, alpha, method='beta')
        boot_simple  = bootstrap_confidence_interval(subtask_data_flat, alpha)
        
        clt_subtask = clt_subtask_confidence_interval(subtask_data, alpha)
        IS_subtask   = bayes_subtask_credible_interval_IS(subtask_data, alpha)
        boot_subtask = bootstrap_subtask_confidence_interval(subtask_data, alpha)


        intervals = {'bayes': bayes_simple, 'freq': freq_simple, 'wils': wils_simple, 'clop': clop_simple, 'boot': boot_simple,
                    'IS': IS_subtask,
                    'clt': clt_subtask,
                    'boot_subtask': boot_subtask}

        for x in intervals:
            print(x)
            print(intervals[x], intervals[x].shape)

        print()

        print("Parallel IS:")
        from utils import generate_sample_subtask_collection

        N_t = 5
        Ts = [2, 3, 4]
        theta_vals = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        alphas = [0.05, 0.1]
        repeats = 6
        seed = 0

        n_vals = [N_t*t for t in Ts]
        n_per_tasks = [np.array([N_t]*t) for t in Ts]

        assert all(n_vals[i] == n_per_tasks[i].sum() for i in range(len(Ts)))

        # generate the data
        subtask_samples, subtask_samples_flat = generate_sample_subtask_collection(n_vals, theta_vals, n_per_tasks, repeats=repeats, seed=seed)

        IS_intervals = bayes_subtask_credible_interval_IS_parallel(subtask_samples, alphas, theta_vals, n_vals, num_samples=10_000)

        print(IS_intervals)
        print(f"IS_intervals.shape = {IS_intervals.shape}")
        print(f"Should be          = {(repeats, len(n_vals), len(alphas), len(theta_vals), 2)} = (repeats, len(n_vals), len(alphas), len(theta_vals), 2)")
