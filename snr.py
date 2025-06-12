import numpy as np
import warnings
from scipy.linalg import lstsq
import matplotlib.pyplot as plt


def computeFiringRate(x, C, b):
    """Compute the firing rate of a log-linear Poisson neuron model"""
    return np.exp(x @ C + b)


def updateBiasToMatchTargetFiringRate(
    currentRatePerBin, currentBias, targetRatePerBin=0.05
):
    # easy to find the bias to compensate for the change in firing rate due to changing C
    assert currentBias.size == currentRatePerBin.size
    return currentBias + np.log(targetRatePerBin / currentRatePerBin)


def scaleCforTargetSNR(latentTraj, C, b, targetRatePerBin, targetSNR, SNR_method):
    maxGain = 1.0
    for _ in range(20):
        SNR, _ = SNR_method(latentTraj, C * maxGain, b, targetRatePerBin)
        if SNR > targetSNR:
            break
        else:
            maxGain = maxGain * 1.5

    minGain = 0.5
    for _ in range(20):
        SNR, _ = SNR_method(latentTraj, C * minGain, b, targetRatePerBin)
        if SNR > targetSNR:
            minGain = minGain * 0.5
        else:
            break

    #  start the bisection search for the target SNR
    for _ in range(40):
        gain = (maxGain + minGain) / 2
        SNR, _ = SNR_method(latentTraj, C * gain, b, targetRatePerBin)
        if SNR > targetSNR:
            maxGain = gain
        elif SNR <= targetSNR:
            minGain = gain

    SNR, b = SNR_method(latentTraj, C * gain, b, targetRatePerBin)

    if (SNR - targetSNR) > 0.1 * np.abs(targetSNR):
        print(
            f"Warning: SNR reached is way greater than the target SNR {targetSNR}. SNR =",
            SNR,
        )
    if (SNR - targetSNR) < -0.1 * np.abs(targetSNR):
        print(
            f"Warning: SNR reached is less than the target SNR {targetSNR}. SNR =", SNR
        )

    return (C * gain), b, SNR


def compute_mutual_coherence(C):
    C_normalized = C / np.linalg.norm(C, axis=0)
    CC = C_normalized.T @ C_normalized

    return np.max(np.abs(CC - np.diag(np.diag(CC))))


def euclidean_proj_l1ball(v, s=1):
    """Compute the Euclidean projection on a L1-ball

    Solves the optimisation problem (using the algorithm from [1]):

        min_w 0.5 * || w - v ||_2^2 , s.t. || w ||_1 <= s

    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project

    s: int, optional, default: 1,
       radius of the L1-ball

    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the L1-ball of radius s

    Notes
    -----
    Solves the problem by a reduction to the positive simplex case

    See also
    --------
    euclidean_proj_simplex
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    (n,) = v.shape  # will raise ValueError if v is not 1-D
    # compute the vector of absolute values
    u = np.abs(v)
    # check if v is already a solution
    if u.sum() <= s:
        # L1-norm is <= s
        return v
    # v is not already a solution: optimum lies on the boundary (norm == s)
    # project *u* on the simplex
    w = euclidean_proj_simplex(u, s=s)
    # compute the solution to the original problem on v
    w *= np.sign(v)
    return w


def euclidean_proj_simplex(v, s=1):
    """Compute the Euclidean projection on a positive simplex

    Solves the optimisation problem (using the algorithm from [1]):

        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0

    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project

    s: int, optional, default: 1,
       radius of the simplex

    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex

    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.

    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    (n,) = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - s) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return w


def compute_SNR(latentTraj, C, b, targetRatePerBin):
    # note that the latentTraj is assumed to be of variance 1
    firing_rates = computeFiringRate(latentTraj, C, b)
    b = updateBiasToMatchTargetFiringRate(
        np.mean(firing_rates, axis=0), b, targetRatePerBin=targetRatePerBin
    )
    firing_rates = computeFiringRate(latentTraj, C, b)

    # U2E2 = _C_to_U2E2(C.T)
    # SNR = -10 * np.log10(np.mean(firing_rates**-1 @ U2E2))

    SNR = 0
    for i, firing_rate in enumerate(firing_rates):
        SNR += np.trace(np.linalg.inv(C @ np.diag(firing_rate) @ C.T))
    SNR = SNR / firing_rates.shape[0]
    SNR = 10 * np.log10(C.shape[0] / SNR)

    return SNR, b


def generate_random_loading_matrix(dLatent, dNeurons, pCoherence, pSparsity=0, C=None):
    # Constructing Low Mutual Coherence Matrix
    # via Direct Mutual Coherence Minimization (DMCM)
    # Lu, Canyi, Huan Li, and Zhouchen Lin. "Optimized projections for compressed sensing via direct mutual coherence minimization." Signal Processing 151 (2018): 45-55.
    if C is None:
        C = np.random.randn(dLatent, dNeurons)
        C = C * (np.random.rand(dLatent, dNeurons) > pSparsity)
        C /= np.linalg.norm(C, axis=0)

    T = 15
    K = 1000
    rho = 0.5
    eta = 1.1
    lbda = 0.9
    alpha = lbda * rho

    coh = compute_mutual_coherence(C)
    for _ in range(T):
        for _ in range(K):
            coh = compute_mutual_coherence(C)
            if coh < pCoherence:
                C /= np.linalg.norm(C, axis=0)
                return C

            VV = (C.T @ C - np.eye(dNeurons)) / rho
            v = euclidean_proj_l1ball(VV.flatten(), s=1)
            V = np.reshape(v, (dNeurons, dNeurons))

            MM = C - alpha * C @ (V + V.T)
            C = MM / np.linalg.norm(MM, axis=0)
        rho = rho / eta
        alpha = lbda * rho

    if coh >= pCoherence:
        warnings.warn(
            f"target Coherence {pCoherence} not reached, Current Coherence {coh}"
        )

    C /= np.linalg.norm(C, axis=0)
    return C


def generate_poisson_observations(
    latentTraj,
    C=None,
    dNeurons=100,
    targetRatePerBin=0.01,
    pCoherence=0.5,
    pSparsity=0.1,
    targetSNR=10.0,
    SNR_method=compute_SNR,
):
    """Automatically generates Poisson observations.

    Args:
        latentTraj: The latent trajectory.
        C: Optional loading matrix.
        dNeurons: Number of neurons.
        targetRatePerBin: Target firing rate per bin.
        pCoherence: Probability of coherence.
        pSparsity: Probability of sparsity.
        targetSNR: Target signal-to-noise ratio.
        SNR_method: Method to compute SNR.

    Returns:
        A tuple containing observations, C, b, firing_rates, and SNR.
    """
    assert pSparsity >= 0, "pSparsity must be between 0 and 1"
    assert pSparsity <= 1, "pSparsity must be between 0 and 1"
    assert dNeurons > 0, "dNeurons must be positive"
    assert targetRatePerBin > 0, "targetRatePerBin must be positive"
    if not np.all(np.isclose(np.std(latentTraj, axis=0), 1)):
        print("WARNING: latent trajectory must have unit variance. Normalizing...")
        latentTraj = latentTraj / np.std(latentTraj, axis=0)

    dLatent = latentTraj.shape[1]
    C = generate_random_loading_matrix(dLatent, dNeurons, pCoherence, pSparsity, C=C)

    b = 1.0 * np.random.rand(1, dNeurons) - np.log(targetRatePerBin)
    C, b, SNR = scaleCforTargetSNR(
        latentTraj, C, b, targetRatePerBin, targetSNR=targetSNR, SNR_method=SNR_method
    )
    firing_rates = computeFiringRate(latentTraj, C, b)

    observations = np.random.poisson(firing_rates)

    return observations, C, b, firing_rates, SNR


if __name__ == "__main__":
    # Example usage
    latentTraj = np.random.randn(1000, 10)
    observations, C, b, firing_rate_per_bin, SNR = generate_poisson_observations(
        latentTraj,
        C=None,
        dNeurons=100,
        targetRatePerBin=0.1,
        pCoherence=0.5,
        pSparsity=0.1,
        targetSNR=1.0,
        SNR_method=compute_SNR,
    )

    print("Generated observations shape:", observations.shape)
    print("C shape:", C.shape)
    print("b shape:", b.shape)
    print("Firing rate per bin shape:", firing_rate_per_bin.shape)
    print("SNR:", SNR)

    # Plot firing rates and observations for a few example neurons
    plt.figure(figsize=(12, 8))
    n_neurons_to_plot = 3
    time_points = np.arange(observations.shape[0])
    
    for i in range(n_neurons_to_plot):
        plt.subplot(n_neurons_to_plot, 1, i+1)
        plt.plot(time_points, firing_rate_per_bin[:, i], 'b-', label='Firing Rate', alpha=0.7)
        plt.plot(time_points, observations[:, i], 'r.', label='Spikes', markersize=2, alpha=0.5)
        plt.title(f'Neuron {i+1}')
        plt.ylabel('Counts')
        if i == n_neurons_to_plot - 1:
            plt.xlabel('Time Bin')
        plt.legend()
    
    plt.tight_layout()
    plt.show()