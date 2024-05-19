import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from tqdm import tqdm

def func_ci(func, alpha: float = 0.95, n: int = 1000, print_results: bool = False, plot_result: bool = False,
            result_names: list = None, **kwargs) -> [tuple]:
    """
    Calculates the confidence interval of results of a given function. As the confidence interval makes sense only
    for numerical values, the function should return numerical values.
    :param func: function to calculate the confidence interval of the results of.
    The function can return multiple values, however the dimension should be consistent across all results.
    :param alpha: confidence level
    :param n: number of samples.
    :param print_results: whether to print the results
    :param plot_result: whether to plot the results
    :param result_names: names of the results to plot. If None, the results will be named Result 1, Result 2, ...
    :param kwargs: arguments to pass to the function
    :return: list of tuples of the form (mean, std, (ci_lower, ci_upper)), where each tuple corresponds
    to a result of the function.
    """

    first_result = func(**kwargs)
    samples = np.zeros((n, len(first_result)))
    samples[0] = first_result

    for i in tqdm(range(1, n), desc=f'Sampling for {func.__name__}', disable=not print_results):
        samples[i] = func(**kwargs)

    results_list = samples.T.tolist()
    confidence_intervals = [list_ci(results_list[i], alpha) for i in range(len(first_result))]

    if result_names is None:
        result_names = [f'Result {i + 1}' for i in range(len(first_result))]

    if print_results:
        for i, ci in enumerate(confidence_intervals):
            print(f'{result_names[i]}: mean={ci[0]:.3f}, std={ci[1]:.3f}, ci=({ci[2][0]:.3f}, {ci[2][1]:.3f})')

    if plot_result:
        if kwargs.get('emb_name') is None:
            emb_name = 'unknown_emb'
        else:
            emb_name = kwargs.get('emb_name')
        # plotting the boxplot for each result
        plot_metrics(metrics=results_list, metrics_names=result_names,
                     model_name=func.__name__, emb_name=emb_name)

    return confidence_intervals


def list_ci(array: list, alpha: float = 0.95) -> tuple:
    """
    Calculates the confidence interval of a given list
    :param array: list of values
    :param alpha: confidence level
    :return: tuple of the form (mean, std, (ci_lower, ci_upper))
    """
    n = len(array)
    mean = np.mean(array)
    std = np.std(array)
    se = std / np.sqrt(n)
    h = se * stats.t.ppf((1 + alpha) / 2, n - 1)
    return mean, std, (mean - h, mean + h)


def plot_metrics(metrics: [list], metrics_names: list, model_name, emb_name):
    """
    Plot the metrics for the models.
    :param metrics: the metrics to plot. List of lists of metric values.
    :param metrics_names: the names of the metrics.
    :param model_name: the names of the model.
    :param emb_name: name of the embedding used
    """
    # initializing a new plot
    plt.figure()
    plt.boxplot(metrics, labels=metrics_names)
    plt.title(f'Model Metrics for {model_name}')
    plt.xlabel('Metrics')
    plt.ylabel('Value')
    plt.savefig(f'plots/{model_name}_w_{emb_name}_metrics.png')
