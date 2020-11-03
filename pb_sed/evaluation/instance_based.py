import numpy as np


def tp_fp_tn_fn(target_mat, decision_mat, reduce_axis=None):
    """Counts true positives, false positives, true negatives and false negatives.

    Args:
        target_mat: multi-hot matrix indicating ground truth
            (num_instances, num_classes)
        decision_mat: multi-hot matrix indicating detected (event) classes
            (N, num_instances, num_classes)
        reduce_axis:

    Returns:
        true positives:
        false positives:
        true negatives:
        false negatives:

    """
    tp = target_mat * decision_mat
    fp = (1. - target_mat) * decision_mat
    tn = (1. - target_mat) * (1. - decision_mat)
    fn = target_mat * (1. - decision_mat)
    if reduce_axis is not None:
        tp = np.sum(tp, axis=reduce_axis)
        fp = np.sum(fp, axis=reduce_axis)
        tn = np.sum(tn, axis=reduce_axis)
        fn = np.sum(fn, axis=reduce_axis)
    return tp, fp, tn, fn


def fscore(target_mat, decision_mat, beta=1., event_wise=False):
    """Computes instance-based f-score given binary decisions, i.e. after a decision threshold has been applied.

    Args:
        target_mat: multi-hot matrix indicating ground truth
            (num_instances, num_classes)
        decision_mat: multi-hot matrix indicating detected (event) classes
            (N, num_instances, num_classes)
        event_wise:
        beta:

    Returns:
        fscore:
        precision:
        recall:

    """
    reduce_axis = -2 if event_wise else (-2, -1)
    tp, fp, tn, fn = tp_fp_tn_fn(target_mat, decision_mat, reduce_axis)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(tp + fn, 1)
    f_beta = (1 + beta**2) * precision * recall / np.maximum(
        beta**2 * precision + recall, 1e-15
    )
    return f_beta, precision, recall


def substitutions_insertions_deletions(
        target_mat, decision_mat, reduce_axis=None
):
    """Counts substitutions, insertions and deletions for computation of an error rate

    Args:
        target_mat: multi-hot matrix indicating ground truth
            (num_instances, num_classes)
        decision_mat: multi-hot matrix indicating detected (event) classes
            (num_instances, num_classes)
        reduce_axis:

    Returns:
        substitutions:
        insertions:
        deletions:

    """
    _, insertions, _, deletions = tp_fp_tn_fn(
        target_mat, decision_mat, reduce_axis=None
    )
    if reduce_axis is not None and (
        reduce_axis in [-1, decision_mat.ndim - 1]
        or (
            isinstance(reduce_axis, (list, tuple))
            and (-1 in reduce_axis or (decision_mat.ndim - 1) in reduce_axis))
    ):
            # substitute
            insertions = np.sum(insertions, axis=-1, keepdims=True)
            deletions = np.sum(deletions, axis=-1, keepdims=True)
            substitutions = np.minimum(insertions, deletions)
            insertions -= substitutions
            deletions -= substitutions
    else:
        substitutions = np.zeros_like(insertions)
    if reduce_axis is not None:
        substitutions = np.sum(substitutions, axis=reduce_axis)
        insertions = np.sum(insertions, axis=reduce_axis)
        deletions = np.sum(deletions, axis=reduce_axis)

    return substitutions, insertions, deletions


def error_rate(target_mat, decision_mat, event_wise=False):
    """Computes instance-based error rate given binary decisions, i.e. after a decision threshold has been applied.

    Args:
        target_mat: multi-hot matrix indicating ground truth
            (num_instances, num_classes)
        decision_mat: multi-hot matrix indicating detected (event) classes
            (num_instances, num_classes)
        event_wise:

    Returns:
        error_rate:
        substitution_rate:
        insertion_rate:
        deletion_rate:

    """
    reduce_axis = -2 if event_wise else (-2, -1)
    substitutions, insertions, deletions = substitutions_insertions_deletions(
        target_mat, decision_mat, reduce_axis=reduce_axis
    )
    n_ref = np.maximum(np.sum(target_mat, axis=reduce_axis), 1e-15)
    er = (insertions + deletions + substitutions) / n_ref
    return er, substitutions / n_ref, insertions / n_ref, deletions / n_ref


def positive_class_precisions(target_mat, score_mat):
    """Calculate precisions for each true class.
    Core calculation of label precisions for one test sample.

    Args:
      target_mat: np.array of (num_samples, num_classes) bools indicating which classes are true.
      score_mat: np.array of (num_samples, num_classes) giving the individual classifier scores.

    Returns:
      class_indices: np.array of indices of the true classes.
      precision_at_hits: np.array of precisions corresponding to each of those
        classes.
    """
    num_samples, num_classes = score_mat.shape
    class_indices = np.cumsum(np.ones_like(score_mat), axis=-1) - 1
    target_mat = target_mat > 0
    # Only calculate precisions if there are some true classes.
    if not target_mat.any():
        return np.array([]), np.array([])
    # Retrieval list of classes for this sample.
    retrieved_classes = np.argsort(score_mat, axis=-1)[:, ::-1]
    sort_idx = (np.arange(num_samples)[:, None], retrieved_classes)
    class_indices = class_indices[sort_idx]
    target_mat = target_mat[sort_idx]
    # Num hits for every truncated retrieval list.
    retrieved_cumulative_hits = np.cumsum(target_mat, axis=-1)
    ranks = np.cumsum(np.ones_like(retrieved_cumulative_hits), axis=-1)
    # Precision of retrieval list truncated at each hit, in order of pos_labels.
    precision_at_hits = (retrieved_cumulative_hits[target_mat] / ranks[target_mat])
    return class_indices[target_mat].astype(np.int), precision_at_hits


def lwlrap_from_precisions(precision_at_hits, class_indices, num_classes=None):
    """Calculate label-weighted label-ranking average precision.

    Arguments:
        precision_at_hits:
        class_indices:

    Returns:
      lwlrap: overall unbalanced lwlrap which is simply
        np.sum(per_class_lwlrap * weight_per_class)
      per_class_lwlrap: np.array of (num_classes,) giving the lwlrap for each
        class.
      weight_per_class: np.array of (num_classes,) giving the prior of each
        class within the truth labels.

    """

    if num_classes is None:
        num_classes = np.max(class_indices) + 1
    per_class_lwlrap = np.zeros(num_classes)
    np.add.at(per_class_lwlrap, class_indices, precision_at_hits)
    labels_per_class = np.zeros(num_classes)
    np.add.at(labels_per_class, class_indices, 1)
    per_class_lwlrap /= np.maximum(1, labels_per_class)
    weight_per_class = labels_per_class / float(np.sum(labels_per_class))
    lwlrap = np.sum(per_class_lwlrap * weight_per_class)
    return lwlrap, per_class_lwlrap, weight_per_class


def lwlrap(target_mat, score_mat, event_wise=False):
    """Calculate label-weighted label-ranking average precision.
    All-in-one calculation of per-class lwlrap.

    Arguments:
      target_mat: np.array of (num_samples, num_classes) giving boolean ground-truth
        of presence of that class in that sample.
      score_mat: np.array of (num_samples, num_classes) giving the classifier-under-
        test's real-valued score for each class for each sample.

    Returns:
      lwlrap: overall unbalanced lwlrap which is simply
        np.sum(per_class_lwlrap * weight_per_class)
      per_class_lwlrap: np.array of (num_classes,) giving the lwlrap for each
        class.
      weight_per_class: np.array of (num_classes,) giving the prior of each
        class within the truth labels.


    >>> num_samples = 100
    >>> num_labels = 20

    >>> truth = np.random.rand(num_samples, num_labels) > 0.5
    >>> truth[0:1, :] = False # Ensure at least some samples with no truth labels.
    >>> scores = np.random.rand(num_samples, num_labels)

    >>> per_class_lwlrap = lwlrap(truth, scores)
    """
    if score_mat.ndim > 2:
        if target_mat.ndim == score_mat.ndim:
            assert len(target_mat) == len(score_mat)
            return np.array([lwlrap(t, s) for t, s in zip(target_mat, score_mat)])
        else:
            return np.array([lwlrap(target_mat, s) for s in score_mat])

    assert target_mat.shape == score_mat.shape
    pos_class_indices, precision_at_hits = positive_class_precisions(
        target_mat, score_mat
    )
    lwlrap_score, per_class_lwlrap, weight_per_class = lwlrap_from_precisions(
        precision_at_hits, pos_class_indices, num_classes=target_mat.shape[1]
    )
    if event_wise:
        return per_class_lwlrap
    else:
        return lwlrap_score


def _metric_curve(target_vec, score_vec, metric):
    """

    Args:
        target_vec:
        score_vec:
        metric:

    Returns:
        metric values:
        thresholds:
    """
    sort_indices = np.argsort(score_vec)
    score_vec = np.concatenate((score_vec[sort_indices], [np.inf]))
    target_vec = target_vec[sort_indices]

    tps = np.cumsum(np.concatenate((target_vec, [0]))[::-1])[::-1]
    n_sys = len(score_vec) - np.arange(len(score_vec)) - 1
    n_ref = tps[0]

    score_vec, valid_idx = np.unique(score_vec, return_index=True)
    tps = tps[valid_idx]
    n_sys = n_sys[valid_idx]
    if metric == 'f1':
        p = tps / np.maximum(n_sys, 1)
        r = tps / np.maximum(n_ref, 1)
        values = 2*p*r / (p+r+1e-15)
    elif metric == 'er':
        i = n_sys - tps
        d = n_ref - tps
        values = (i+d)/n_ref
    else:
        raise NotImplementedError
    thresholds = np.concatenate(([-np.inf], (score_vec[1:]+score_vec[:-1])/2))
    return values, thresholds


def f1_curve(target_vec, score_vec):
    """Computes f1 score for decision thresholds between two adjacent scores.

    Args:
        target_vec: binary targets indicating ground truth (num_instances,)
        score_vec: binary classification scores (num_instances,)

    Returns:
        f1_scores: len(set(score_vec)+1) f1 scores
        thresholds: len(set(score_vec)+1) decision thresholds

    >>> target_vec = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    >>> score_vec = np.array([0.6, 0.2, 0.5, 0.4, 0.3, 0.1, 0.7, 0.0, 0.0])
    >>> f1_curve(target_vec, score_vec)
    (array([0.5       , 0.6       , 0.66666667, 0.5       , 0.57142857,
           0.33333333, 0.4       , 0.        , 0.        ]), array([-inf, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65,  inf]))

    """
    return _metric_curve(target_vec, score_vec, metric='f1')


def er_curve(target_vec, score_vec):
    """Given single-class soft scores computes error rate for each threshold between two adjacent score values.

    Args:
        target_vec: binary targets indicating ground truth (num_instances,)
        score_vec: binary classification scores (num_instances,)

    Returns:
        error_rates: len(set(score_vec)+1) f1 error_rates
        thresholds: len(set(score_vec)+1) decision thresholds

    """
    return _metric_curve(target_vec, score_vec, metric='er')


def get_optimal_thresholds(target_mat, score_mat, metric):
    """Given multi-class soft scores returns the optimal threshold for each class w.r.t. a provided metric.

    Args:
        target_mat: multi-hot matrix indicating ground truth
            (num_instances, num_classes)
        score_mat: classification scores for multi-label classification
            (num_instances, num_classes)
        metric: metric to be optimized \in {'f1', 'er'}

    Returns:
        thresholds: opitmal thresholds (num_classes,)
        metric values: optimal metric values


    >>> target_mat = np.array([[1.0], [1.0], [0.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
    >>> score_mat = np.array([[0.6], [0.2], [0.5], [.4], [0.3], [0.1], [0.7], [0.0], [0.0]])
    >>> get_optimal_thresholds(target_mat, score_mat, metric='f1')
    (array([0.15]), array([0.66666667]))
    """
    thresholds = []
    best_values = []
    for label_idx in range(target_mat.shape[-1]):
        target_vec = target_mat[:, label_idx]
        score_vec = score_mat[:, label_idx]
        if metric == 'f1':
            cur_values, cur_thresholds = f1_curve(target_vec, score_vec)
            best_idx = np.argmax(cur_values)
        elif metric == 'er':
            cur_values, cur_thresholds = er_curve(target_vec, score_vec)
            best_idx = np.argmin(cur_values)
        else:
            raise NotImplementedError
        best_values.append(cur_values[best_idx])
        thresholds.append(cur_thresholds[best_idx])
    return np.array(thresholds), np.array(best_values)
