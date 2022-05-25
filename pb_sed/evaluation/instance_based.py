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
    n_ref = np.maximum(np.sum(target_mat, axis=reduce_axis), 1)
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


def lwlrap(target_mat, score_mat):
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
    if not target_mat.any():
        return 0.0, np.zeros(target_mat.shape[-1])
    assert score_mat.ndim == 2, score_mat.shape
    assert target_mat.shape == score_mat.shape
    pos_class_indices, precision_at_hits = positive_class_precisions(
        target_mat, score_mat
    )
    lwlrap_score, per_class_lwlrap, weight_per_class = lwlrap_from_precisions(
        precision_at_hits, pos_class_indices, num_classes=target_mat.shape[1]
    )
    return lwlrap_score, per_class_lwlrap


def _positives_curve(targets, scores):
    """

    Args:
        targets:
        scores:

    Returns:
        metric values:
        thresholds:
    """
    sort_indices = np.argsort(scores)
    scores = np.concatenate((scores[sort_indices], [np.inf]))
    targets = targets[sort_indices]

    tps = np.cumsum(np.concatenate((targets, [0]))[::-1])[::-1]
    n_sys = len(scores) - np.arange(len(scores)) - 1

    scores, index, inverse = np.unique(scores, return_index=True, return_inverse=True)
    tps = tps[index][inverse]
    n_sys = n_sys[index][inverse]
    thresholds = np.concatenate(([-np.inf], (scores[1:] + scores[:-1]) / 2))
    return thresholds[inverse], n_sys, tps


def fscore_curve(targets, scores, beta=1., tp_bias=0, n_ref_bias=0, n_pos_bias=0):
    """Computes fscore for decision thresholds between two adjacent scores.

    Args:
        targets: binary targets indicating ground truth (num_instances,)
        scores: binary classification scores (num_instances,)

    Returns:
        f1_scores: len(set(score_vec)+1) f1 scores
        thresholds: len(set(score_vec)+1) decision thresholds

    >>> targets = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    >>> scores = np.array([0.6, 0.2, 0.5, 0.4, 0.3, 0.1, 0.7, 0.0, 0.0])
    >>> fscore_curve(targets, scores)
    (array([-inf, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65,  inf]), array([0.5       , 0.6       , 0.66666667, 0.5       , 0.57142857,
           0.33333333, 0.4       , 0.        , 0.        ]), array([0.33333333, 0.42857143, 0.5       , 0.4       , 0.5       ,
           0.33333333, 0.5       , 0.        , 0.        ]), array([1.        , 1.        , 1.        , 0.66666667, 0.66666667,
           0.33333333, 0.33333333, 0.        , 0.        ]))
    >>> targets = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    >>> scores = np.array([0.6, 0.2, 0.5, 0.4, 0.3, 0.1, 0.7, 0.0, 0.0])
    >>> fscore_curve(np.stack([targets, targets]).T, np.stack([scores, scores]).T)
    (array([-inf, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65,  inf]), array([0.5       , 0.6       , 0.66666667, 0.5       , 0.57142857,
           0.33333333, 0.4       , 0.        , 0.        ]), array([0.33333333, 0.42857143, 0.5       , 0.4       , 0.5       ,
           0.33333333, 0.5       , 0.        , 0.        ]), array([1.        , 1.        , 1.        , 0.66666667, 0.66666667,
           0.33333333, 0.33333333, 0.        , 0.        ]))

    """
    assert 0 < scores.ndim <= 2, scores.shape
    assert scores.shape == targets.shape, (scores.shape, targets.shape)
    if scores.ndim == 2:
        thresholds, f, p, r = list(zip(*[fscore_curve(t, s) for t, s in zip(targets.T, scores.T)]))
        return np.array(thresholds).T, np.array(f).T, np.array(p).T, np.array(r).T
    thresholds, n_pos, tps = _positives_curve(targets, scores)
    n_ref = tps[0]
    p = (tps + tp_bias) / np.maximum(n_pos + n_pos_bias, 1)
    r = (tps + tp_bias) / np.maximum(n_ref + n_ref_bias, 1)
    f = (1 + beta**2) * p * r / (beta**2*p+r+1e-18)
    return thresholds, f, p, r


def get_best_fscore_thresholds(targets, scores, beta=1., min_precision=0., min_recall=0., tp_bias=0, n_ref_bias=0, n_pos_bias=0):
    """
    >>> target_mat = np.array([[1.0], [1.0], [0.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
    >>> score_mat = np.array([[0.6], [0.2], [0.5], [.4], [0.3], [0.1], [0.7], [0.0], [0.0]])
    >>> get_best_fscore_thresholds(target_mat, score_mat)
    (array([0.15]), array([0.66666667]), array([0.5]), array([1.]))
    >>> get_best_fscore_thresholds(target_mat.flatten(), score_mat.flatten())
    (0.15000000000000002, 0.6666666666666666, 0.5, 1.0)
    >>> get_best_fscore_thresholds(target_mat.flatten(), score_mat.flatten(), min_precision=.5)
    (0.15000000000000002, 0.6666666666666666, 0.5, 1.0)
    >>> get_best_fscore_thresholds(target_mat.flatten(), score_mat.flatten(), min_precision=.51)
    (inf, 0.0, 0.0, 0.0)
    >>> get_best_fscore_thresholds(target_mat.flatten(), score_mat.flatten(), min_recall=1.0)
    (0.15000000000000002, 0.6666666666666666, 0.5, 1.0)
    >>> get_best_fscore_thresholds(target_mat.flatten(), score_mat.flatten(), tp_bias=1, n_ref_bias=1, n_pos_bias=1)
    (0.15000000000000002, 0.6666666666666666, 0.5, 1.0)
    """
    thresholds, f, p, r = fscore_curve(targets, scores, beta, tp_bias=tp_bias, n_ref_bias=n_ref_bias, n_pos_bias=n_pos_bias)
    assert min_precision == 0. or min_recall == 0.
    f[p < min_precision] = 0.
    f[r < min_recall] = 0.
    best_idx = (-1-np.argmax(f[::-1], axis=0))
    if thresholds.ndim == 1:
        return thresholds[best_idx], f[best_idx], p[best_idx], r[best_idx]
    elif thresholds.ndim == 2:
        class_idx = np.arange(targets.shape[1])
        return thresholds[best_idx, class_idx], f[best_idx, class_idx], p[best_idx, class_idx], r[best_idx, class_idx]
    else:
        raise AssertionError('ndim must be less equal 2.')


def er_curve(targets, scores):
    """Given single-class soft scores computes error rate for each threshold between two adjacent score values.

    Args:
        targets: binary targets indicating ground truth (num_instances,)
        scores: binary classification scores (num_instances,)

    Returns:
        error_rates: len(set(score_vec)+1) f1 error_rates
        thresholds: len(set(score_vec)+1) decision thresholds

    """
    assert 0 < scores.ndim <= 2, scores.shape
    assert scores.shape == targets.shape, (scores.shape, targets.shape)
    if scores.ndim == 2:
        thresholds, f, p, r = list(zip(*[er_curve(t, s) for t, s in zip(targets.T, scores.T)]))
        return np.array(thresholds).T, np.array(f).T, np.array(p).T, np.array(r).T
    thresholds, n_pos, tps = _positives_curve(targets, scores)
    n_ref = tps[0]
    i = n_pos - tps
    d = n_ref - tps
    er = (i+d) / np.maximum(n_ref, 1)
    ir = i / np.maximum(n_ref, 1)
    dr = d / np.maximum(n_ref, 1)
    return thresholds, er, ir, dr


def get_best_er_thresholds(targets, scores, max_insertion_rate=None, max_deletion_rate=None):
    """
    >>> target_mat = np.array([[1.0], [1.0], [0.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
    >>> score_mat = np.array([[0.6], [0.2], [0.5], [.4], [0.3], [0.1], [0.7], [0.0], [0.0]])
    >>> get_best_er_thresholds(target_mat, score_mat)
    (array([inf]), array([1.]), array([0.]), array([1.]))
    >>> get_best_er_thresholds(target_mat.flatten(), score_mat.flatten())
    (inf, 1.0, 0.0, 1.0)
    """
    thresholds, er, ir, dr = er_curve(targets, scores)
    if max_insertion_rate is not None:
        er[ir > max_insertion_rate] = np.inf
    if max_deletion_rate is not None:
        er[dr > max_deletion_rate] = np.inf
    best_idx = (-1-np.argmin(er[::-1], axis=0))
    if thresholds.ndim == 1:
        return thresholds[best_idx], er[best_idx], ir[best_idx], dr[best_idx]
    elif thresholds.ndim == 2:
        class_idx = np.arange(targets.shape[1])
        return thresholds[best_idx, class_idx], er[best_idx, class_idx], ir[best_idx, class_idx], dr[best_idx, class_idx]
    else:
        raise AssertionError('ndim must be less equal 2.')
