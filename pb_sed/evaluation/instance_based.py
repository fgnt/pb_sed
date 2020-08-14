import numpy as np


def tp_fp_tn_fn(target_mat, decision_mat, reduce_axis=None):
    """

    Args:
        target_mat: multi-hot matrix indicating ground truth events/labels
            (num_frames, num_labels)
        decision_mat: multi-hot matrix indicating detected events/labels
            (N, num_frames, num_labels)
        reduce_axis:

    Returns:

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
    """
    Computes frame-based f-score

    Args:
        target_mat: multi-hot matrix indicating ground truth events/labels
            (num_frames, num_labels)
        decision_mat: multi-hot matrix indicating detected events/labels
            (N, num_frames, num_labels)
        event_wise:
        beta:

    Returns: frame-based f-scores  (N,)

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
    """

    Args:
        target_mat: multi-hot matrix indicating ground truth events/labels
            (num_frames times num_labels)
        decision_mat: multi-hot matrix indicating detected events/labels
            (num_frames times num_labels)
        reduce_axis:

    Returns:

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
    """

    Args:
        target_mat: multi-hot matrix indicating ground truth events/labels
            (num_frames times num_labels)
        decision_mat: multi-hot matrix indicating detected events/labels
            (num_frames times num_labels)
        event_wise:

    Returns:

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


def get_optimal_threshold(target_mat, score_mat, metric):
    """

    Args:
        target_mat:
        score_mat:
        metric:

    Returns:

    >>> target_mat = np.array([[1.0], [1.0], [0.0], [1.0], [0.0], [0.0], [0.0]])
    >>> score_mat = np.array([[0.6], [0.2], [0.5], [.4], [0.3], [0.1], [0.7]])
    >>> get_optimal_threshold(target_mat, score_mat, metric='fscore')
    (array([0.15]), array([0.66666667]))
    """
    threshold = []
    values = []
    for label_idx in range(target_mat.shape[-1]):
        cur_targets = target_mat[:, label_idx]
        cur_scores = score_mat[:, label_idx]
        sort_indices = np.argsort(cur_scores)
        cur_scores = np.concatenate((cur_scores[sort_indices], [np.inf]))
        cur_targets = cur_targets[sort_indices]
        edge_detection = np.correlate(2*cur_targets-1, [-1, 1], mode='full')
        edge_indices = np.argwhere(edge_detection > 0.5).flatten()
        _, valid_idx = np.unique(cur_scores[edge_indices], return_index=True)
        edge_indices = edge_indices[valid_idx]
        tps = np.cumsum(np.concatenate((cur_targets, [0]))[::-1])[::-1][edge_indices]
        n_sys = len(cur_scores) - 1 - edge_indices
        n_ref = tps[0]
        if metric == 'fscore':
            p = tps / np.maximum(n_sys, 1)
            r = tps / np.maximum(n_ref, 1)
            value = 2*p*r / (p+r+1e-15)
            idx = np.argmax(value)
        elif metric == 'er':
            i = n_sys - tps
            d = n_ref - tps
            value = (i+d)/n_ref
            idx = np.argmin(value)
        else:
            raise NotImplementedError
        values.append(value[idx])
        idx = edge_indices[idx]
        threshold.append(
            -np.inf if idx == 0 else (cur_scores[idx-1]+cur_scores[idx])/2
        )
    return np.array(threshold), np.array(values)


def get_thresholds(target_mat, score_mat):
    """

    Args:
        target_mat:
        score_mat:

    Returns:

    >>> target_mat = np.array([[1.0], [1.0], [0.0], [1.0], [0.0], [0.0], [0.0]])
    >>> score_mat = np.array([[0.6], [0.2], [0.5], [.4], [0.3], [0.1], [0.7]])
    >>> get_thresholds(target_mat, score_mat)
    [array([0.15, 0.35, 0.55,  inf])]
    """
    candidate_thresholds = []
    for label_idx in range(target_mat.shape[-1]):
        cur_targets = target_mat[:, label_idx]
        cur_scores = score_mat[:, label_idx]
        sort_indices = np.argsort(cur_scores)
        cur_scores = cur_scores[sort_indices]
        cur_targets = cur_targets[sort_indices]
        edge_detection = np.correlate(2*cur_targets-1, [-1, 1], mode='full')
        edge_indices = np.argwhere(edge_detection > 0.5).flatten()
        cur_candidate_thresholds = np.array([
            -np.inf if i == 0 else np.inf if i == len(cur_scores)
            else (cur_scores[i-1]+cur_scores[i])/2
            for i in edge_indices
        ])
        candidate_thresholds.append(cur_candidate_thresholds)
    return candidate_thresholds
