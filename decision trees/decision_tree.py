import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    feature_vector = np.asarray(feature_vector)
    target_vector = np.asarray(target_vector)
    order = np.argsort(feature_vector)
    fv_sorted = feature_vector[order]
    y_sorted = target_vector[order]

    unique_vals = np.unique(fv_sorted)
    if len(unique_vals) == 1:
        return np.array([]), np.array([]), None, None

    mid_mask = fv_sorted[1:] != fv_sorted[:-1]
    thresholds = (fv_sorted[1:] + fv_sorted[:-1]) / 2
    thresholds = thresholds[mid_mask]
    split_positions = np.where(mid_mask)[0] + 1
    if len(thresholds) == 0:
        return np.array([]), np.array([]), None, None

    n = len(y_sorted)
    classes, y_encoded = np.unique(y_sorted, return_inverse=True)
    K = len(classes)

    Y_left_cnt = np.zeros((n, K))
    Y_left_cnt[np.arange(n), y_encoded] = 1
    Y_left_cnt = np.cumsum(Y_left_cnt, axis=0)

    n_left = split_positions
    n_right = n - n_left
    valid_mask = (n_left > 0) & (n_right > 0)
    if not np.any(valid_mask):
        return np.array([]), np.array([]), None, None

    thresholds = thresholds[valid_mask]
    n_left = n_left[valid_mask]
    n_right = n_right[valid_mask]
    left_cnt = Y_left_cnt[split_positions[valid_mask] - 1]
    total_counts = Y_left_cnt[-1]
    right_cnt = total_counts - left_cnt

    p_left = left_cnt / n_left[:, None]
    p_right = right_cnt / n_right[:, None]

    H_left = 1 - np.sum(p_left ** 2, axis=1)
    H_right = 1 - np.sum(p_right ** 2, axis=1)
    ginis = -(n_left / n) * H_left - (n_right / n) * H_right

    best_idx = np.argmax(ginis)
    threshold_best = thresholds[best_idx]
    gini_best = ginis[best_idx]

    return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("Unknown feature type")
        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        if self._max_depth is not None and depth >= self._max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if self._min_samples_split is not None and sub_X.shape[0] < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, split_threshold, best_gini, best_split_mask = None, None, None, None

        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]

            if feature_type == "real":
                feature_values = sub_X[:, feature]
                mapping = None
            else:
                categories = sub_X[:, feature]
                unique_cats, cat_indices = np.unique(categories, return_inverse=True)

                unique_labels, label_encoded = np.unique(sub_y, return_inverse=True)
                category_sums = np.bincount(cat_indices, weights=label_encoded, minlength=len(unique_cats))
                category_counts = np.bincount(cat_indices, minlength=len(unique_cats))
                category_means = category_sums / np.maximum(category_counts, 1)

                sort_order = np.argsort(category_means)
                sorted_categories = unique_cats[sort_order]
                category_mapping = {cat: idx for idx, cat in enumerate(sorted_categories)}

                feature_values = np.array([category_mapping[cat] for cat in categories])
                mapping = category_mapping

            if len(np.unique(feature_values)) < 2:
                continue

            _, _, split_value, gini = find_best_split(feature_values, sub_y)

            if split_value is None:
                continue

            candidate_split = feature_values < split_value

            if self._min_samples_leaf is not None:
                left_samples = np.sum(candidate_split)
                right_samples = np.sum(~candidate_split)
                if left_samples < self._min_samples_leaf or right_samples < self._min_samples_leaf:
                    continue

            if best_gini is None or gini > best_gini:
                feature_best = feature
                best_gini = gini
                best_split_mask = candidate_split

                if feature_type == "real":
                    split_threshold = split_value
                else:
                    split_threshold = [cat for cat, idx in mapping.items() if idx < split_value]

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best

        if self._feature_types[feature_best] == "real":
            node["threshold"] = split_threshold
        else:
            node["categories_split"] = split_threshold

        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[best_split_mask], sub_y[best_split_mask],
                       node["left_child"], depth + 1)
        self._fit_node(sub_X[~best_split_mask], sub_y[~best_split_mask],
                       node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature_idx = node["feature_split"]
        feature_value = x[feature_idx]

        if self._feature_types[feature_idx] == "real":
            if feature_value < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            if feature_value in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree, 0)

    def predict(self, X):
        predictions = []
        for sample in X:
            predictions.append(self._predict_node(sample, self._tree))
        return np.array(predictions)
