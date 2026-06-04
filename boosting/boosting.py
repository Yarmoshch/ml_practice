from __future__ import annotations

from collections import defaultdict
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import roc_auc_score
from typing import Optional, Iterable
from tqdm.auto import tqdm
from sklearn.base import ClassifierMixin
from sklearn.tree import DecisionTreeClassifier


class Boosting(ClassifierMixin):
    def __init__(
            self,
            base_model_class=DecisionTreeRegressor,
            base_model_params: Optional[dict] = None,
            n_estimators: int = 20,
            learning_rate: float = 0.05,
            random_state: int | None = None,
            verbose: bool = False,
            early_stopping_rounds: int | None = 0,
            cat_features: list[int] | None = None,
            cat_smoothing: float = 20.0,
            cat_add_count: bool = True,
            cat_count_log: bool = True,
            subsample: float = 1.0,
            bagging_temperature: float = 1.0,
            bootstrap_type: str | None = "Bernoulli",
            rsm: float = 1.0,
            ordered_counters: bool = False,
            quantization_type: str | None = None,
            nbins: int = 255,
            goss: bool = False,
            goss_k: float = 0.2,
    ):
        super().__init__()
        self.base_model_class = base_model_class
        self.base_model_params = base_model_params or {}
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.verbose = verbose

        self.models, self.gammas, self.model_features = [], [], []
        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.mean(np.log(self.sigmoid(y * z) + 1e-15))
        self.loss_derivative = lambda y, z: y / (1.0 + np.exp(y * z))

        self.early_stopping_rounds = early_stopping_rounds
        self.cat_features = list(cat_features) if cat_features else []
        self.cat_smoothing = float(cat_smoothing)
        self.cat_add_count = bool(cat_add_count)
        self.cat_count_log = bool(cat_count_log)

        self._cat_stats, self._cat_prior = {}, None
        self.subsample, self.bagging_temperature = float(subsample), float(bagging_temperature)
        self.bootstrap_type = bootstrap_type
        self._rng = np.random.default_rng(self.random_state)
        self.rsm, self.ordered_counters = float(rsm), ordered_counters
        self.quantization_type, self.nbins = quantization_type, int(nbins)
        self.n_features_ = None
        self.goss = goss
        self.goss_k = goss_k

    def _fit_counters_ordered(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Реализация Ordered Target Statistics (Bonus 3.0).
        Значение для i-го объекта считается только по объектам 0...i-1.
        """
        y_binary = (y == 1).astype(float)
        self._cat_prior = np.mean(y_binary)
        n, p = X.shape

        self._fit_counters(X, y)

        results = []
        for j in range(p):
            if j not in self.cat_features:
                results.append(X[:, j].astype(float))
                continue

            col_values = X[:, j]
            res_ctr = np.zeros(n)
            res_cnt = np.zeros(n)

            # Словари для хранения накопленных сумм и количеств (running totals)
            running_stats = defaultdict(lambda: [0.0, 0.0])

            for i in range(n):
                val = col_values[i]
                cur_sum, cur_count = running_stats[val]

                # считаем CTR на основе накопленных данных (до текущего i)
                res_ctr[i] = (cur_sum + self.cat_smoothing * self._cat_prior) / (cur_count + self.cat_smoothing)
                res_cnt[i] = cur_count

                # обновляем статистики для следующих объектов
                running_stats[val][0] += y_binary[i]
                running_stats[val][1] += 1

            results.append(res_ctr)
            if self.cat_add_count:
                count_feature = np.log1p(res_cnt) if self.cat_count_log else res_cnt
                results.append(count_feature)

        return np.column_stack(results).astype(float)

    def _fit_counters(self, X: np.ndarray, y: np.ndarray):
        """Обычный расчет счетчиков по всей выборке (для теста/валидации)."""
        y_f = (y == 1).astype(float)
        self._cat_prior = np.mean(y_f)
        for j in self.cat_features:
            keys, inv = np.unique(X[:, j], return_inverse=True)
            counts = np.bincount(inv)
            pos = np.bincount(inv, weights=y_f)
            self._cat_stats[j] = {
                "uniq": keys,
                "ctr": (pos + self.cat_smoothing * self._cat_prior) / (counts + self.cat_smoothing),
                "cnt": counts.astype(float)
            }

    def _transform_counters(self, X: np.ndarray) -> np.ndarray:
        """Применяет уже обученные счетчики к новым данным (X_test, X_valid)."""
        if not self.cat_features:
            return X.astype(float)

        cols = []
        for j in range(X.shape[1]):
            if j not in self.cat_features:
                cols.append(X[:, j].astype(float))
                continue

            st = self._cat_stats[j]
            # Быстрый поиск индексов категорий
            idx = np.searchsorted(st["uniq"], X[:, j])
            valid = (idx < len(st["uniq"]))
            valid[valid] &= (st["uniq"][idx[valid]] == X[:, j][valid])

            res_ctr = np.full(X.shape[0], self._cat_prior)
            res_ctr[valid] = st["ctr"][idx[valid]]
            cols.append(res_ctr)

            if self.cat_add_count:
                c = np.zeros(X.shape[0])
                c[valid] = st["cnt"][idx[valid]]
                cols.append(np.log1p(c) if self.cat_count_log else c)

        return np.column_stack(cols).astype(float)

    def _raw_predict(self, X: np.ndarray) -> np.ndarray:
        if not self.models:
            return np.zeros(X.shape[0])
        return sum(self.learning_rate * g * m.predict(X[:, f])
                   for m, g, f in zip(self.models, self.gammas, self.model_features))

    def partial_fit(self, X: np.ndarray, y: np.ndarray):
        n = X.shape[0]
        logits = self._raw_predict(X)
        targets = self.loss_derivative(y, logits)

        # RSM: выбор случайных признаков
        f_total = X.shape[1]
        feat_indices = np.where(self._rng.random(f_total) <= self.rsm)[0]
        if not feat_indices.size:
            feat_indices = np.array([self._rng.integers(0, f_total)])

        X_fit, y_fit, weights = None, None, None

        if getattr(self, "goss", False):
            # логика GOSS
            abs_grads = np.abs(targets)
            top_k = int(self.goss_k * n)

            sorted_indices = np.argsort(abs_grads)[::-1]
            group_a = sorted_indices[:top_k]
            group_b = sorted_indices[top_k:]

            # Сэмплируем из "маленьких" градиентов
            n_b_sample = int(self.subsample * len(group_b))
            sampled_b = self._rng.choice(group_b, n_b_sample, replace=False) if n_b_sample > 0 else np.array([], dtype=int)

            idx = np.concatenate([group_a, sampled_b])
            X_fit, y_fit = X[idx][:, feat_indices], targets[idx]

            # веса для компенсации смещения
            weights = np.ones(len(idx))
            if n_b_sample > 0:
                weights[top_k:] = (1 - self.goss_k) / self.subsample
        else:
            # обычный бутстрап
            X_fit, y_fit = X[:, feat_indices], targets
            if self.bootstrap_type == "Bernoulli":
                mask = self._rng.random(n) <= self.subsample
                if not mask.any(): mask[0] = True
                X_fit, y_fit = X_fit[mask], y_fit[mask]
            elif self.bootstrap_type == "Bayesian":
                weights = (-np.log(self._rng.random(n) + 1e-12)) ** self.bagging_temperature

        model = self.base_model_class(**{**self.base_model_params, "random_state": self.random_state})
        model.fit(X_fit, y_fit, sample_weight=weights)

        # подбор гаммы на всей выборке
        new_preds = model.predict(X[:, feat_indices])
        gamma = self.find_optimal_gamma(y, logits, self.learning_rate * new_preds)

        self.models.append(model)
        self.gammas.append(gamma)
        self.model_features.append(feat_indices)
        self.history["train_loss"].append(self.loss_fn(y, logits + gamma * self.learning_rate * new_preds))

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            eval_set: tuple[np.ndarray, np.ndarray] | None = None,
            use_best_model: bool = False):

        if self.cat_features:
            # Используем Ordered Counters, если флаг включен
            X_train = self._fit_counters_ordered(X_train, y_train) if self.ordered_counters \
                      else (self._fit_counters(X_train, y_train) or self._transform_counters(X_train))

            if eval_set:
                X_v, y_v = eval_set
                eval_set = (self._transform_counters(X_v), y_v)

        if self.quantization_type is not None:
            self._fit_quantization(X_train, y_train)
            X_train = self._transform_quantization(X_train)
            if eval_set is not None:
                x_v, y_v = eval_set
                eval_set = (self._transform_quantization(x_v), y_v)

        if self.quantization_type is not None:
            self._fit_quantization(X_train)
            X_train = self._transform_quantization(X_train)
            if eval_set is not None:
                x_v, y_v = eval_set
                eval_set = (self._transform_quantization(x_v), y_v)

        X_val, y_val = eval_set if eval_set else (None, None)
        best_l, best_s, p = float("inf"), 0, 0

        for i in tqdm(range(self.n_estimators)) if self.verbose else range(self.n_estimators):
            self.partial_fit(X_train, y_train)
            if X_val is not None:
                cur_l = self.loss_fn(y_val, self._raw_predict(X_val))
                self.history["valid_loss"].append(cur_l)
                if cur_l < best_l:
                    best_l, best_s, p = cur_l, i, 0
                else:
                    p += 1
                if self.early_stopping_rounds and p >= self.early_stopping_rounds: break

        if use_best_model:
            self.models, self.gammas, self.model_features = self.models[:best_s+1], self.gammas[:best_s+1], self.model_features[:best_s+1]

        for k in self.history: self.history[k] = np.array(self.history[k])
        self.n_features_ = X_train.shape[1]

    def _fit_quantization(self, X: np.ndarray, y: np.ndarray = None):
        self._quantile_bins = {}

        for j in range(X.shape[1]):
            col = X[:, j].reshape(-1, 1)

            if self.quantization_type == 'uniform':
                low, high = col.min(), col.max()
                self._quantile_bins[j] = np.linspace(low, high, self.nbins + 1)[1:-1]

            elif self.quantization_type == 'quantile':
                qs = np.linspace(0, 1, self.nbins + 1)[1:-1]
                self._quantile_bins[j] = np.unique(np.quantile(col, qs))

            elif self.quantization_type == 'piecewise':
                # Обучаем дерево небольшой глубины, чтобы найти nbins порогов
                max_depth = int(np.ceil(np.log2(self.nbins)))
                dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
                dt.fit(col, y)

                # Извлекаем все пороги (thresholds), которые дерево использовало для сплитов
                thresholds = dt.tree_.threshold[dt.tree_.feature != -2]
                self._quantile_bins[j] = np.sort(np.unique(thresholds))

    def _transform_quantization(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_quantile_bins") or self.quantization_type is None:
            return X

        X_q = X.copy().astype(float)
        for j, bins in self._quantile_bins.items():
            # превращает значения в индексы бинов
            X_q[:, j] = np.digitize(X[:, j], bins)
        return X_q

    def predict_proba(self, X: np.ndarray):
        if self.cat_features and X.dtype == object:
            X = self._transform_counters(X)
        p1 = self.sigmoid(self._raw_predict(X))
        return np.column_stack([1 - p1, p1])

    def find_optimal_gamma(self, y: np.ndarray, old_predictions: np.ndarray, new_predictions: np.ndarray) -> float:
        candidates = np.linspace(0, 1, 100)
        return candidates[np.argmin([self.loss_fn(y, old_predictions + g * new_predictions) for g in candidates])]

    def score(self, X: np.ndarray, y: np.ndarray):
        return roc_auc_score(y == 1, self.predict_proba(X)[:, 1])