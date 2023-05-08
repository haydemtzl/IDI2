import numpy as np
from collections import Counter

class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, class_weight=None):
        """
        Inicializa el clasificador.

        Args:
            max_depth (int): Profundidad máxima del árbol. Si es None, el árbol se expandirá hasta que todas las hojas
                contengan menos de min_samples_split muestras.
            min_samples_split (int): Número mínimo de muestras necesarias para dividir un nodo interno.
            min_samples_leaf (int): Número mínimo de muestras necesarias en cada hoja.
            class_weight (dict): Diccionario que asigna un peso a cada clase, para tratar el desbalanceo de clases. Si
                es None, se asigna un peso de 1 a cada clase.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight

        if class_weight is not None:
            self.class_weight = {k: v/sum(class_weight.values()) for k, v in class_weight.items()}

    def fit(self, X, y):
        """
        Entrena el árbol de decisión.

        Args:
            X (np.array): Matriz de características (m x n).
            y (np.array): Vector de etiquetas (m x 1).
        """
        # Verificar que X y y tengan la misma cantidad de filas
        assert X.shape[0] == y.shape[0], "X and y must have the same number of rows"

        self.n_classes_ = len(np.unique(y))
        self.n_features_ = X.shape[1]
        self.tree_ = []  # Lista de nodos del árbol
        self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Caso base: se ha alcanzado la profundidad máxima o no hay suficientes muestras para continuar dividiendo
        if depth == self.max_depth or n_samples < self.min_samples_split or n_labels == 1 or y.size == 0:
            # Si y está vacío, agregamos un nodo hoja con una etiqueta aleatoria
            if y.size == 0:
                self.tree_.append({"is_leaf": True, "label": np.random.choice(range(self.n_classes_)), "depth": depth})
            else:
                # Seleccionamos la clase mayoritaria como etiqueta de la hoja
                self.tree_.append({"is_leaf": True, "label": Counter(y.tolist()).most_common(1)[0][0], "depth": depth})
            return

        # En otro caso, buscamos la mejor división
        split_feature, split_value = self._best_split(X, y)
        print("aquí")
        print(split_feature, split_value)
        if split_feature is None:
            # No se encontró ninguna división que mejorara la pureza de las hojas
            self.tree_.append({"is_leaf": True, "label": Counter(y.tolist()).most_common(1)[0][0], "depth": depth})
            return

        # Creamos el nodo interno y lo añadimos al árbol
        left_idx = X.iloc[:, split_feature] < split_value
        right_idx = X.iloc[:, split_feature] >= split_value
        node = {"is_leaf": False, "split_feature": split_feature, "split_value": split_value, "depth": depth}
        self.tree_.append(node)

        # Creamos las ramas izquierda y derecha del árbol
        self._grow_tree(X[left_idx], y[left_idx], depth+1)
        self._grow_tree(X[right_idx], y[right_idx], depth+1)


    # def _grow_tree(self, X, y, depth=0):
    #     n_samples, n_features = X.shape
    #     n_labels = len(np.unique(y))
    #
    #     # Caso base: se ha alcanzado la profundidad máxima o no hay suficientes muestras para continuar dividiendo
    #     if depth == self.max_depth or n_samples < self.min_samples_split or n_labels == 1 or y.empty:
    #         # Si y está vacío, agregamos un nodo hoja con una etiqueta aleatoria
    #         if y.empty:
    #             self.tree_.append({"is_leaf": True, "label": 1, "depth": depth})
    #         else:
    #             # Seleccionamos la clase mayoritaria como etiqueta de la hoja
    #             self.tree_.append({"is_leaf": True, "label": Counter(y).most_common(1)[0][0], "depth": depth})
    #         return
    #
    #     # En otro caso, buscamos la mejor división
    #     split_feature, split_value = self._best_split(X, y)
    #     if split_feature is None:
    #         # No se encontró ninguna división que mejorara la pureza de las hojas
    #         self.tree_.append({"is_leaf": True, "label": Counter(y).most_common(1)[0][0], "depth": depth})
    #         return
    #
    #     # Creamos el nodo interno y lo añadimos al árbol
    #     left_idx = X.iloc[:, split_feature] < split_value
    #     right_idx = X.iloc[:, split_feature] >= split_value
    #     node = {"is_leaf": False, "split_feature": split_feature, "split_value": split_value, "depth": depth}
    #     self.tree_.append(node)
    #
    #     # Creamos las ramas izquierda y derecha del árbol
    #     self._grow_tree(X[left_idx], y[left_idx], depth+1)
    #     self._grow_tree(X[right_idx], y[right_idx], depth+1)
    #


    def _best_split(self, X, y):
        """Encuentra la mejor división para un nodo interno del árbol."""
        best_feature, best_value, best_gain = None, None, -np.inf

        n_samples = X.shape[0]

        # Calculamos la impureza actual del nodo
        current_score = self._impurity_score(y)

        # Iteramos sobre todas las posibles divisiones
        for feature_idx in range(X.shape[1]):
            #print(feature_idx)
            feature_values = X.iloc[:, feature_idx]
            for split_value in np.unique(feature_values):
                left_idx = feature_values < split_value
                right_idx = feature_values >= split_value
                if len(left_idx) == 0 or len(right_idx) == 0:
                    continue
                left_labels, right_labels = y[left_idx], y[right_idx]

                # Calculamos la impureza y el peso de cada hoja después de la división
                left_score = self._impurity_score(left_labels)
                right_score = self._impurity_score(right_labels)
                left_weight, right_weight = len(left_labels) / n_samples, len(right_labels) / n_samples

                # Calculamos la ganancia de información (Information Gain) de la división actual
                info_gain = self._information_gain(current_score, left_labels, right_labels)
                if info_gain > best_gain:
                    best_gain = info_gain
                    best_feature = feature_idx
                    best_value = split_value

        if best_gain == -np.inf:
            return None, None
        else:
            return best_feature, best_value

    def _impurity_score(self, y):
        """
        Calcula la impureza de un nodo.

        Args:
            y (np.array): Vector de etiquetas (m x 1).

        Returns:
            float: Impureza del nodo.
        """
        if self.class_weight is not None:
            # Si se especificó un peso para cada clase, se calcula la entropía ponderada
            class_counts = np.bincount(y)
            class_weights = [self.class_weight.get(i, 1) for i in range(len(class_counts))]
            p = class_counts / len(y)
            impurity = -np.sum([w * self._entropy(p_i) for w, p_i in zip(class_weights, p)])
        else:
            # Si no se especificó un peso para cada clase, se calcula la entropía normal
            counts = np.bincount(y)
            p = counts / len(y)
            impurity = self._entropy(p)

        return impurity

    def _entropy(self, p):
        """
        Calcula la entropía de un nodo.

        Args:
            p (np.array): Vector de probabilidades de cada clase.

        Returns:
            float: Entropía del nodo.
        """
        return -np.sum(np.where(p != 0, p * np.log2(p), 0))

    def _information_gain(self, parent, left_child, right_child):
        """
        Calcula la ganancia de información de una división de nodos.

        Args:
            parent (float): Impureza del nodo padre.
            left_child (np.array): Vector de etiquetas del nodo hijo izquierdo.
            right_child (np.array): Vector de etiquetas del nodo hijo derecho.

        Returns:
            float: Ganancia de información de la división.
        """
        n = len(left_child) + len(right_child)
        pl = len(left_child) / n
        pr = len(right_child) / n
        gain = parent - (pl * self._impurity_score(left_child) + pr * self._impurity_score(right_child))
        return gain

    def predict(self, X):
        """
        Predice las etiquetas para los datos X.

        Args:
            X (np.array): Matriz de características (m x n).

        Returns:
            np.array: Vector de etiquetas (m x 1)."""

        predicted_labels = []
        X = X.to_numpy()
        print(X)
        for sample in X:
            node = 0  # Comenzamos en la raíz del árbol
            while not self.tree_[node]["is_leaf"]:
                print("si entré")
                if sample[self.tree_[node]["split_feature"]] < self.tree_[node]["split_value"]:
                    node = node * 2 + 1  # Nos movemos hacia la rama izquierda
                else:
                    node = node * 2 + 2  # Nos movemos hacia la rama derecha
            predicted_labels.append(self.tree_[node]["label"])
        print(predicted_labels)
        return np.array(predicted_labels)

    def score(self, X_test, y_true):
        """
        Calcula la precisión del modelo.

        Args:
            X_test (np.array): Matriz de características de prueba (m x n).
            y_true (np.array): Vector de etiquetas verdaderas de prueba (m x 1).

        Returns:
            float: Precisión del modelo.
        """
        y_pred = self.predict(X_test)
        accuracy = np.sum(y_pred == y_true) / len(y_true)
        return accuracy
