import os
from typing import List, Tuple
import numpy as np
from sklearn.metrics import roc_curve, auc
from scipy.ndimage import convolve
import cv2


############################################################################
#                          MSLD IMPLEMENTATION                             #
############################################################################
class MSLD:
    """
    Classe implémentant l'algorithme de MSLD, ainsi que différents outils de
    mesure de performances.

    Les attributs de cette classe sont:
        W: Taille de la fenêtre.
        L: Vecteur contenant les longueurs des lignes à détecter.
        n_orientation: Nombre d'orientation des lignes à détecter.
        threshold: Seuil de segmentation (à apprendre).
        line_detectors_masks: Masques pour la détection des lignes pour chaque valeur de L et chaque valeur de
            n_orientation.
        avg_mask: Masque moyenneur de taille W x W.
    """

    def __init__(self, W, L, orientation):
        """Constructeur qui initialise un objet de type MSLD. Cette méthode est appelée par
        >>> msld = MSLD(W=..., L=..., n_orientation=...)

        Args:
            W (int): Taille de la fenêtre (telle que définie dans l'article).
            L (List[int]): Une liste contenant les valeurs des longueurs des lignes qui seront détectées par la MSLD.
            orientation (int | list[int]): Nombre d'orientations des lignes à détecter.
        """
        self.W = W
        self.L = L
        self.threshold = 0.56

        self.avg_mask = np.ones((W, W)) / (W**2)

        self.line_detectors_masks = {}
        for l in L:
            # On calcule le détecteur de ligne initial de taille l (les dimensions du masque sont lxl).
            line_detector = np.zeros((l, l))
            line_detector[:, l // 2] = 1

            # On initialise la liste des n_orientation masques de taille lxl.
            line_detectors_masks = [line_detector / l]
            if isinstance(orientation, int):
                angle_step = 180 / orientation
                orientation = np.arange(angle_step, 180, angle_step)

            for angle in orientation:
                # On effectue n_orientation-1 rotations du masque line_detector.
                # Pour un angle donné, la rotation sera effectué par

                r = cv2.getRotationMatrix2D((l // 2, l // 2), angle, 1)
                rotated_mask = cv2.warpAffine(line_detector, r, (l, l))
                line_detectors_masks.append(rotated_mask / rotated_mask.sum())

            # On assemble les n_orientation masques ensemble:
            self.line_detectors_masks[l] = np.stack(line_detectors_masks, axis=2)
            self.n_orientation = len(orientation)

    def basicLineDetector(self, grey_lvl, L):
        """Applique l'algorithme Basic Line Detector sur la carte d'intensité grey_lvl avec des lignes de longueurs L.

        Args:
            grey_lvl (np.ndarray): Carte d'intensité 2D avec dtype float sur laquelle est appliqué le BLD.
            L (int): Longueur des lignes (on supposera que L est présent dans self.L et donc que
                self.line_detectors_masks[L] existe).

        Returns:
            R (np.ndarray): Carte de réponse 2D en float du Basic Line Detector.
        """
        line_detector = self.line_detectors_masks[L]

        Iavg = convolve(grey_lvl, self.avg_mask, mode="nearest")

        Imax = np.zeros(grey_lvl.shape)
        for i in range(self.n_orientation):
            Iline = convolve(grey_lvl, line_detector[:, :, i], mode="nearest")
            Imax = np.maximum(Iline, Imax)

        R = Imax - Iavg

        R = (R - R.mean()) / R.std()

        return R

    def multiScaleLineDetector(self, imap):
        """Applique l'algorithme de Multi-Scale Line Detector et combine les réponses des BLD pour obtenir la carte
        d'intensité de l'équation 4 de la section 3.3 Combination Method.

        Args:
            image (np.ndarray): Image RGB aux intensitées en float comprises entre 0 et 1 et de dimensions
                (hauteur, largeur, canal) (canal: R=1 G=2 B=3)

        Returns:
            Rcombined (np.ndarray): Carte d'intensité combinée.
        """

        Rcombined = imap.copy()
        for l in self.L:
            R = self.basicLineDetector(imap, l)
            Rcombined += R

        Rcombined = Rcombined / (len(self.L) + 1)

        return Rcombined

    def learnThreshold(self, dataset):
        """
        Apprend le seuil optimal pour obtenir la précision la plus élevée
        sur le dataset donné.
        Cette méthode modifie la valeur de self.threshold par le seuil
        optimal puis renvoie ce seuil et la précision obtenue.

        Args:
            dataset (List[dict]): Liste de dictionnaires contenant les champs ["image", "label", "mask"].

        Returns:
            threshold (float): Seuil proposant la meilleure précision
            accuracy (float): Valeur de la meilleure précision
        """

        fpr, tpr, thresholds = self.roc(dataset)

        P = 0
        S = 0
        for d in dataset:
            label = d["label"]
            mask = d["mask"]
            S += mask.sum()
            P += label[mask].sum()
        N = S - P

        accuracies = ((1 - fpr) * N + tpr * P) / S

        i = np.argmax(accuracies)
        threshold = thresholds[i]
        accuracy = accuracies[i]

        self.threshold = threshold
        return threshold, accuracy

    def segmentVessels(self, image):
        """
        Segmente les vaisseaux sur une image en utilisant la MSLD.

        Args:
            image (np.ndarray): Image RGB sur laquelle appliquer l'algorithme MSLD.

        Returns:
            vessels (np.ndarray): Carte binaire 2D de la segmentation des vaisseaux.
        """

        pred = self.multiScaleLineDetector(image)
        return pred > self.threshold

    ############################################################################
    #                         Segmentation Metrics                             #
    ############################################################################
    def roc(self, dataset):
        """
        Calcule la courbe ROC de l'algorithme MSLD sur un dataset donné et sur la région d'intérêt indiquée par le
        champ "mask".

        Args:
            dataset (List[dict]): Base de données sur laquelle calculer la courbe ROC.

        Returns:
            fpr (np.ndarray): Vecteur float des taux de faux positifs.
            tpr (np.ndarray): Vecteur float des taux de vrais positifs.
            thresholds (np.ndarray): Vecteur float des seuils associés à ces taux.
        """

        y_true = []
        y_pred = []

        for d in dataset:
            # Pour chaque élément de dataset
            label = d["label"]  # On lit le label
            mask = d["mask"]  # le masque
            image = d["image"]  # et l'image de l'élément.

            # On calcule la prédiction du msld sur cette image.
            prediction = self.multiScaleLineDetector(image)

            # On applique les masques à label et prediction pour qu'ils contiennent uniquement
            # la liste des pixels qui appartiennent au masque.
            label = label[mask]
            prediction = prediction[mask]

            # On ajoute les vecteurs label et prediction aux listes y_true et y_pred
            y_true.append(label)
            y_pred.append(prediction)

        # On concatène les vecteurs de la listes y_true pour obtenir un unique vecteur contenant
        # les labels associés à tous les pixels qui appartiennent au masque du dataset.
        y_true = np.concatenate(y_true)
        # Même chose pour y_pred.
        y_pred = np.concatenate(y_pred)

        # On calcule le taux de vrai positif et de faux positif du dataset pour chaque seuil possible.
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)

        return fpr, tpr, thresholds  # DIFF tpr, fpr

    def naiveMetrics(self, dataset):
        """
        Évalue la précision et la matrice de confusion de l'algorithme sur
        un dataset donné et sur la région d'intérêt indiquée par le
        champs mask.

        Args:
            dataset (List[dict]): Base de données sur laquelle calculer les métriques.

        Returns:
            accuracy (float): Précision.
            confusion_matrix (np.ndarray): Matrice de confusion 2 x 2 normalisée par le nombre de labels positifs et
                négatifs.
        """

        conf_mat = np.zeros((2, 2), np.int)
        for d in dataset:
            image = d["image"]
            pred = self.segmentVessels(image)
            label = d["label"]

            mask = d["mask"]
            pred = pred[mask]
            label = label[mask]

            conf_mat[1, 1] += (pred & label).sum()
            conf_mat[0, 1] += ((np.invert(pred)) & label).sum()
            conf_mat[1, 0] += (pred & np.invert(label)).sum()
            conf_mat[0, 0] += (np.invert(pred) & np.invert(label)).sum()

        accuracy = conf_mat.trace() / conf_mat.sum()
        confusion_matrix = conf_mat / conf_mat.sum(axis=1, keepdims=True)

        return accuracy, confusion_matrix

    def dice(self, dataset):
        """
        Évalue l'indice Sørensen-Dice de l'algorithme sur un dataset donné et sur la région d'intérêt indiquée par le
        champ "mask".

        Args:
            dataset (List[dict]): Base de données sur laquelle calculer l'indice Dice.

        Returns:
            dice_index (float): Indice de Sørensen-Dice.
        """

        preds = []
        labels = []

        for d in dataset:
            pred = self.segmentVessels(d["image"])
            label = d["label"]
            mask = d["mask"]

            preds.append(pred[mask])
            labels.append(label[mask])

        preds = np.concatenate(preds)
        labels = np.concatenate(labels)

        return dice(labels, preds)


def dice(targets, predictions):
    """Calcule l'indice de Sørensen-Dice entre les prédictions et la vraie segmentation. Les deux arrays doivent avoir
    la même forme.

    Args:
        targets (np.ndarray): Vraie segmentation.
        predictions (np.ndarray): Prédiction de la segmentation.

    Returns:
        dice_index (float): Indice de Sørensen-Dice.
    """

    return 2 * np.sum(targets * predictions) / (targets.sum() + predictions.sum())
