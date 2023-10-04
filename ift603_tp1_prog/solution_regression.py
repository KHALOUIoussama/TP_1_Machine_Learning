# -*- coding: utf-8 -*-

#####
# Khaloui Oussama (23130746) .
# ASSABAR Taoufik (cip: Asst1001)
# ~= À MODIFIER =~.
###

import numpy as np
import random

class Regression:
    def __init__(self, lamb, m=1):
        self.lamb = lamb
        self.w = None
        self.M = m

    def fonction_base_polynomiale(self, x):
        """
        Fonction de base qui projette la donnee x vers un espace polynomial tel que mentionne au chapitre 3.
        Si x est un scalaire, alors phi_x sera un vecteur à self.M dimensions : (x^1,x^2,...,x^self.M)
        Si x est un vecteur de N scalaires, alors phi_x sera un tableau 2D de taille NxM
        """
        if isinstance(x, np.ndarray):
            return np.column_stack([x**i for i in range(1, self.M + 1)])
        else:
            return np.array([x**i for i in range(1, self.M + 1)])

    def recherche_hyperparametre(self, X, t):
        k = 10 if X.shape[0] >= 10 else X.shape[0]
        maxErrorDifference = 0.1
        minError = [1, np.max(t)]
        self.M = 1
        tmp = -1

        while True:
            indices = list(range(X.shape[0]))
            random.shuffle(indices)
            errorList = []

            for i in range(k):
                start = int(i * len(indices) / k)
                end = int((i + 1) * len(indices) / k)
                testIdx = indices[start:end]
                trainIdx = list(set(indices) - set(testIdx))

                trainX = X[trainIdx]
                trainT = t[trainIdx]
                testX = X[testIdx]
                testT = t[testIdx]

                self.resolution(trainX, trainT)
                prediction = self.prediction(testX)
                err = self.erreur(testT, prediction)
                errorList.append(np.mean(err))

            mean = np.mean(errorList)
            print("M = {}, erreur après cross-validation : {:.4f}".format(self.M, mean))

            toContinue = True if tmp == -1 else maxErrorDifference > (mean - tmp)

            if toContinue:
                if minError[1] > mean:
                    minError[0] = self.M
                    minError[1] = mean
                tmp = mean
                self.M += 1
            else:
                print("Dépassement de maxErrorDifference entre deux runs")
                break

        self.M = minError[0]
        print("=> M optimal : {}, erreur de cross-validation : {:.4f}".format(minError[0], minError[1]))

    def resolution(self, X, t):
        phiX = self.fonction_base_polynomiale(X)
        phiX = np.column_stack([np.ones(phiX.shape[0]), phiX])
        toInvert = (np.identity(self.M + 1) * self.lamb) + (phiX.T @ phiX)
        self.w = np.linalg.solve(toInvert, phiX.T @ t)

    def entrainement(self, X, t):
        if self.M <= 0:
            self.recherche_hyperparametre(X, t)

        self.resolution(X, t)

    def prediction(self, x):
        return self.w[0] + self.fonction_base_polynomiale(x) @ self.w[1:]

    @staticmethod
    def erreur(t, prediction):
        return np.power(t - prediction, 2)
