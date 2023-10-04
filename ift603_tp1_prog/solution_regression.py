from operator import length_hint
import numpy as np
import random
from sklearn import linear_model, model_selection

class Regression:
    def __init__(self, lamb, m=1, using_sklearn=False):
        self.lamb = lamb
        self.w = None
        self.M = m
        self.usingSkl = using_sklearn

    def fonction_base_polynomiale(self, x):
        """
        Fonction de base qui projette la donnee x vers un espace polynomial tel que mentionne au chapitre 3.
        Si x est un scalaire, alors phi_x sera un vecteur à self.M dimensions : (x^1,x^2,...,x^self.M)
        Si x est un vecteur de N scalaires, alors phi_x sera un tableau 2D de taille NxM

        NOTE : En mettant phi_x = x, on a une fonction de base lineaire qui fonctionne pour une regression lineaire
        """
        isVector = isinstance(x,np.ndarray)

        # Cas de la régression linéaire
        if self.M == 1:
            return x.reshape(len(x),1) if isVector else np.array([[x]])
        
        # Cas de la régression non-linéaire
        powerArray = np.arange(self.M) + 1
        
        # Si x est un vecteur
        if isVector:
            length = len(x)
            phi = np.zeros((length, self.M))

            for i in range(length):
                phi[i,] = np.power(x[i], powerArray)

            return phi

        # Si x est un scalaire
        return np.power(x, powerArray).reshape(1,self.M)

    def recherche_hyperparametre(self, X, t):
        """
        Trouver la meilleure valeur pour l'hyper-parametre self.M (pour un lambda fixe donné en entrée).

        Option 1
        Validation croisée de type "k-fold" avec k=10. La méthode array_split de numpy peut être utlisée 
        pour diviser les données en "k" parties. Si le nombre de données en entrée N est plus petit que "k", 
        k devient égal à N. Il est important de mélanger les données ("shuffle") avant de les sous-diviser
        en "k" parties.

        Option 2
        Sous-échantillonage aléatoire avec ratio 80:20 pour Dtrain et Dvalid, avec un nombre de répétition k=10.

        Note: 

        Le resultat est mis dans la variable self.M

        X: vecteur de donnees
        t: vecteur de cibles
        """
        k = 10 if X.shape[0] >= 10 else X.shape[0]
        # permet d'arrêter la recherche lorsque la différence d'erreur 
        # entre deux runs est trop grande
        maxErrorDifference = 0.1
        minError = [1,np.max(t)]
        self.M = 1
        tmp = -1
        
        while(True):
            kFold = model_selection.KFold(n_splits=k,shuffle=True,random_state=self.M)
            errorList = list()

            for trainIdx, testIdx in kFold.split(X):
                trainX = X[trainIdx]
                trainT = t[trainIdx]
                testX = X[testIdx]
                testT = t[testIdx]

                self.resolution(trainX, trainT)
                prediction = self.prediction(testX)
                err = self.erreur(testT,prediction)
                errorList.append(np.mean(err))
            
            mean = np.mean(errorList)
            print("M = {}, erreur après cross-validation : {:.4f}".format(self.M,mean))

            toContinue = True if tmp == -1 else maxErrorDifference > (mean - tmp)

            if toContinue :
                if minError[1] > mean:
                    minError[0] = self.M
                    minError[1] = mean 
                tmp = mean
                self.M += 1
            
            else:
                print("Dépassement de maxErrorDifference entre deux runs")
                break
