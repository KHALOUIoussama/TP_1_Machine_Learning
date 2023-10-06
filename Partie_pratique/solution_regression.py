
""" 
Lien du git : https://github.com/KHALOUIoussama/TP_1_Machine_Learning/tree/master/ift603_tp1_prog
 """
#####
# Khaloui Oussama (23130746)
# Assabar Taoufik (Asst1001)
# Bellari Nada (beln1812)
###

import random

import numpy as np
from sklearn.linear_model import Ridge


class Regression:
    def __init__(self, lamb, m=1):
        self.lamb = lamb
        self.w = None
        self.M = m

    def fonction_base_polynomiale(self, x):
        """
        Fonction de base qui projette la donnee x vers un espace polynomial tel que mentionne au chapitre 3.
        Si x est un scalaire, alors phi_x sera un vecteur à  M dimensions : (x^1,x^2,...,x^ M)
        Si x est un vecteur de N scalaires, alors phi_x sera un tableau 2D de taille NxM

        NOTE : En mettant phi_x = x, on a une fonction de base lineaire qui fonctionne pour une regression lineaire
        """
        if isinstance(x, (int, float)): # Si x est un scalaire
            phi_x = np.array([x ** i for i in range(1, self.M + 1)])
        else: # Si x est un vecteur de N scalaires
            phi_x = np.array([[x_i ** j for j in range(1, self.M + 1)] for x_i in x])
        return phi_x
    
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
        k = 10
        N = len(X)
        if N < k:
            k = N
        
        # Mélange des données avant la validation croisée
        indices = np.arange(N)
        np.random.shuffle(indices)
        X = X[indices]
        t = t[indices]

        # Diviser les données en k folds
        X_folds = np.array_split(X, k)
        t_folds = np.array_split(t, k)

        M_values = range(1, 41)  # Par exemple, tester M de 1 à 40
        mean_errors = []

        for M in M_values:
            self.M = M
            errors = []
            
            for i in range(k):
                # Utiliser le i-ème fold comme ensemble de validation et les autres pour l'entraînement
                X_valid = X_folds[i]
                t_valid = t_folds[i]
                
                X_train = np.concatenate([X_folds[j] for j in range(k) if j != i])
                t_train = np.concatenate([t_folds[j] for j in range(k) if j != i])
                
                # Entraînement et calcul de l'erreur
                self.entrainement(X_train, t_train)
                predictions = self.prediction(X_valid)
                error = self.erreur(t_valid, predictions)
                errors.append(error)
            
            # Calculer l'erreur moyenne pour cette valeur de M
            mean_errors.append(np.mean(errors))

        # Choisir la meilleure valeur de M
        self.M = M_values[np.argmin(mean_errors)]
        '''
        # Option 2: Sous-échantillonage aléatoire avec ratio 80:20
        
        for M in M_values:
            self.M = M
            errors = []
            
            for _ in range(k):
                # Séparation aléatoire des données en Dtrain et Dvalid
                indices_train = np.random.choice(N, int(0.8 * N), replace=False)
                indices_valid = np.setdiff1d(np.arange(N), indices_train)
                
                X_train = X[indices_train]
                t_train = t[indices_train]
                
                X_valid = X[indices_valid]
                t_valid = t[indices_valid]
                
                # Entraînement et calcul de l'erreur
                self.entrainement(X_train, t_train)
                predictions = self.prediction(X_valid)
                error = self.erreur(t_valid, predictions)
                errors.append(error)
            
            # Calculer l'erreur moyenne pour cette valeur de M
            mean_errors.append(np.mean(errors))

        # Choisir la meilleure valeur de M
        self.M = M_values[np.argmin(mean_errors)]
        '''


    def entrainement(self, X, t, using_sklearn=False):
        """
        Entraîne la regression lineaire sur l'ensemble d'entraînement forme des
        entrees ``X`` (un tableau 2D Numpy, ou la n-ieme rangee correspond à l'entree
        x_n) et des cibles ``t`` (un tableau 1D Numpy ou le
        n-ieme element correspond à la cible t_n). L'entraînement doit
        utiliser le poids de regularisation specifie par ``self.lamb``.

        Cette methode doit assigner le champs ``self.w`` au vecteur
        (tableau Numpy 1D) de taille D+1, tel que specifie à la section 3.1.4
        du livre de Bishop.
        
        Lorsque using_sklearn=True, vous devez utiliser la classe "Ridge" de 
        la librairie sklearn (voir http://scikit-learn.org/stable/modules/linear_model.html)
        
        Lorsque using_sklearn=Fasle, vous devez implementer l'equation 3.28 du
        livre de Bishop. Il est suggere que le calcul de ``self.w`` n'utilise
        pas d'inversion de matrice, mais utilise plutôt une procedure
        de resolution de systeme d'equations lineaires (voir np.linalg.solve).

        Aussi, la variable membre self.M sert à projeter les variables X vers un espace polynomiale de degre M
        (voir fonction self.fonction_base_polynomiale())

        NOTE IMPORTANTE : lorsque self.M <= 0, il faut trouver la bonne valeur de self.M

        """
        
        if self.M <= 0:
            self.recherche_hyperparametre(X, t)

        phi_x = self.fonction_base_polynomiale(X)
        
        if not using_sklearn:
            # *** Entraînement par maximum de vraisemblance ***
            A = np.dot(phi_x.T, phi_x)
            b = np.dot(phi_x.T, t)
            self.w = np.linalg.solve(A, b)
            
        if using_sklearn:
            # **Entraînement par maximum a posteriori**
            # Utilisation de Ridge regression de sklearn
            
            ridge = Ridge(alpha=self.lamb, fit_intercept=False)
            
            ridge.fit(phi_x, t)
            self.w = ridge.coef_

    def prediction(self, x):
        """
        Retourne la prediction de la regression lineaire
        pour une entree, representee par un tableau 1D Numpy ``x``.

        Cette methode suppose que la methode ``entrainement()``
        a prealablement ete appelee. Elle doit utiliser le champs ``self.w``
        afin de calculer la prediction y(x,w) (equation 3.1 et 3.3).
        """
        phi_x = self.fonction_base_polynomiale(x)
        y = np.dot(phi_x, self.w)
        
        return y

    @staticmethod
    def erreur(t, prediction):
        """
        Retourne l'erreur de la difference au carre entre
        la cible ``t`` et la prediction ``prediction``.
        """
        erreur_moyenne = np.mean((t - prediction) ** 2)
        return erreur_moyenne
