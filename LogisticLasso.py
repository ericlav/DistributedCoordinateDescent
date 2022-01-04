import numpy as np
from numpy.linalg import norm
from numba import njit

"""
Implémentation de la classe LogisticLasso qui
implémente la descente de gradient proximal
et la descente par coordonnée classique
"""


@njit
def numba_sigmoid(z):
    """
    Fonction sigmoide 
    
    z: réel ou objet numpy
    
    return: fonction sigmoide évaluée en z
    """
    return 1. / (1. + np.exp(-z))

@njit
def numba_prox_L1(z, lmbd):
    """
    Proximal associé à la régularisation L1 (soft thresholding)
    
    z: réel
    lmbd: réel, paramètre de régularisation
    
    return: proximal évalué en z et lmbd
    """
    return np.sign(z) * np.maximum(np.abs(z) - lmbd, 0.)

@njit
def numba_grad_sigmoid(y, X, yXw):  
    """
    Evalue le gradient de la fonction de perte dans l'objectif
    
    y: vecteur des classes à prédire
    X: matrice des observations  
    yXw: produit de y vecteur des classes à prédire
         avec le produit matriciel de X matrice des observations  
         et w vecteur des coefficients du modèle
         
    return: gradient évalué avec les paramètres en entrée
    """
    return -(X.T*y*numba_sigmoid(-yXw)).sum(axis=1)

@njit
def numba_objectif(w, yXw, lmbd):
    """
    Objectif de l'apprentissage (somme fonction de perte et pénalisation)
    
    yXw: produit de y vecteur des classes à prédire
         avec le produit matriciel de X matrice des observations  
         et w vecteur des coefficients du modèle
    lmbd: réel, paramètre de régularisation
    
    return: évaluation de l'objectif pour les paramètres en entrée
    """
    return np.log(1. + np.exp(-yXw)).sum() + lmbd * norm(w, ord=1)


@njit
def numba_pgd_solver(X, y, lmbd, max_iter, tol, lips_const):
    """
    Descente de Gradient proximal
        
    X: matrice des observations
    y: vecteur des classes à prédire
    lmbd: réel, paramètre de régularisation
    max_iter: entier, nombre maximal d'itérations à exécuter
    tol: réel, seuil de convergence sous lequel arrêter l'algorithme
    lips_const: réel, constante de lipschitz
        
    return: vecteur des coefficients appris et vecteur des valeurs
    de la fonction objectif évaluée à chaque itération
    """
    
    # n nombre de données, p nombre de features
    n, p = X.shape
    
    # Evolution de l'objectif au cours des itérations
    objectif_list_ = np.zeros(max_iter)

    # Initialisation des paramètres à 0
    w = np.zeros(p)

    # Stockage du produit de y, X et w pour ne pas répéter les calculs
    yXw = y * X.dot(w)
                
    for i in range(max_iter):
            
        # Stockage de w pour critère de convergence
        old_w = w.copy()
            
        # Mise à jour de w
        w = numba_prox_L1(w - numba_grad_sigmoid(y, X, yXw)/lips_const, lmbd/lips_const)
            
        # Stockage du produit de y, X et w pour ne pas répéter les calculs
        yXw = y * X.dot(w)
            
        # Evaluation de la fonction objectif à minimiser
        objectif_list_[i] = numba_objectif(w, yXw, lmbd)
            
        # Critère d'arrêt de convergence
        if (np.max(np.abs(w - old_w)) < tol):
            objectif_list_ = objectif_list_[:i+1]
            break

    return w, objectif_list_

@njit
def numba_cd_solver(X, y, lmbd, max_iter, tol, lips_const):
        """
        Descente de coordonnée proximal

        X: matrice des observations
        y: vecteur des classes à prédire
        lmbd: réel, paramètre de régularisation
        max_iter: entier, nombre maximal d'itérations à exécuter
        tol: réel, seuil de convergence sous lequel arrêter l'algorithme
        lips_const: réel, constante de lipschitz

        return: vecteur des coefficients appris et vecteur des valeurs
        de la fonction objectif évaluée à chaque itération
        """
    
        # n nombre de données, p nombre de features
        n, p = X.shape

        # Evolution de l'objectif au cours des itérations
        objectif_list_ = np.zeros(max_iter)

        # Initialisation des paramètres à 0
        w = np.zeros(p)

        # Stockage du produit de X et w pour réduire les calculs
        Xw = X.dot(w)

        for i in range(max_iter):

            # Stockage de w pour critère de convergence
            old_w = w.copy()

            # Cyclic coordinate descent
            for j in range(p):

                old_w_j = old_w[j]
                
                yXw = y * Xw

                # Mise à jour de wj
                grad_j = (- y * X[:, j] * numba_sigmoid(- yXw)).sum()
                w[j] = numba_prox_L1(old_w_j - grad_j / lips_const[j], lmbd / lips_const[j])

                # Mise à jour Xw
                if old_w_j != w[j]:
                    Xw += X[:, j] * (w[j] - old_w_j)
            
            yXw = y * Xw
            
            # Evaluation de la fonction objectif à minimiser
            objectif_list_[i] = numba_objectif(w, yXw, lmbd)

            # Critère d'arrêt de convergence
            if (np.max(np.abs(w - old_w)) < tol):
                objectif_list_ = objectif_list_[:i+1]
                break

        return w, objectif_list_



class LogisticLasso:
    
    def __init__(self, solver, lmbd, max_iter, tol):
        """
        solver: méthode d'optimisation à utiliser
        lmbd: réel, paramètre de régularisation
        max_iter: entier, nombre maximal d'itérations à exécuter
        tol: réel, seuil de convergence sous lequel arrêter l'algorithme
        """        
        
        self.solver = solver
        self.lmbd = lmbd
        self.max_iter = max_iter
        self.tol = tol
        
        self.coef_ = None
        self.objectif_list_ = None
        
        
    def fit(self, X, y):
        
        if self.solver == "pgd":
            w, objectif_list = self._pgd_solver(X, y)
        elif self.solver == "pgd_numba_low":
            w, objectif_list = self._pgd_numba_low_solver(X, y)
        elif self.solver == "pgd_numba":
            w, objectif_list = self._pgd_numba_solver(X, y)
        elif self.solver == "cd":
            w, objectif_list = self._cd_solver(X, y)
        elif self.solver == "cd_numba_low":
            w, objectif_list = self._cd_numba_low_solver(X, y)
        elif self.solver == "cd_numba":
            w, objectif_list = self._cd_numba_solver(X, y)
        else:
            print("Solver inconnu, options disponibles 'pgd'/'pgd_numba_low'/'pgd_numba' et 'cd'/'cd_numba_low'/'cd_numba'.")
            
    
    def _sigmoid(self, z):
        """
        Fonction sigmoide 

        z: réel ou objet numpy

        return: fonction sigmoide évaluée en z
        """
        return 1. / (1. + np.exp(-z))
    
    def _prox_L1(self, z, lmbd):
        """
        Proximal associé à la régularisation L1 (soft thresholding)

        z: réel
        lmbd: réel, paramètre de régularisation

        return: proximal évalué en z et lmbd
        """
        return np.sign(z) * np.maximum(np.abs(z) - lmbd, 0.)
    
    def _grad_sigmoid(self, y, X, yXw):
        """
        Evalue le gradient de la fonction de perte dans l'objectif

        y: vecteur des classes à prédire
        X: matrice des observations  
        yXw: produit de y vecteur des classes à prédire
             avec le produit matriciel de X matrice des observations  
             et w vecteur des coefficients du modèle

        return: gradient évalué avec les paramètres en entrée
        """
        return -(X.T*y*self._sigmoid(-yXw)).sum(axis=1)
    
    def _objectif(self, w, yXw, lmbd):
        """
        Objectif de l'apprentissage (somme fonction de perte et pénalisation)

        yXw: produit de y vecteur des classes à prédire
             avec le produit matriciel de X matrice des observations  
             et w vecteur des coefficients du modèle
        lmbd: réel, paramètre de régularisation

        return: évaluation de l'objectif pour les paramètres en entrée
        """
        return np.log(1. + np.exp(-yXw)).sum() + lmbd * norm(w, ord=1)
    
    
    def _pgd_solver(self, X, y):
        """
        Descente de Gradient proximal
        
        X: matrice des observations
        y: vecteur des classes à prédire
        
        return: vecteur des coefficients appris et vecteur des valeurs
        de la fonction objectif évaluée à chaque itération
        """
    
        # n nombre de données, p nombre de features
        n, p = X.shape
        
        # Constante de lipschitz
        P, D, Q = np.linalg.svd(X)
        lips_const = (D[0]**2)/4

        # Evolution de l'objectif au cours des itérations
        self.objectif_list_ = np.zeros(self.max_iter)

        # Initialisation des paramètres à 0
        w = np.zeros(p)

        # Stockage du produit de y, X et w pour ne pas répéter les calculs
        yXw = y * X.dot(w)
                
        for i in range(self.max_iter):
            
            # Stockage de w pour critère de convergence
            old_w = w.copy()
            
            # Mise à jour de w
            w = self._prox_L1(w - self._grad_sigmoid(y, X, yXw)/lips_const, self.lmbd/lips_const)
            
            # Stockage du produit de y, X et w pour ne pas répéter les calculs
            yXw = y * X.dot(w)
            
            # Evaluation de la fonction objectif à minimiser
            self.objectif_list_[i] = self._objectif(w, yXw, self.lmbd)
            
            # Critère d'arrêt de convergence
            if (np.max(np.abs(w - old_w)) < self.tol):
                self.objectif_list_ = self.objectif_list_[:i+1]
                break
                
        self.coef_ = w

        return self.coef_, self.objectif_list_
    
    
    def _pgd_numba_low_solver(self, X, y):
        """
        Descente de Gradient proximal avec les fonctions numba_prox_L1,
        grad_sigmoid et objectif sous numba
        
        X: matrice des observations
        y: vecteur des classes à prédire
        
        return: vecteur des coefficients appris et vecteur des valeurs
        de la fonction objectif évaluée à chaque itération
        """
    
        # n nombre de données, p nombre de features
        n, p = X.shape
        
        # Constante de lipschitz
        P, D, Q = np.linalg.svd(X)
        lips_const = (D[0]**2)/4

        # Evolution de l'objectif au cours des itérations
        self.objectif_list_ = np.zeros(self.max_iter)

        # Initialisation des paramètres à 0
        w = np.zeros(p)

        # Stockage du produit de y, X et w pour ne pas répéter les calculs
        yXw = y * X.dot(w)
                
        for i in range(self.max_iter):
            
            # Stockage de w pour critère de convergence
            old_w = w.copy()
            
            # Mise à jour de w
            w = numba_prox_L1(w - numba_grad_sigmoid(y, X, yXw)/lips_const, self.lmbd/lips_const)
            
            # Stockage du produit de y, X et w pour ne pas répéter les calculs
            yXw = y * X.dot(w)
            
            # Evaluation de la fonction objectif à minimiser
            self.objectif_list_[i] = numba_objectif(w, yXw, self.lmbd)
            
            # Critère d'arrêt de convergence
            if (np.max(np.abs(w - old_w)) < self.tol):
                self.objectif_list_ = self.objectif_list_[:i+1]
                break
                
        self.coef_ = w

        return self.coef_, self.objectif_list_
    
    
    def _pgd_numba_solver(self, X, y):
        """
        Descente de Gradient proximal avec pgd_solver sous numba
        
        X: matrice des observations
        y: vecteur des classes à prédire
        
        return: vecteur des coefficients appris et vecteur des valeurs
        de la fonction objectif évaluée à chaque itération
        """
       
        # Constante de lipschitz
        P, D, Q = np.linalg.svd(X)
        lips_const = (D[0]**2)/4
        
        self.coef_, self.objectif_list_ = numba_pgd_solver(X, y, self.lmbd, self.max_iter, self.tol, lips_const)
        
        return self.coef_, self.objectif_list_ 
    
    
    def _cd_solver(self, X, y):
        """
        Descente de coordonnée proximal
        
        X: matrice des observations
        y: vecteur des classes à prédire
        
        return: vecteur des coefficients appris et vecteur des valeurs
        de la fonction objectif évaluée à chaque itération
        """
    
        # n nombre de données, p nombre de features
        n, p = X.shape

        # Constantes de lipschitz
        lips_const = np.linalg.norm(X, axis=0) ** 2 / 4

        # Evolution de l'objectif au cours des itérations
        self.objectif_list_ = np.zeros(self.max_iter)

        # Initialisation des paramètres à 0
        w = np.zeros(p)

        # Stockage du produit de X et w pour réduire les calculs
        Xw = X.dot(w)

        for i in range(self.max_iter):

            # Stockage de w pour critère de convergence
            old_w = w.copy()

            # Cyclic coordinate descent
            for j in range(p):

                old_w_j = old_w[j]
                
                yXw = y * Xw

                # Mise à jour de wj
                grad_j = (- y * X[:, j] * self._sigmoid(- yXw)).sum()
                w[j] = self._prox_L1(old_w_j - grad_j / lips_const[j], self.lmbd / lips_const[j])

                # Mise à jour Xw
                if old_w_j != w[j]:
                    Xw += X[:, j] * (w[j] - old_w_j)
            
            yXw = y * Xw
            
            # Evaluation de la fonction objectif à minimiser
            self.objectif_list_[i] = self._objectif(w, yXw, self.lmbd)

            # Critère d'arrêt de convergence
            if (np.max(np.abs(w - old_w)) < self.tol):
                self.objectif_list_ = self.objectif_list_[:i+1]
                break
                
        self.coef_ = w

        return self.coef_, self.objectif_list_
    
    
    def _cd_numba_low_solver(self, X, y):
        """
        Descente de coordonée proximal avec les fonctions numba_prox_L1,
        sigmoid et objectif sous numba
        
        X: matrice des observations
        y: vecteur des classes à prédire
        
        return: vecteur des coefficients appris et vecteur des valeurs
        de la fonction objectif évaluée à chaque itération
        """
        
        # n nombre de données, p nombre de features
        n, p = X.shape

        # Constantes de lipschitz
        lips_const = np.linalg.norm(X, axis=0) ** 2 / 4

        # Evolution de l'objectif au cours des itérations
        self.objectif_list_ = np.zeros(self.max_iter)

        # Initialisation des paramètres à 0
        w = np.zeros(p)

        # Stockage du produit de X et w pour réduire les calculs
        Xw = X.dot(w)

        for i in range(self.max_iter):

            # Stockage de w pour critère de convergence
            old_w = w.copy()

            # Cyclic coordinate descent
            for j in range(p):

                old_w_j = old_w[j]
                
                yXw = y * Xw

                # Mise à jour de wj
                grad_j = (- y * X[:, j] * numba_sigmoid(- yXw)).sum()
                w[j] = numba_prox_L1(old_w_j - grad_j / lips_const[j], self.lmbd / lips_const[j])

                # Mise à jour Xw
                if old_w_j != w[j]:
                    Xw += X[:, j] * (w[j] - old_w_j)
            
            yXw = y * Xw
            
            # Evaluation de la fonction objectif à minimiser
            self.objectif_list_[i] = numba_objectif(w, yXw, self.lmbd)

            # Critère d'arrêt de convergence
            if (np.max(np.abs(w - old_w)) < self.tol):
                self.objectif_list_ = self.objectif_list_[:i+1]
                break
                
        self.coef_ = w

        return self.coef_, self.objectif_list_
    
    
    def _cd_numba_solver(self, X, y):
        """
        Descente de coordonnée proximal avec cd_solver sous numba
        
        X: matrice des observations
        y: vecteur des classes à prédire
        
        return: vecteur des coefficients appris et vecteur des valeurs
        de la fonction objectif évaluée à chaque itération
        """
        
        # Constantes de lipschitz
        lips_const = np.linalg.norm(X, axis=0) ** 2 / 4
        
        self.coef_, self.objectif_list_ = numba_cd_solver(X, y, self.lmbd, self.max_iter, self.tol, lips_const)
        
        return self.coef_, self.objectif_list_ 
    

    def predict(self, X):
        """
        Prédictions de la régression logistique

        X: X matrice des observations (exemples x variables)

        return: vecteur des prédictions
        """
        return np.sign(self._sigmoid(X.dot(self.coef_))-0.5)
    