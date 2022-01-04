import numpy as np
from numpy.linalg import norm
from multiprocessing import Pool
from pathos.multiprocessing import ProcessPool
from numba import njit

"""
Implémentation de dGLMNET distribué multiprocessus
Descente de coordonnées par blocs pour l'apprentissage
d'une régression logistique avec régularisation L1
"""


# Fonctions utiles

@njit(nogil=True)
def sigmoid(t):
    """
    Fonction sigmoide
    
    t: réel ou objet numpy
    
    return: fonction sigmoide évaluée en t
    """
    return 1. / (1. + np.exp(-t))

@njit(nogil=True)
def prox_L1(t, lmbd):
    """
    Proximal associé à la régularisation L1 (soft thresholding)
    
    t: réel
    lmbd: réel, paramètre de régularisation
    
    return: proximal évalué en t et lmbd
    """
    return np.sign(t) * np.maximum(np.abs(t) - lmbd, 0.)

@njit(nogil=True)
def objectif(yX, w, lmbd):
    """
    Objectif de l'apprentissage (somme fonction de perte et pénalisation)
    
    yX: produit de y vecteur des classes à prédire
        et X matrice des observations (exemples x variables)
    w : vecteur des coefficients du modèle
    lmbd: réel, paramètre de régularisation
    
    return: évaluation de l'objectif pour les paramètres en entrée
    """
    return np.log(1. + np.exp(-yX.dot(w))).sum() + lmbd * norm(w, ord=1)

@njit(nogil=True)
def grad_sigmoid(yX, w):  
    """
    Evalue le gradient de la fonction de perte dans l'objectif
    
    yX: produit de y vecteur des classes à prédire
        et X matrice des observations (exemples x variables)
    w: vecteur des coefficients
    
    return: gradient évalué avec les paramètres en entrée
    """
    return -(yX.T*sigmoid(-yX.dot(w))).sum(axis=1)


# Composantes de dGLMNET

#Calcul de la mise à jour partielle pour une machine m
@njit(cache=True)
def partial_solver(Xm, y, wm, lmbd):
    """
    Résolution du problème d'optimisation pour un bloc donné
    Par méthode de quasi-newton (approximation de l'ordre 2)
    avec une passe unique sur les coordonnées (voire Trofimov et Genkin)
    
    Xm: matrice des observations pour le bloc de variables considérées
    y: y vecteur des classes à prédire
    wm: vecteur des coefficients actuels pour les variables du bloc
    lmbd: réel, paramètre de régularisation
    
    return : vecteur des incrémentations des coefficients pour les variables du bloc
    """
    
    n, pm = Xm.shape
    
    DeltaWm = np.zeros(pm)
    
    # voir (4) de l'article
    px = sigmoid(Xm @ wm)
    doublev = px*(1-px)
    z = (((y+1)/2) - px) / doublev
    
    for j in range(pm):
    
        # voir (6) de l'article
        
        q = z - Xm @ DeltaWm + (wm[j] + DeltaWm[j])*Xm[:,j]
        
        prox = prox_L1(np.vdot(doublev*q, Xm[:,j]), lmbd)
        
        DeltaWm[j] += prox/np.vdot(doublev, Xm[:,j]**2) - wm[j]
    
    return DeltaWm


@njit
def line_search(yX, w, DeltaW, lmbd):
    """
    Trouve le facteur alpha adéquat par linear search
    
    y: vecteur des classes à prédire
    X: matrice des observations (exemples x variables)
    w: vecteur des coefficients
    DeltaW: direction pour laquelle trouver alpha
    lmbd: réel, paramètre de régularisation
    
    return: facteur alpha adéquat
    """
        
    alpha_init_list = np.linspace(0.1,1,19)
    alpha_init = 0.1
    f_min = objectif(yX, w + alpha_init*DeltaW, lmbd)
    
    for x in alpha_init_list:
        f_new = objectif(yX, w + x*DeltaW, lmbd)
        if f_min > f_new:
            f_min = f_new
            alpha_init = x
       
    cond = True
    b = 0.5
    sigma = 0.01
    alpha = alpha_init
    
    while cond:
        
        f_new = objectif(yX, w + alpha*DeltaW, lmbd)
        D = grad_sigmoid(yX, w).dot(DeltaW) + lmbd*(norm(w + DeltaW, ord=1) - norm(w, ord=1))
        
        if f_new < objectif(yX, w, lmbd) + alpha * sigma * D:
            cond=False
        else:
            alpha *= b
        
    return alpha


# dGLMNET_solver

def dGLMNET_solver(X, y, lmbd, M=2, max_iter=100, tol=1e-6):
    """
    Apprentissage de la régression logistique avec régularisation L1
    par le biais de la méthode dGLMNET
    descente de coordonnées par blocs distribuée
    Programmée de manière distribuée avec multiprocessus
    
    X: matrice des observations (exemples x variables)
    y: vecteur des classes à prédire
    M: entier, nombre de blocs de la descente par coordonnées
    max_iter: entier, nombre maximal d'itérations à exécuter
    tol: réel, seuil de convergence sous lequel arrêter l'algorithme
    
    return: vecteur des coefficients appris et vecteur des valeurs
    de la fonction objectif évaluée à chaque itération
    """

    # n nombre de données, p nombre de features
    n, p = X.shape

    # Evolution de l'objectif au cours des itérations
    objectif_list_ = np.zeros(max_iter + 1)

    # Initialisation des paramètres à 0
    w = np.zeros(p)
    
    # Calcul de yX pour réduire la répétition des calculs
    yX = np.einsum('i,ij->ij', y, X)
    
    # Objectif initial
    objectif_list_[0] = objectif(yX, w, lmbd)
    
    # Tirage des splits des variables
    indexs = np.random.permutation(p)
    S = np.array_split(indexs, M)
    
    for i in range(max_iter):
        
        # Stockage de w pour critère de convergence
        old_w = w.copy()
        
        DeltaW = np.zeros(p)
        
        Xms = (X[:, S[m]] for m in range(M))
        wms = (w[S[m]] for m in range(M))
                
        #Calculs parallélisés grâce au multiprocessing Python
        with ProcessPool(nodes=4) as P:
            map_function = lambda Xm, wm: partial_solver(Xm, y, wm, lmbd)
            DeltaWms = P.map(map_function, Xms, wms)
        
        # Calcul de DeltaW
        for m in range(M):
            DeltaW[S[m]] += DeltaWms[m]
        
        # Recherche de alpha (Algorithme 3 de l'article)
        loss = objectif(yX, w + DeltaW, lmbd)
        if loss<objectif_list_[i]:
            alpha = 1
        else:
            alpha = line_search(yX, w, DeltaW, lmbd)
                
        # Mise à jour de w
        w += alpha*DeltaW
        
        # Evaluation de la fonction objectif à minimiser
        objectif_list_[i+1] = objectif(yX, w, lmbd)

        # Critère d'arrêt de convergence
        if (np.max(np.abs(w - old_w)) < tol):
            objectif_list_ = objectif_list_[:i+2]
            break

    coef_ = w

    return coef_, objectif_list_


@njit
def predict(X, coef_):
    """
    Prédictions de la régression logistique
    
    X: X matrice des observations (exemples x variables)
    coef_: vecteur des coefficients de la régression logistique
    
    return: vecteur des prédictions
    """
    return np.sign(sigmoid(X.dot(coef_))-0.5)