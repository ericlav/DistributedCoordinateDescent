# cython: profile=False, boundscheck=False, wraparound=False, nonecheck=False
import numpy as np
from numpy.linalg import norm
from numpy.random import multivariate_normal
from numpy.random import randn
from scipy.linalg.special_matrices import toeplitz
#libraires specifiques à cython et au c
cimport cython
from libc.math cimport exp as c_exp
from libc.math cimport fabs as c_abs
from libc.stdlib cimport malloc, free
from cython.parallel import prange
from libc.stdio cimport printf
#import logging

"""
Implémentation de dGLMNET sous Cython
Descente de coordonnées par blocs pour l'apprentissage
d'une régression logistique avec régularisation L1
"""


@cython.cdivision(True)
@cython.wraparound(False)
cdef float sigmoid(float t) nogil:
    """
    Sigmoid sous Cython
    
    t: float
    
    return: sigmoid(t)
    """
    return 1. / (1. + c_exp(-t))


def np_sigmoid(z):
    """
    Sigmoid avec numpy
    
    z: réel ou objet numpy
    
    return: sigmoid(z)
    """
    return 1. / (1. + np.exp(-z))


def objectif(y, X, w, lmbd):
    """
    Objectif de l'apprentissage (log-vraisemblance négative)
    
    y: vecteur des classes à prédire
    X: matrice des observations (exemples x variables)
    w : vecteur des coefficients du modèle
    lmbd: réel, paramètre de régularisation
    
    return: évaluation de l'objectif pour les paramètres en entrée
    """
    return np.log(1. + np.exp(-y * X.dot(w))).sum() + lmbd * norm(w, ord=1)


@cython.wraparound(False)
cdef float prox_L1(float t, float lmbd) nogil:
    """
    Proximal associé à la régularisation L1 sous Cython (soft thresholding)
    
    t: float
    lmbd: float, paramètre de régularisation
    
    return: proximal évalué en t et lmbd
    """
    if t >= 0:
        return max(0, c_abs(t) - lmbd)
    else :
        return min(0, lmbd - c_abs(t))

    
@cython.wraparound(False)
def dGLMNET_solver(X, y, lmbd, M=2, max_iter=100, tol=1e-6):
    """
    Apprentissage de la régression logistique avec régularisation L1
    par le biais de la méthode dGLMNET
    descente de coordonnées par blocs distribuée
    Programmée sous Cython avec la parallelisation prange
    avec nogil
    
    X: matrice des observations (exemples x variables)
    y: vecteur des classes à prédire
    M: entier, nombre de blocs de la descente par coordonnées
    max_iter: entier, nombre d'itérations maximal à exécuter
    tol: réel, seuil de convergence sous lequel arrêter l'algorithme
    
    return: vecteur des coefficients appris et vecteur des valeurs
    de la fonction objectif évaluée à chaque itération
    """
    
    #logging.basicConfig(filename='cythondGLMNET.log', level=logging.INFO)
    #logging.info("debut dglmnet")
    
    # n nombre de données, p nombre de features
    cdef int n = X.shape[0]
    cdef int p = X.shape[1]
    
    # Initialisation des paramètres à 0
    w = np.zeros(p, dtype=np.float32)
    DeltaW = np.zeros(p, dtype=np.float32)

    # Evolution de l'objectif au cours des itérations
    objectif_list_ = np.zeros(max_iter + 1, dtype=np.float32)
    
    # Objectif initial
    objectif_list_[0] = objectif(y, X, w, lmbd)
    
    #static typing des itérateurs 
    cdef int i
    cdef int a
    cdef int m
    
    #static typing des paramètres d'entrée pour adapter à Cython
    #(par exemple, les tableaux numpy sont des objets Python qui
    #ne permettent pas d'utiliser l'option nogil)
    cdef int Mc = M
    cdef float lmbdc = lmbd
    cdef float [::1] wc = w
    #La matrice des observations est transposée pour accélérer les calculs par la suite
    cdef float [:,::1] XcT = np.ascontiguousarray(X.T, dtype=np.float32)
    cdef float [::1] yc = np.ascontiguousarray(y, dtype=np.float32)
    cdef int pas = (p/M)
    #Allocation d'un tableau de float pour les incrémentations des coefficients
    cdef float * DeltaWcs = <float *> malloc(p * sizeof(float))

    alpha = 1
    
    #Boucle d'apprentissage
    for i in range(max_iter):
        
        # Stockage de w pour critère de convergence
        old_w = w.copy()
        
        #parallélisation des calculs par blocs de coordonnées avec Cython
        #l'option nogil permet de s'affranchir du gil pour aller plus vite
        for m in prange(Mc, nogil=True):
        #for m in range(Mc): #méthode sequentielle presque 3 fois plus longue
            #Appel à la fonction de descente de coordonnées pour un bloc donné
            DeltaWm_ref = partial_solver(&XcT[pas*m,0], n, pas, &yc[0], &wc[pas*m], lmbdc)
            
            #Aggrégation du vecteur partiel obtenu au vecteur complet des incrémentations
            for a in range(pas):
                DeltaWcs[pas*m + a] = DeltaWm_ref[a]
            free(DeltaWm_ref)
        
        for a in range(p):
            DeltaW[a] = DeltaWcs[a]

        #Mise a jour de w
        w += alpha*DeltaW
            
        # Evaluation de la fonction objectif à minimiser
        objectif_list_[i+1] = objectif(y, X, w + DeltaW, lmbd)
        
        # Critère d'arrêt de convergence
        if (np.max(np.abs(w - old_w)) < tol):
            objectif_list_ = objectif_list_[:i+2]
            break
    
    #Libération de la zone mémoire allouée
    free(DeltaWcs)
    
    coef_ = w

    return coef_, objectif_list_


#Calcul de la mise à jour partielle pour une machine m
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef float * partial_solver(float * XmcT, int n_, int pm_, float * yc, float * wmc, float lmbdc) nogil:
    """
    Résolution du problème d'optimisation pour un bloc donné
    Par méthode de quasi-newton (approximation de l'ordre 2)
    avec une passe unique sur les coordonnées (voire Trofimov et Genkin)
    implémentée sous Cython de façon à pouvoir se passer du GIL
    
    XmcT: adresse de la matrice transposée des observations (variables x exemples)
    pour le bloc de variable considéré
    n_: entier, nombre d'observations dans la matrice
    pm_: entier, nombre de variables dans le bloc considéré
    yc: adresse du vecteur des classes à prédire
    wmc: adresse du vecteur des coefficients actuels pour les variables du bloc
    lmbdc: paramètre de régularisation
    
    return : vecteur des incrémentations des coefficients pour les variables du bloc
    """
    
    #static typing des itérateurs
    cdef int j
    cdef int k
    cdef int l
    
    cdef int n = n_
    cdef int pm = pm_
    
    #Allocation d'un tableau de float pour les incrémentations des coefficients
    #Finalement on aurait pu directement passer en argument de partial_solver l'adresse du tableau complet
    cdef float* DeltaWm = <float *> malloc(pm * sizeof(float))
    for j in range(pm):
        DeltaWm[j] = 0
    
    #Allocations de tableaux pour calculs intermédiaires
    #voir (4) dans Trofimov et Genkin
    cdef float * px = <float *> malloc(n * sizeof(float))
    cdef float * doublev = <float *> malloc(n * sizeof(float))
    cdef float * z = <float *> malloc(n * sizeof(float))
    #Variables temporaires pour réduire les appels mémoires
    cdef float temp
    cdef float pxk
    
    #Calculs intermédiaires
    for k in range(n):
        temp = 0
        for j in range(pm):
            temp += XmcT[j*n+k]*wmc[j]
        px[k] = sigmoid(temp)
    
    for k in range(n):
        doublev[k] = px[k]*(1-px[k])
        z[k] = (((yc[k]+1)/2) - px[k]) / doublev[k]
    
    cdef float * q = <float *> malloc(n * sizeof(float))
    
    #Variables temporaires pour réduire les appels mémoires
    cdef float prox
    cdef float DeltaWmXmck
    cdef float doublevqXmcj
    cdef float doublevXmcj2
    cdef float XmcTk
    cdef float doublevk
    
    #Passe unique de descente de coordonnées sur le bloc concerné
    #voir (6) dans Trofimov et Genkin
    for j in range(pm):
        
        for k in range(n):
            DeltaWmXmck = 0
            for l in range(pm):
                DeltaWmXmck += DeltaWm[l]*XmcT[l*n + k]
            q[k] = z[k] - DeltaWmXmck + (wmc[j] + DeltaWm[j])*XmcT[j*n + k]
        
        doublevqXmcj = 0
        doublevXmcj2 = 0
        for k in range(n):
            XmcTk = XmcT[j*n + k]
            doublevk = doublev[k]
            doublevqXmcj += doublevk*q[k]*XmcTk
            doublevXmcj2 += doublevk*(XmcTk*XmcTk)
        
        prox = prox_L1(doublevqXmcj, lmbdc)
        
        DeltaWm[j] += prox/doublevXmcj2 - wmc[j]
    
    #libère la zone mémoire allouée par les arrays temporaires
    free(doublev)
    free(px)
    free(q)
    free(z)
    
    return DeltaWm


def grad_sigmoid(y, X, w):
    """
    Evalue le gradient de la partie du sigmoid dans l'objectif
    
    y: vecteur des classes à prédire
    X: matrice des observations (exemples x variables)
    w: vecteur des coefficients
    
    return: gradient évalué avec les paramètres en entrée
    """
    return -(y*X.T*sigmoid(-y*X.dot(w))).sum(axis=1)


#Peu appelée en pratique
def line_search(y, X, w, DeltaW, lmbd):
    """
    Trouve le facteur alpha adéquat par linear search
    
    y: vecteur des classes à prédire
    X: matrice des observations (exemples x variables)
    w: vecteur des coefficients
    DeltaW: direction pour laquelle trouver alpha
    lmbd: paramètre de régularisation
    
    return: facteur alpha adéquat
    """
    alpha_init_list = np.linspace(0.1,1,19)
    alpha_init = 0.1
    f_min = objectif(y, X, w + alpha_init*DeltaW, lmbd)
    
    for x in alpha_init_list:
        f_new = objectif(y, X, w + x*DeltaW, lmbd)
        if f_min > f_new:
            f_min = f_new
            alpha_init = x
       
    cond = True
    b = 0.5
    sigma = 0.01
    alpha = alpha_init
    
    while cond:
        
        f_new = objectif(y, X, w + alpha*DeltaW, lmbd)
        D = grad_sigmoid(y, X, w).dot(DeltaW) + lmbd*(norm(w + DeltaW, ord=1) - norm(w, ord=1))
        
        if f_new < objectif(y, X, w, lmbd) + alpha * sigma * D:
            cond=False
        else:
            alpha *= b
        
    return alpha


def predict(X, coef_):
    """
    Prédictions de la régression logistique
    
    X: matrice des observations pour lesquelles prédire
    coef_: vecteur des coefficients de la régression logistique
    
    return: vecteur des prédictions
    """
    return np.sign(np_sigmoid(X.dot(coef_))-0.5)
