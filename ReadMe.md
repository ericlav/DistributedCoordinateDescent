# Descente par coordonnée distribuée pour la Régression Logistique avec pénalisation L1

Ceci est le repo contenant le code du projet de Christos Katsoulakis et Eric Lavergne.

Il contient les scripts suivants : 
<ul>
<li>LogisticLasso.py : classe LogisticLasso qui implémente l'algorithme de descente de gradient proximal et de descente par coordonnée proximal dans le cas non distribué </li>
<li>dGLMNET_sequential.py : code de l'algorithme d-GLMNET implémenté de manière séquentielle  </li>
<li>dGLMNET_multiprocessing.py : code de l'algorithme d-GLMNET implémenté de manière distribuée avec des multiprocessus via le module pathos</li>
<li>dGLMNET_multithread.py : code de l'algorithme d-GLMNET implémenté de manière distribuée avec des multithreads via le module threading</li>
<li>dGLMNET_cython.pyx : code de l'algorithme d-GLMNET implémenté de manière distribuée avec des multithreads via le module cython</li>
<li>setup.py : script à exécuter pour compiler dGLMNET_cython.pyx (par exemple avec la commande "python setup.py build_ext --inplace" en ligne de commande sous réserve d'avoir installé un compilateur C++) </li>
</ul>

Ces différents scripts sont mis en oeuvre dans le notebooks Demo_principal_distribue.ipynb qui produit un jeu de données artificielles puis exécute et compare les différentes implémentations de l'algorithme d-GLMNET. Nous avons également  le notebook Demo_annexe_non_distribue.ipynb qui met en oeuvre et compare les algorithmes d'optimisation non distribués.