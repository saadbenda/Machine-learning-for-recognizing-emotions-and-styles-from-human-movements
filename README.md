# Machine learning for recognizing emotions and styles from human movements
 
Résume — Ce projet présente la mise en place d’algorithmes 
de Machines Learning et de Deep Learning afin de reconnaitre et 
identifier  les  personnes  qui  réalisent  des  actions  données.  On 
utilise notamment des réseaux de neurones convolutifs (CNN) et 
récursifs  (LSTM)  ainsi  qu’un  algorithme  basique  des  K  plus 
proches voisins (KNN).  
Keywords—  Reconnaissance  de  style/sujet  ;  KNN,  CNN, 
LSTM, Séries temporelles.

# I.  INTRODUCTION 
 
  La  vision  par  ordinateur  a  fait  d’énormes  progrès  et  les 
domaines  d’utilisation  ont  en  même  temps  augmentés.  On 
l’utilise  actuellement  dans  des  domaines  tels  que  la  vidéo-
surveillance,  la  reconnaissance  faciale  etc.  Ces  progrès  sont 
possibles sont rendus possibles grâce à des cameras tels que le 
Microsoft Kinect permettant de générer des squelettes en 3D en 
se  basant  sur  les  différentes  positions  des  articulations  du 
squelette, et cela en intégrant le point de vue. 
  C’est ce type de données numériques qui sont présentes dans 
nos trois bases de données d’étude, à savoir MSR Action 3D, 
Dance  Motion  Capture  Database  et  Emotional  Body  Motion 
Database qui contiennent des coordonnées de joints. 
  Outre  la  reconnaissance  d'action  qui  est  actuellement  très 
utilisés avec des algorithmes performants, la reconnaissance de 
style et de sujets dans des séquences de vidéos quant à elle n'a 
pas encore vraiment été analysé pourtant cela revêt d'une grande 
importance pour une reconnaissance plus approfondie.  
Cette reconnaissance plus approfondie passe par une analyse et 
une capacité plus approfondie sur l'analyse d'action étant donné 
que deux personnes n'ont pas la même gestuelle pour effectuer 
une action.  Et c’est justement ce sur quoi porte notre projet. 
 
# II.  OBJECTIFS 
L’objectif de ce projet est de pouvoir mettre en place des 
outils  de  Machine  Learning  (ML)  permettant  non  pas  de 
reconnaitre des actions mais plutôt reconnaitre les personnes qui 
effectuent ces actions.
On  parle  ici  donc  de  reconnaissance  de  style  car  chaque 
personne  possède  un  style  particulier  dans  l’exécution  de  ses 
mouvements.  
Et  donc  bien  que  les  mouvements  réalisés  par  les  deux 
personnes soient très proches et mêmes indiscernables pour un 
humain,  l’enjeu  est  donc  de  mettre  des  outils  informatiques 
permettant de le faire. 
### A.  Reconnaissance de style et d’émotions à partir d’un algorithme KNN (K Nearest Neighbors). 
L’idée  est  de  créer  un  algorithme  assez  basique  basé  sur 
KNN afin de pouvoir prédire les sujets réalisant ces actions. Le 
choix de cette méthode bien que n’étant pas la plus adaptée pour 
ce genre de problématique est utilisé dans tout le projet qu’afin 
de faire des analyses et des comparaisons entre ces algorithmes 
basiques  et  des  algorithmes  plus  évolués  tels  que  les 
Convolutional  Neural  Network  (CNN)  et  Long  Short-Term 
Memory (LSTM). Cet algorithme est ainsi donc appliqué sur les 
trois bases de données. 
 
### B.  Reconnaissance de style et d’émotions basés sur des algorithmes de Deep Learning. 
Après une première prédiction basée sur l’algorithme KNN, 
ce  qui  est  demandé  par  la  suite  c’est  la  construction 
d’architectures de  réseaux de  neurones  afin d’approfondir les 
analyses. 
 
# III.  REALISATION DU PROJET 
### A.  Prise en main des bases de données et gestion des données 

La  première  étape  de  ce  projet  est  la  récupération  des 
données sur internet et la constitution des différentes bases de 
données. Cette extraction de fichiers a été automatisée grâce à 
un script Javascript permettant de toutes les télécharger. Puis il 
a fallu nettoyer les données. 

![img1](./doc/img1.jpg)

Les fichiers dans les bases de données Emotional Body Motion 
(EBMDB)  et  Dance  Motion  Capture ont  pour  extension 
Biovision Hierarchy (BVH) organisé en arbre. Tandis que dans 
MSR Action 3D il s’agit de simple fichier txt.   
