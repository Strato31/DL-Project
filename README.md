# DL-Project

Le papier sujet : chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://arxiv.org/pdf/1606.04671
Le papier étudiant l'application à la reconnaissance des émotions : chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://arxiv.org/pdf/1706.03256



Le fichier demo.py est le seul fichier à lancer pour faire tourner la démo.
Bien s'assurer d'avoir les bibliothèques listées dans requirements.txt installées.
Les graphes tracés sont enregistrés dans le dossier courant.
La démo devrait fonctionner en l'état, mais il est possible de changer certains paramètres pour essayer :
- TASK_ORDER : qui détermine dans quel ordre on considère les 4 tâches modélisées
- NUM_TASKS : le nombre de tasks considérées (parmi 2, 3 ou 4)
- prune_percents : qui détermine les % de pruning dont on teste les performances

Que fait la démo ?
    Comparaison de 3 méthodes pour l'apprentissage multi-tâches
    On utilise le dataset emotion trouvable sur HuggingFace, qui contient des textes labellisés selon 6 émotions. 
    Les 6 émotions du dataset sont : 0=sadness, 1=joy, 2=love, 3=anger, 4=fear, 5=surprise

    Avec ce dataset, on se propose de réaliser jusqu'à 4 tâches de classification d'émotions :
    - Tâche 1 : Emotion binaire (négatif [0,3,4] versus positif [1,2,5])
    - Tâche 2 : Émotions négatives détail (sadness=0, anger=1, fear=2)
    - Tâche 3 : Émotions positives détail (joy=0, love=1, surprise=2)
    - Tâche 4 : Toutes émotions (6 classes)

    On compare 3 méthodes d'apprentissage multi-tâches :
    1. Modèles indépendants : Un modèle par tâche (témoin)
    2. Fine-tuning : Entraîner sur Tâche 1 puis fine-tuner sur Tâche 2 (, puis sur Tâche 3, puis Tâche 4)
    3. PNN : Progressive Neural Network avec connexions latérales

    Enfin, en ouverture, on teste l'impact du pruning sur le modèle PNN entraîné.



Le fichier PNN.py est largement inspiré du code au lien suivant : https://github.com/Natializd/ProgressiveNeuralNetwork
Le code en lui-même se base sur le papier Progressive Neural Networks qui est le sujet de ce travail.

La license du code repris : 
MIT License

Copyright (c) 2023 Petro Fagurel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.