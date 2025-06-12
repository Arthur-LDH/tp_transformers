*# tp_transformers

## Contexte du TP

Ce projet s'inscrit dans le cadre d'un Travail Pratique (TP) axé sur l'exploration et l'application des modèles de Transformers. L'objectif principal est de développer une solution de bout en bout, comprenant typiquement :
*   La sélection et l'expérimentation avec un modèle de Transformer adapté à une tâche spécifique (par exemple, classification de texte, génération de texte, question-réponse).
*   Le fine-tuning (ou l'entraînement) du modèle sur un jeu de données pertinent.
*   L'exposition des fonctionnalités du modèle entraîné via une API RESTful.
*   La création d'une interface utilisateur simple (frontend) pour interagir avec le modèle via l'API.

Ce TP vise à mettre en pratique les concepts clés liés aux Transformers, à leur déploiement et à leur intégration dans une application simple.

## Technologies Utilisées

Voici les principales technologies et outils envisagés pour ce projet :

*   **Deep Learning & Backend :**
    *   ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
    *   ![Hugging Face Transformers](https://img.shields.io/badge/🤗%20Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
    *   ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
*   **API :**
    *   ![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi&logoColor=white)
*   **Frontend :**
    *   ![Vue.js](https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vue.js&logoColor=%234FC08D)
*   **Gestion de version & Environnement :**
    *   ![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)
    *   ![Visual Studio Code](https://img.shields.io/badge/VS%20Code-007ACC?style=for-the-badge&logo=visualstudiocode&logoColor=white)

## Recherche de Modèles sur Hugging Face Hub

Lors de la recherche de modèles de Transformers adaptés pour ce TP, plusieurs facteurs ont été pris en compte, notamment la performance, la taille du modèle, et la compatibilité avec du matériel courant (ex: une carte graphique RTX 3070 avec 8GB de VRAM).

### Modèles Envisagés

*   **`distilbert-base-uncased`**:
    *   **Description**: Il s'agit d'une version distillée de BERT (Bidirectional Encoder Representations from Transformers). La distillation est une technique qui permet de réduire la taille d'un modèle tout en essayant de conserver une grande partie de ses performances. `distilbert-base-uncased` est significativement plus petit et plus rapide que `bert-base-uncased`.
    *   **Avantages**:
        *   **Taille réduite**: Moins de paramètres, ce qui signifie moins d'utilisation de mémoire VRAM et un chargement plus rapide.
        *   **Inférence rapide**: Convient bien pour des applications nécessitant des réponses rapides.
        *   **Bonnes performances**: Bien qu'étant plus petit, il maintient des performances respectables sur de nombreuses tâches de compréhension du langage naturel (NLU).
        *   **Compatibilité**: Fonctionne bien sur des GPU avec une VRAM modérée comme une RTX 3070 8GB.
    *   **Cas d'usage typiques**: Classification de texte, analyse de sentiment, question-réponse (après fine-tuning).
    *   **Remarques**: Le suffixe "uncased" signifie que le modèle ne fait pas de distinction entre les majuscules et les minuscules.

### Considérations pour le choix d'un modèle sur une RTX 3070 (8GB VRAM)

*   **Taille du modèle (Nombre de paramètres)**: Les modèles plus grands (ex: BERT-large, GPT-2 medium/large) peuvent dépasser la capacité de 8GB de VRAM, surtout pendant l'entraînement ou si la taille du batch est importante. Les modèles "base" ou "distilled" sont généralement plus adaptés.
*   **Quantification**: Certains modèles sont disponibles en versions quantifiées (ex: INT8). La quantification réduit la précision des poids du modèle (ex: de FP32 à INT8), ce qui diminue l'utilisation de la mémoire et peut accélérer l'inférence, parfois avec une légère perte de performance.
*   **Précision (FP16 vs FP32)**: L'utilisation de la précision mixte (FP16) pendant l'entraînement ou l'inférence peut réduire de moitié l'utilisation de la VRAM par rapport à la pleine précision (FP32), tout en accélérant les calculs sur les GPU compatibles.
*   **Taille du batch**: Lors de l'entraînement ou de l'inférence par batch, une taille de batch plus petite consomme moins de VRAM. Il faut trouver un équilibre entre la taille du batch et la stabilité/vitesse de l'entraînement.
*   **Longueur de séquence**: Les Transformers ont une consommation mémoire qui augmente quadratiquement avec la longueur de la séquence d'entrée. Des séquences plus courtes nécessitent moins de mémoire.

### Conclusion Préliminaire

Pour ce TP, commencer avec des modèles comme `distilbert-base-uncased` ou d'autres modèles BERT-like de taille "base" semble être une approche judicieuse. Ils offrent un bon compromis entre performance et ressources nécessaires. Il sera toujours possible d'explorer des modèles plus grands ou des techniques d'optimisation (quantification, pruning) si les besoins du projet évoluent et si les ressources le permettent.

Il est recommandé de consulter régulièrement le [Hugging Face Hub](https://huggingface.co/models) pour découvrir de nouveaux modèles ou des versions fine-tunées spécifiques à certaines tâches.