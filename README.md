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

## Lancement du Projet

Cette section décrit les étapes pour configurer l'environnement et lancer les différents composants du projet.

### 1. Prérequis

*   Python 3.8+
*   pip (généralement inclus avec Python)

### 2. Installation des dépendances

Clonez d'abord le dépôt si ce n'est pas déjà fait. Ensuite, ouvrez un terminal à la racine du projet et exécutez la commande suivante pour installer toutes les bibliothèques nécessaires listées dans `requirements.txt` :

```bash
pip install -r requirements.txt
```

### 3. Lancement du Backend (API FastAPI)

L'API FastAPI expose les modèles de deep learning. Pour la démarrer, exécutez la commande suivante depuis la racine du projet :

```bash
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```
*   `python -m uvicorn` : Exécute uvicorn en tant que module Python, ce qui est plus portable entre les environnements et les systèmes d\'exploitation.
*   `--reload` : Permet au serveur de redémarrer automatiquement après des modifications du code. Utile en développement.
*   `--host 0.0.0.0` : Rend l'API accessible depuis d'autres machines sur le réseau (et pas seulement en localhost).
*   `--port 8000` : Spécifie le port d'écoute (par défaut pour FastAPI avec uvicorn).

L'API sera alors accessible à l'adresse `http://localhost:8000` (ou `http://<votre-ip-locale>:8000`).

### 4. Lancement du Frontend (Gradio)

L'interface utilisateur Gradio permet d'interagir avec l'API. Pour la démarrer, exécutez la commande suivante depuis la racine du projet (en supposant que votre application Gradio se trouve dans `frontend/app.py`) :

```bash
python frontend/app.py
```

L'interface Gradio sera généralement accessible via une URL locale affichée dans le terminal (souvent `http://127.0.0.1:7860` ou similaire). Assurez-vous que le backend FastAPI est en cours d'exécution pour que le frontend puisse communiquer avec les modèles.

## Réponses aux Questions du TP

### Partie 1 : Classification de sentiment

#### 4.1 Questions

**1. Exploration du Hub et sélection de modèles :**

*   **Identification de 3 modèles différents adaptés à la classification de texte :**
    1.  `distilbert-base-uncased`
    2.  `bert-base-uncased`
    3.  `camembert-base`

*   **Analyse de leurs caractéristiques :**
    *   **`distilbert-base-uncased`**:
        *   **Taille**: Environ 66 millions de paramètres.
        *   **Langue**: Anglais (non sensible à la casse - "uncased").
        *   **Domaine d’entraînement**: Principalement BookCorpus et Wikipedia anglais.
        *   **Architecture**: Encodeur seul (Transformer DistilBERT).
    *   **`bert-base-uncased`**:
        *   **Taille**: Environ 110 millions de paramètres.
        *   **Langue**: Anglais (non sensible à la casse - "uncased").
        *   **Domaine d’entraînement**: Principalement BookCorpus et Wikipedia anglais.
        *   **Architecture**: Encodeur seul (Transformer BERT).
    *   **`camembert-base`**:
        *   **Taille**: Environ 110 millions de paramètres.
        *   **Langue**: Français.
        *   **Domaine d’entraînement**: Corpus français volumineux et varié (OSCAR).
        *   **Architecture**: Encodeur seul (Transformer RoBERTa-like).

*   **Justification du choix de 2 modèles pour votre expérience :**

    Pour cette expérience, nous choisirons **`distilbert-base-uncased`** et **`bert-base-uncased`**.
    *   **Comparaison directe**: Les deux modèles sont entraînés sur des corpus anglais similaires (BookCorpus et Wikipedia anglais), ce qui permet une comparaison plus directe de leurs performances sur des tâches en anglais.
    *   **Trade-off performance/ressources**: `distilbert-base-uncased` est une version distillée, plus petite et plus rapide de `bert-base-uncased`. Les comparer permettra d'analyser le compromis entre la taille/vitesse du modèle et ses performances. Ce critère est important étant donné la contrainte d'utilisation sur un PC classique avec une RTX 3070 8GB VRAM.
    *   **Popularité et documentation**: Ce sont des modèles très populaires et largement documentés, facilitant leur utilisation, le fine-tuning, et la recherche de solutions en cas de problème.

*   **Comparaison des architectures sous-jacentes (encoder-only vs encoder-decoder) :**

    Les trois modèles identifiés (`distilbert-base-uncased`, `bert-base-uncased`, `camembert-base`) sont des modèles de type **encodeur seul (encoder-only)**.

    *   **Architecture encodeur seul**:
        *   Ces modèles sont constitués d'une pile d'encodeurs Transformer. Leur rôle est de lire l'intégralité de la séquence d'entrée et de générer une représentation contextuelle riche de chaque token dans la séquence.
        *   Ils sont particulièrement bien adaptés aux tâches de compréhension du langage naturel (NLU) comme la classification de texte (notre cas ici), la reconnaissance d'entités nommées, ou l'extraction de réponses dans un contexte donné. Pour la classification, la représentation de sortie (souvent l'état caché du token spécial `[CLS]` ou une moyenne des états cachés des tokens de la séquence) est ensuite utilisée comme entrée pour une couche de classification simple.

    *   **Différence avec les architectures encodeur-décodeur**:
        *   Les modèles **encodeur-décodeur** (exemples : T5, BART, MarianMT pour la traduction) possèdent deux principales composantes : un encodeur qui traite la séquence d'entrée pour la transformer en une représentation continue, et un décodeur qui utilise cette représentation pour générer une séquence de sortie token par token.
        *   Ils sont nativement conçus pour les tâches de séquence à séquence (seq2seq) telles que la traduction automatique, le résumé de texte, ou la génération de texte conditionnelle.
        *   Bien qu'un modèle encodeur-décodeur puisse être adapté pour la classification (par exemple, en lui faisant générer le nom de la classe sous forme de texte : "positif", "négatif"), les modèles encodeur seul sont généralement plus directs, plus légers en termes de paramètres pour une tâche équivalente, et souvent plus performants pour les tâches de classification pure.

    *   **Pertinence pour la classification de texte**: Pour la classification de sentiment, l'objectif est de comprendre le sens global d'un texte pour lui assigner une étiquette. Une architecture encodeur seul est donc tout à fait appropriée et constitue le choix standard et le plus efficace pour cette tâche.


### Partie 2 : Génération de texte et question-réponse

#### 5.1 Questions

**(a) Sélection et comparaison de modèles génératifs :**

*   **Identification de 2 modèles adaptés à la génération de texte sur le Hub :**

    Pour la génération de texte, nous allons considérer les modèles suivants :
    1.  **`gpt2`** (ou sa version distillée `distilgpt2` pour une empreinte plus faible) : Un modèle auto-régressif basé sur l'architecture Transformer de type décodeur seul.
    2.  **`t5-small`** : Un modèle séquence à séquence (encodeur-décodeur) qui peut être utilisé pour la génération de texte conditionnelle.

*   **Comparaison de leurs approches : autoregressive vs seq2seq :**

    *   **Modèles Auto-régressifs (Decoder-Only, ex: GPT-2)**:
        *   **Principe**: Ces modèles génèrent du texte un token à la fois. Chaque nouveau token est prédit en se basant sur la séquence des tokens précédemment générés. Ils lisent la séquence de gauche à droite (pour les langues occidentales) et prédisent le token suivant.
        *   **Architecture**: Ils utilisent typiquement uniquement la partie décodeur d'une architecture Transformer. Le mécanisme d'attention leur permet de prendre en compte l'ensemble du contexte précédent lors de la génération de chaque nouveau token.
        *   **Cas d'usage**: Très efficaces pour la génération de texte libre (non conditionnée ou conditionnée par un prompt initial), la complétion de texte, l'écriture créative.
        *   **Exemple**: `gpt2` est un exemple canonique. Si on lui donne le prompt "Once upon a time, in a land far away,", il va prédire le token suivant, puis le suivant, en se basant à chaque fois sur tout ce qui a été généré jusqu'alors.

    *   **Modèles Séquence à Séquence (Encoder-Decoder, ex: T5-small)**:
        *   **Principe**: Ces modèles sont conçus pour transformer une séquence d'entrée en une séquence de sortie. L'encodeur traite la séquence d'entrée complète pour en créer une représentation (un état caché). Le décodeur utilise ensuite cette représentation pour générer la séquence de sortie, token par token, de manière auto-régressive (similaire à un modèle décodeur seul, mais conditionné par la sortie de l'encodeur).
        *   **Architecture**: Ils utilisent à la fois un encodeur et un décodeur.
        *   **Cas d'usage**: Idéaux pour les tâches où la sortie est une transformation de l'entrée, comme la traduction automatique (entrée : phrase en langue A, sortie : phrase en langue B), le résumé de texte (entrée : texte long, sortie : résumé court), la réponse à des questions (entrée : question + contexte, sortie : réponse). Pour la génération de texte plus "libre", on peut les utiliser en donnant un prompt à l'encodeur et en laissant le décodeur générer la suite.
        *   **Exemple**: `t5-small` est pré-entraîné sur une multitude de tâches en utilisant des préfixes spécifiques pour indiquer la tâche à effectuer (ex: "translate English to French: ...", "summarize: ..."). Pour la génération de texte à partir d'un prompt, on peut simplement donner le prompt comme entrée.

*   **Test des capacités de génération avec différents prompts (Exemples conceptuels) :**

    *   **Prompt 1 (Créatif)**: "Le dragon ouvrit un œil et dit à la princesse :"
        *   **`gpt2` (attendu)**: Pourrait générer une suite narrative, par exemple : "... 'Votre quête est noble, mais le chemin est semé d'embûches. Cherchez l'oracle de la montagne interdite.'"
        *   **`t5-small` (attendu, si utilisé pour la complétion)**: Pourrait générer quelque chose de similaire, peut-être plus concis ou factuel selon son entraînement, par exemple : "... 'Bonjour.'"

    *   **Prompt 2 (Factuel/Question simple)**: "La capitale de la France est"
        *   **`gpt2` (attendu)**: Devrait générer "Paris." et potentiellement continuer avec des informations liées si on le laisse générer plus de tokens.
        *   **`t5-small` (attendu)**: Devrait également générer "Paris." de manière concise, car il est entraîné sur des tâches de type question-réponse.

    *   **Prompt 3 (Instruction simple)**: "Écris un poème sur la lune."
        *   **`gpt2` (attendu)**: Pourrait générer quelques vers, dont la qualité poétique varierait. Exemple : "Astre pâle dans la nuit noire, / Tu veilles sur nos espoirs, / Silencieuse et lointaine, / Reine de la nuit sereine."
        *   **`t5-small` (attendu, si le prompt est bien formulé pour une tâche de génération)**: Pourrait aussi tenter de générer un poème, bien que sa force réside plus dans la transformation de tâches structurées. La qualité pourrait être plus variable pour une tâche aussi ouverte sans fine-tuning spécifique.

*   **Analyse de la qualité et de la cohérence des textes générés (Attendus) :**

    *   **`gpt2` / `distilgpt2`**:
        *   **Qualité**: Tendance à produire un texte grammaticalement correct et souvent plausible localement. La cohérence sur de longues séquences peut parfois se dégrader, avec des répétitions ou des dérives de sujet, surtout pour les versions plus petites comme `distilgpt2`.
        *   **Cohérence**: Bonne cohérence à court et moyen terme. Pour des textes plus longs, un prompt bien formulé et des techniques de décodage (comme le beam search, top-k/top-p sampling) peuvent aider à maintenir la cohérence. Peut parfois générer des informations factuellement incorrectes (hallucinations) car il est optimisé pour la plausibilité linguistique plutôt que la véracité.

    *   **`t5-small`**:
        *   **Qualité**: Lorsqu'utilisé pour des tâches pour lesquelles il a été explicitement entraîné (via les préfixes), la qualité est généralement bonne et factuelle. Pour la génération de texte plus libre à partir d'un prompt, la qualité peut être plus hétérogène sans fine-tuning spécifique. Le texte est souvent grammaticalement correct.
        *   **Cohérence**: Bonne cohérence pour les tâches structurées. Pour la génération libre, il peut être plus enclin à générer des phrases plus courtes ou à s'arrêter plus tôt que GPT-2 s'il n'est pas certain de la suite. Sa nature encodeur-décodeur le rend bon pour suivre les instructions implicites du prompt (s'il est formulé comme une tâche qu'il connaît).
        *   **Spécificité**: `t5-small` est un modèle plus petit de la famille T5, donc ses capacités de génération seront moins impressionnantes que les versions plus grandes (T5-base, T5-large). Il est cependant plus gérable en termes de ressources.

    **Note**: Ces analyses sont basées sur les caractéristiques générales des architectures et des modèles. Les résultats réels dépendront fortement des prompts exacts, des paramètres de génération (température, top-k, etc.), et d'un éventuel fine-tuning.