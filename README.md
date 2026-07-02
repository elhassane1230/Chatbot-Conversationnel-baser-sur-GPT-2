# Chatbot Médical — GPT-2 fine-tuné (PyTorch)

Un chatbot médical conversationnel obtenu par **fine-tuning de GPT-2** sur un jeu
de questions/réponses de santé, avec analyse exploratoire des données, entraînement
via 🤗 Transformers, évaluation automatique (BLEU / ROUGE) et une interface de
démonstration **Gradio**.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Framework](https://img.shields.io/badge/GPT--2-Transformers-orange)
![DL](https://img.shields.io/badge/PyTorch-red)
![UI](https://img.shields.io/badge/demo-Gradio-yellow)

> Projet académique — module de **fouille de données**, encadré par le Pr. **Lebbah
> Mustapha**. ⚠️ *Ce chatbot est un travail pédagogique : il ne fournit pas d'avis
> médical et ne remplace en aucun cas un professionnel de santé.*

---

## Aperçu du projet

Le projet reprend le modèle de langage **GPT-2** pré-entraîné et l'adapte
(fine-tuning) à un corpus médical de paires *question → réponse*. Le pipeline
complet, dans le notebook, suit ces étapes :

1. **Nettoyage des données** — normalisation du texte (caractères spéciaux, espaces)
   à partir d'un fichier `QA.json`.
2. **Analyse exploratoire (EDA)** — distributions des longueurs de questions et de
   réponses, fréquences de mots, diversité lexicale, analyse de sentiment
   (TextBlob), similarité TF-IDF / cosinus.
3. **Fine-tuning de GPT-2** — les paires sont concaténées au format
   `question <sep> réponse`, tokenisées, puis le modèle est entraîné avec le
   `Trainer` de Hugging Face (3 époques).
4. **Évaluation** — métriques **BLEU** et **ROUGE** sur l'ensemble de validation.
5. **Déploiement de démonstration** — une interface **Gradio** pour dialoguer avec
   le modèle.

## Jeu de données

Le corpus est un fichier `QA.json` structuré en intentions :

```json
{
  "intents": [
    { "pattern": "What are the symptoms of the flu?",
      "response": "Common flu symptoms include fever, cough, sore throat, ..." }
  ]
}
```

Chaque entrée fournit une question (`pattern`) et sa réponse (`response`). Placez
votre `QA.json` à la racine (ou adaptez le chemin `filepath` dans le notebook).

## Modèle et entraînement

| Élément | Choix |
|---|---|
| Modèle de base | `gpt2` (Hugging Face) |
| Format d'entraînement | `question <sep> réponse` |
| Découpage | 90 % entraînement / 10 % validation |
| Époques | 3 |
| Batch size | 4 |
| Optimisation | warmup 500 pas, weight decay 0.01 |
| Génération | échantillonnage `top-k=50`, `top-p=0.95`, `repetition_penalty=1.2` |

## Installation

```bash
pip install "transformers[torch]" accelerate gradio rouge textblob \
            scikit-learn matplotlib seaborn nltk pandas
```

Le notebook est prévu pour **Google Colab** (montage Google Drive pour sauvegarder
le modèle). En local, remplacez les chemins Colab (`/content/…`, Google Drive) par
des chemins locaux.

## Utilisation

1. Ouvrez `Chatbot-Medical.ipynb` (Colab ou Jupyter).
2. Placez `QA.json` à l'emplacement attendu.
3. Exécutez les cellules dans l'ordre : nettoyage → EDA → tokenisation →
   entraînement → évaluation.
4. Lancez la dernière cellule pour ouvrir l'interface **Gradio** et discuter avec
   le chatbot.

Génération par programme :

```python
generate_text("What are the symptoms of the flu?")
# -> réponse générée par le modèle
```

## Évaluation

Le modèle est évalué sur l'ensemble de validation avec :

- **BLEU** — recouvrement en n-grammes entre la réponse générée et la référence ;
- **ROUGE-1 / ROUGE-L** — rappel de mots et de plus longues sous-séquences communes.

La fonction `evaluate(sample_size=...)` du notebook calcule ces métriques et
affiche un récapitulatif.

## Structure

```
.
├── Chatbot-Medical.ipynb   # pipeline complet (données → modèle → démo)
├── QA.json                 # corpus questions/réponses (à fournir)
└── README.md
```

## Améliorations récentes

Le notebook a été révisé pour corriger et enrichir le pipeline :

- **Cohérence entraînement/inférence** : la génération amorce désormais le modèle
  avec le séparateur `<sep>` utilisé à l'entraînement et ne renvoie que la partie
  *réponse* (auparavant, le prompt brut était passé sans séparateur).
- **Évaluation ajoutée** : BLEU et ROUGE étaient importés mais inutilisés ; une
  cellule d'évaluation quantitative a été ajoutée.
- **Reproductibilité** : graine aléatoire fixée et `random_state` sur le découpage.
- **Nettoyage des dépendances** : suppression de l'import TensorFlow inutilisé
  (le projet est en PyTorch) et dédoublonnage des imports.
- **Interface** : démo Gradio étoffée (libellés, exemples, avertissement médical).

## Limites & pistes

- Corpus et nombre d'époques modestes → réponses parfois génériques ou imprécises.
- GPT-2 n'a aucune garantie de véracité médicale : **usage pédagogique uniquement**.
- Pistes : corpus plus large, modèle plus récent, filtrage de sûreté des réponses,
  garde-fous et citations de sources.

## Équipe

Projet réalisé par **Assellaou Mouhcine**, **Bouyahyaoui Soufiane** et
**Talha Elhassane**, sous la supervision du Pr. **Lebbah Mustapha**
(module de fouille de données).
