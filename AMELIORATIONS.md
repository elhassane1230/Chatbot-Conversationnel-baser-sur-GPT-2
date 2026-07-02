# Récapitulatif des améliorations du notebook

Ces changements ont été appliqués à `Chatbot-Medical.ipynb`. Le notebook n'a pas
pu être ré-exécuté hors de Colab (dataset `QA.json` et GPU requis) : relancez-le
une fois de bout en bout pour valider.

1. **`generate_text` corrigé** — amorçage avec `<sep>` (cohérent avec
   l'entraînement) + extraction de la seule réponse + `pad_token_id` défini.
2. **Cellule d'évaluation BLEU/ROUGE ajoutée** — `evaluate(sample_size)` mesure la
   qualité sur la validation (les imports étaient présents mais inutilisés).
3. **Reproductibilité** — `SEED = 42` (random/numpy/torch) et `random_state` sur
   `train_test_split`.
4. **Imports nettoyés** — suppression de `tensorflow` (projet PyTorch) et des
   doublons ; installations regroupées.
5. **Interface Gradio améliorée** — libellés, exemples, avertissement médical.
