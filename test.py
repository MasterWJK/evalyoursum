from bert_score import BERTScorer

scorer = BERTScorer(lang="en")

# Example reference and candidate sentences
references = ["The cat is on the mat."]
candidates = ["The cat is sitting on the mat."]

# Calculate BERT scores
P, R, F1 = scorer.score(candidates, references)

# Print the scores
print(f"Precision: {P.item():.4f}")
print(f"Recall: {R.item():.4f}")
print(f"F1 score: {F1.item():.4f}")