import pickle

with open("word_embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

print("Total keys:", len(embeddings))
print("Sample keys:", list(embeddings.keys())[:10])

word = "student"

# try different formats safely
for key in [
    word,
    "cbow_" + word,
    "sgns_" + word,
    "skipgram_" + word
]:
    if key in embeddings:
        vec = embeddings[key]
        print(f"\nFOUND: {key}")
        print(word + " - " + ", ".join([f"{x:.4f}" for x in vec]))
        break
else:
    print("Word not found in any format")