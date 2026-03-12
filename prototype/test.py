from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from sentence_transformers.quantization import quantize_embeddings

# Can also have trucate_dim=512 to keep sizes at 512 or something
model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", local_files_only=True)
query = "A man is eating a piece of bread"
docs = [
    "A man is eating food.",
    "A man is eating pasta.",
    "The girl is carrying a baby.",
    "A man is riding a horse.",
]

query_embedding = model.encode(query, prompt_name="query")
docs_embeddings = model.encode(docs)
similarities = cos_sim(query_embedding, docs_embeddings)

print('similarities:', similarities)
