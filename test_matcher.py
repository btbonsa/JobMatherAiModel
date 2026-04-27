from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model (first run will download it)
model = SentenceTransformer("BAAI/bge-m3")

# Freelancers
freelancers = [
    "Java backend developer with Spring Boot, MySQL, REST APIs",
    "Frontend developer with React and Tailwind CSS",
    "Full stack developer with Node.js, Express, MongoDB"
]

# Job
job = "Looking for Java backend developer with Spring Boot and database experience"

# Encode
freelancer_embeddings = model.encode(freelancers)
job_embedding = model.encode([job])

# Similarity
scores = cosine_similarity(job_embedding, freelancer_embeddings)[0]

# Rank
results = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

print("\n🔥 Job Matching Results:\n")

for idx, score in results:
    print(f"Freelancer {idx}")
    print(f"Score: {round(score, 4)}")
    print(f"Text: {freelancers[idx]}\n")