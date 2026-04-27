from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()
model = SentenceTransformer("BAAI/bge-m3")

class MatchRequest(BaseModel):
    jobs: list[str]  # Must be plural to match Node.js change
    talent: str      # Must be singular to match Node.js change

@app.post("/match-jobs")
def match(req: MatchRequest):
    try:
        # 1. Encode the single talent string
        talent_embedding = model.encode([req.talent])

        # 2. Encode the list of job strings
        job_embeddings = model.encode(req.jobs)

        # 3. Calculate similarity
        scores = cosine_similarity(talent_embedding, job_embeddings)[0]

        results = sorted(
            list(enumerate(scores)),
            key=lambda x: x[1],
            reverse=True
        )

        # 4. Map back to the list
        return [
            {
                "jobId": idx,
                "score": float(score),
                "text": req.jobs[idx] # Changed from 'freelancers' to 'jobs'
            }
            for idx, score in results
        ]
    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}
