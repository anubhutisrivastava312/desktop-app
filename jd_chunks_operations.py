from app.config import project_collection
from bson import ObjectId
from bson.errors import InvalidId
from app.models.jd_chunk_model import JDChunk, VectorSearchResponse
import nltk
from nltk.tokenize import sent_tokenize
from typing import List
from app.model_loader import embedding_model
import numpy as np
from app.config import jd_chunks_collection

nltk.download('punkt_tab')

async def get_project_info(project_id: str) -> str:
    try:
        object_id = ObjectId(project_id)
    except InvalidId:
        raise ValueError("Invalid project ID format")
    
    project = await project_collection.find_one({"_id": object_id})
    # jd = project['job_description'].get('content','')
    # chunks = chunk_text_nltk(jd)
    # documents = insert_chunks_with_embeddings(chunks)
    # print(documents)

    if project and 'job_description' in project:
        project_id = str(project["_id"])
        job_description_content = project['job_description'].get('content', '')
        save_document = await process_and_store_job_description(project_id, job_description_content)
        print(save_document)
        return{"success": "Saved"}
        
    else:
        raise ValueError(f"Project '{project_id}' not found.")


def chunk_text_nltk(text: str, max_chunk_size: int = 500) -> List[str]:
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def cosine_similarity(vector1, vector2):
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))


def generate_embedding(text: str) -> List[float]:
    embedding = embedding_model.encode(text)
    return embedding.tolist()


def insert_chunks_with_embeddings(chunks: List[str]) -> list:
  documents =[]
  for chunk in chunks:
    embedding = generate_embedding(chunk)
    document = {
        "chunk": chunk,
        "embedding": embedding
    }
    documents.append(document)
  return documents


async def process_and_store_job_description(project_id: str, job_description: str) -> str:
    chunks = chunk_text_nltk(job_description)

    existing_chunks = await jd_chunks_collection.count_documents({"project_id": project_id})
    chunk_id = existing_chunks + 1


    documents = []
    for chunk_text in chunks:

        embedding = generate_embedding(chunk_text)


        jd_chunk = JDChunk(
            project_id=project_id,
            chunk_id=chunk_id,
            text=chunk_text,
            embedding=embedding
        )


        documents.append(jd_chunk.dict())


        chunk_id += 1


    if documents:
        jd_chunks_collection.insert_many(documents)
    return "Job description processed and stored successfully."



def qa_embed(query:str)-> List[float]:
    embed_qa =   generate_embedding(query)
    return embed_qa



async def perform_vector_search(query_embedding, project_id, top_k=5, num_candidates=100, distance_metric="cosine")-> VectorSearchResponse:
    """
    Performs vector search on MongoDB and groups results by project_id.

    Args:
        query_embedding (list): The embedding vector to search with.
        project_id (str): The project ID to filter the search.
        top_k (int): Number of top results to return.
        num_candidates (int): Number of candidate vectors for optimization.
        distance_metric (str): Metric to calculate vector similarity ('cosine', 'euclidean', etc.).

    Returns:
        list: Aggregated results grouped by project_id.
    """
    pipeline = [
        # Filter by project_id
        {
            "$match": {
                "project_id": project_id
            }
        },
        # Vector search using Atlas Search
        {
            "$search": {
                "index": "embedding_vector_index",  # Use your vector index name
                "knnBeta": {
                    "vector": query_embedding,
                    "path": "embedding",
                    "k": top_k,
                    "numCandidates": num_candidates,
                    "distanceMetric": distance_metric
                }
            }
        },
        # Add score to documents
        {
            "$addFields": {
                "score": {"$meta": "searchScore"}
            }
        },
        # Group by project_id and calculate metrics
        {
            "$group": {
                "_id": "$project_id",
                "average_score": {"$avg": "$score"},
                "max_score": {"$max": "$score"},
                "chunks": {
                    "$push": {
                        "chunk_id": "$chunk_id",
                        "text": "$text",
                        "score": "$score"
                    }
                }
            }
        },
        # Sort by average_score in descending order
        {
            "$sort": {"average_score": -1}
        },
        # Limit results to top_k projects
        {
            "$limit": top_k
        }
    ]

    # Execute the pipeline
    results = list(jd_chunks_collection.aggregate(pipeline))
    return results
