from fastapi import APIRouter, HTTPException, Response, status
from app.models.jd_chunk_model import JDChunk, JDChunkResponse, QALoad, VectorSearchResponse
from app.models.project_model import Project
from app.operations.jd_chunks_operations import cosine_similarity, generate_embedding, get_project_info, qa_embed
from bson import ObjectId
from app.models.api_response_model import ApiResponseModel
from app.config import jd_chunks_collection

router_chunk = APIRouter()

@router_chunk.get("/chunks/jd/{project_id}", response_description = "Get the project description", response_model=JDChunkResponse)
async def get_the_description(project_id:str, response : Response):
    try:
        project_detail = await get_project_info(project_id)
        return project_detail
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred: {str(e)}",
        )


@router_chunk.post("/chunks/qa")
async def ask_question(request: QALoad):
    """
    Handle a question: convert it to an embedding and find relevant chunks.
    """
    # Generate embedding for the question
    question_embedding = generate_embedding(request.question)

    # Retrieve embeddings for the specified project_id
    project_chunks = await jd_chunks_collection.find({"project_id": request.project_id}).to_list(None)
    if not project_chunks:
        raise HTTPException(status_code=404, detail="No chunks found for the given project ID.")

    # Compute similarities
    results = []
    for chunk in project_chunks:
        similarity = cosine_similarity(question_embedding, chunk["embedding"])
        results.append({
            "project_id": chunk["project_id"],
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "similarity": similarity
        })

    # Sort results by similarity in descending order
    results = sorted(results, key=lambda x: x["similarity"], reverse=True)

    return {"question": request.question, "results": results[:5]}  # Return top 5 results
