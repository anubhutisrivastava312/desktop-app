from typing import List
from pydantic import BaseModel

class JDChunk(BaseModel):
    project_id:str
    chunk_id:int
    text:str
    embedding:List[float]

class JDChunkResponse(BaseModel):
    success: str

class QAText(BaseModel):
    question:str
    answer:str

class QALoad(BaseModel):
    project_id:str
    question:str
    
    
class ChunkResponse(BaseModel):
    chunk_id: int
    text: str
    similarity: float

class Chunk(BaseModel):
    chunk_id: int
    text: str
    score: float


class ProjectResult(BaseModel):
    _id: str
    average_score: float
    max_score: float
    chunks: List[Chunk]

class VectorSearchResponse(BaseModel):
    results: List[JDChunk]

class QAChunk(BaseModel):
    project_id:str
    chunk_id:int
    section_name:str
    tags:str
    qa_text:QAText
    embedding:List[float]


