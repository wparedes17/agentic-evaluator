from pydantic import BaseModel, Field
from typing import Optional


class EvaluationRequest(BaseModel):
    question: str = Field(..., description="The question that was asked")
    guideline: str = Field(..., description="The evaluation guideline to assess the response against")
    endpoint_url: str = Field(..., description="URL to POST the question to and retrieve a response")
    endpoint_question_key: str = Field(
        default="question",
        description="The JSON body key to use when sending the question to the endpoint",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "What is the capital of France?",
                "guideline": "The response should be factually correct, concise, and mention Paris.",
                "endpoint_url": "http://localhost:9000/ask",
                "endpoint_question_key": "question",
            }
        }
    }


class EvaluationResponse(BaseModel):
    question: str
    guideline: str
    endpoint_url: str
    raw_response: str = Field(..., description="The raw response received from the endpoint")
    evaluation: str = Field(..., description="Full evaluation text from the agent")
    score: Optional[float] = Field(None, description="Score from 0 to 10")
    passed: Optional[bool] = Field(None, description="Whether the response passed the guideline check")
