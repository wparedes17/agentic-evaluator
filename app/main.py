import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.config import settings
from app.evaluator import evaluate
from app.models import EvaluationRequest, EvaluationResponse

app = FastAPI(
    title="Agentic Response Evaluator",
    description=(
        "Fetches a response from an external endpoint and evaluates its quality "
        "against a guideline using a LangGraph-based ReflectionAgent."
    ),
    version="1.0.0",
)


async def _fetch_response(endpoint_url: str, question: str, question_key: str) -> str:
    """POST the question to the target endpoint and return the response text."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(endpoint_url, json={question_key: question})
            resp.raise_for_status()
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail=f"Timeout while contacting endpoint: {endpoint_url}")
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Endpoint returned {exc.response.status_code}: {exc.response.text[:200]}",
            )
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=f"Failed to reach endpoint: {exc}")

    data = resp.json()

    # Accept a plain string or any common response envelope
    if isinstance(data, str):
        return data
    for key in ("response", "answer", "text", "output", "result", "content"):
        if key in data and isinstance(data[key], str):
            return data[key]

    # Fall back to the full JSON as text
    return resp.text


@app.get("/health", summary="Health check")
async def health():
    return {"status": "ok"}


@app.post("/evaluate", response_model=EvaluationResponse, summary="Evaluate a response")
async def evaluate_endpoint(request: EvaluationRequest):
    """
    1. POSTs `question` to `endpoint_url` to obtain a response.
    2. Evaluates that response against `guideline` using a ReflectionAgent.
    3. Returns the evaluation, score (0-10), and pass/fail verdict.
    """
    raw_response = await _fetch_response(
        request.endpoint_url, request.question, request.endpoint_question_key
    )

    try:
        evaluation_text, score, passed = evaluate(
            question=request.question,
            guideline=request.guideline,
            response=raw_response,
            settings=settings,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {exc}")

    return EvaluationResponse(
        question=request.question,
        guideline=request.guideline,
        endpoint_url=request.endpoint_url,
        raw_response=raw_response,
        evaluation=evaluation_text,
        score=score,
        passed=passed,
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    return JSONResponse(status_code=500, content={"detail": str(exc)})
