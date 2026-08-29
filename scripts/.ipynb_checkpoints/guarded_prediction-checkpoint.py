import vertexai
from vertexai.generative_models import GenerativeModel

from input_guardrails import InputGuardrail
from output_guardrails import OutputGuardrail


PROJECT_ID = "iitm-mlops-assignments"
LOCATION = "us-central1"

V2_ENDPOINT = (
    "projects/888678748237/"
    "locations/us-central1/"
    "endpoints/9061634426164740096"
)

V1_ENDPOINT = (
    "projects/888678748237/"
    "locations/us-central1/"
    "endpoints/1382997061498044416"
)


# ============================================================
# Initialize Vertex AI
# ============================================================

vertexai.init(
    project=PROJECT_ID,
    location=LOCATION
)


# ============================================================
# Guardrails
# ============================================================

input_guardrail = InputGuardrail()
output_guardrail = OutputGuardrail()


# ============================================================
# Prediction with complete guardrail pipeline
# ============================================================

def predict_with_guardrail(
    model_version,
    raw_input
):

    # ========================================================
    # STEP 1 — INPUT GUARDRAIL
    # ========================================================

    validation = input_guardrail.validate(
        raw_input
    )

    if validation["blocked"]:

        return {
            "blocked": True,
            "filtered": False,
            "reason": validation["reason"],
            "response": None
        }


    # ========================================================
    # STEP 2 — SELECT MODEL
    # ========================================================

    if model_version == "v1":

        endpoint = V1_ENDPOINT

    elif model_version == "v2":

        endpoint = V2_ENDPOINT

    else:

        raise ValueError(
            "model_version must be 'v1' or 'v2'"
        )


    # ========================================================
    # STEP 3 — CREATE CLASSIFICATION PROMPT
    # ========================================================

    prompt = (
        f"sepal_length: {raw_input['sepal_length']}, "
        f"sepal_width: {raw_input['sepal_width']}, "
        f"petal_length: {raw_input['petal_length']}, "
        f"petal_width: {raw_input['petal_width']}. "
        f"Identify the iris species."
    )


    # ========================================================
    # STEP 4 — CALL VERTEX AI
    # ========================================================

    model = GenerativeModel(endpoint)

    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0,
            "max_output_tokens": 500,
        }
    )

    raw_response = response.text


    # ========================================================
    # STEP 5 — OUTPUT GUARDRAIL
    # ========================================================

    filtered_result = output_guardrail.filter(
        raw_response
    )


    # ========================================================
    # STEP 6 — RETURN SAFE RESPONSE
    # ========================================================

    return {
        "blocked": False,
        "filtered": filtered_result["filtered"],
        "reason": filtered_result["reason"],
        "response": filtered_result["response"]
    }