import vertexai
from vertexai.generative_models import GenerativeModel
import re


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


def test_model(endpoint, prompt):
    model = GenerativeModel(endpoint)

    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0,
            "max_output_tokens": 500,
        },
    )

    return extract_species(response.text)


def main():

    vertexai.init(
        project=PROJECT_ID,
        location=LOCATION,
    )

    # Test sample
    v1_prompt = (
        "sepal_length: 5.0, "
        "sepal_width: 3.5, "
        "petal_length: 1.3, "
        "petal_width: 0.3"
    )

    v2_prompt = (
        "A flower specimen has a sepal length of 5.0 cm, "
        "sepal width of 3.5 cm, "
        "petal length of 1.3 cm, "
        "and petal width of 0.3 cm. "
        "Identify the iris species."
    )

    print("=" * 60)
    print("V1")
    print("=" * 60)

    try:
        result = test_model(V1_ENDPOINT, v1_prompt)
        print("Response:", repr(result))
    except Exception as e:
        print("V1 ERROR:", e)

    print()
    print("=" * 60)
    print("V2")
    print("=" * 60)

    try:
        result = test_model(V2_ENDPOINT, v2_prompt)
        print("Response:", repr(result))
    except Exception as e:
        print("V2 ERROR:", e)

    

def extract_species(text):
    match = re.search(r"(Iris\s+(setosa|versicolor|virginica))", text, re.IGNORECASE)
    if match:
        return match.group(1)
    return "Species not found"



if __name__ == "__main__":
    main()