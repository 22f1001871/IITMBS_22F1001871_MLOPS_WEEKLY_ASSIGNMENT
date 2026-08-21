import argparse
import os
import time

from google import genai
from google.genai import types


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--version",
        choices=["v1", "v2"],
        required=True,
    )

    parser.add_argument("--train-gcs", required=True)
    parser.add_argument("--validation-gcs", required=True)
    parser.add_argument("--output-gcs", required=True)
    parser.add_argument("--display-name", required=True)

    parser.add_argument(
        "--project",
        default=os.getenv("GOOGLE_CLOUD_PROJECT"),
    )

    parser.add_argument(
        "--location",
        default=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
    )

    parser.add_argument(
        "--base-model",
        default=os.getenv(
            "GEMINI_BASE_MODEL",
            "gemini-2.5-flash",
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--learning-rate-multiplier",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--adapter-size",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    if not args.project:
        raise ValueError(
            "Set GOOGLE_CLOUD_PROJECT or pass --project."
        )

    # Required for Gemini tuning through Vertex AI.
    os.environ["GOOGLE_GENAI_USE_ENTERPRISE"] = "True"

    print("=" * 60)
    print("Gemini Supervised Fine-Tuning")
    print("=" * 60)
    print(f"Version:                 {args.version}")
    print(f"Project:                 {args.project}")
    print(f"Location:                {args.location}")
    print(f"Base model:              {args.base_model}")
    print(f"Training dataset:        {args.train_gcs}")
    print(f"Validation dataset:      {args.validation_gcs}")
    print(f"Epochs:                  {args.epochs}")
    print(f"LR multiplier:            {args.learning_rate_multiplier}")
    print(f"Adapter size:             {args.adapter_size}")
    print("=" * 60)

    # Current Google Gen AI SDK.
    client = genai.Client(
        vertexai=True,
        project=args.project,
        location=args.location,
        http_options=types.HttpOptions(
            api_version="v1beta1"
        ),
    )

    training_dataset = types.TuningDataset(
        gcs_uri=args.train_gcs
    )
    
    validation_dataset = types.TuningValidationDataset(
        gcs_uri=args.validation_gcs
    )

    config = types.CreateTuningJobConfig(
        tuned_model_display_name=args.display_name,
        epoch_count=args.epochs,
        learning_rate_multiplier=args.learning_rate_multiplier,
        adapter_size=types.AdapterSize.ADAPTER_SIZE_EIGHT,
        validation_dataset=validation_dataset
    )

    print("\nSubmitting tuning job...")

    tuning_job = client.tunings.tune(
        base_model=args.base_model,
        training_dataset=training_dataset,
        config=config,
    )

    print("\nTuning job submitted successfully.")
    print(f"Job name: {tuning_job.name}")

    # Poll until completion.
    completed_states = {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
    }

    while tuning_job.state not in completed_states:
        print(f"Current state: {tuning_job.state}")

        time.sleep(30)

        tuning_job = client.tunings.get(
            name=tuning_job.name
        )

    print("\n" + "=" * 60)
    print("TUNING FINISHED")
    print("=" * 60)

    print(f"State: {tuning_job.state}")

    if tuning_job.state == "JOB_STATE_SUCCEEDED":
        print(
            f"Tuned model: "
            f"{tuning_job.tuned_model.model}"
        )

        print(
            f"Tuned endpoint: "
            f"{tuning_job.tuned_model.endpoint}"
        )

    else:
        print("Tuning did not succeed.")
        print(tuning_job)


if __name__ == "__main__":
    main()