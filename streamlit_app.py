from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)

from src.config import load_config
from src.data.dataloader import build_transforms

LOGGER = logging.getLogger(__name__)


@st.cache_resource
def get_pipeline_config(config_path: str):
    pipeline_config = load_config(Path(config_path))
    return pipeline_config


def load_label_names(data_config) -> Tuple[List[int], Dict[int, str]]:
    if not Path(data_config.csv_path).exists():
        raise FileNotFoundError(
            f"Dataset CSV not found at '{data_config.csv_path}'. Update data.csv_path in config."
        )

    df = pd.read_csv(
        data_config.csv_path,
        usecols=[data_config.label_column, data_config.label_names_column],
    ).dropna()

    mapping = (
        df.drop_duplicates(subset=[data_config.label_column])
        .set_index(data_config.label_column)[data_config.label_names_column]
        .to_dict()
    )
    sorted_ids = sorted(mapping.keys())
    return sorted_ids, mapping


def prepare_image(image: Image.Image, image_size: int) -> np.ndarray:
    _, eval_transform = build_transforms(image_size=image_size)
    tensor = eval_transform(image).unsqueeze(0)
    return tensor.numpy()


def fetch_prediction_service(
    pipeline_name: str,
    step_name: str,
    running: bool,
    wait_seconds: int,
):
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()
    services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=step_name,
        running=running,
    )
    if not services:
        return None

    service = services[0]
    try:
        service.start(timeout=wait_seconds)
    except Exception:  # pragma: no cover - already running
        LOGGER.debug("Service already running.")

    try:
        if hasattr(service, "wait_for_service_ready"):
            service.wait_for_service_ready(timeout=wait_seconds)
        else:
            service.wait(timeout_seconds=wait_seconds)
    except Exception as exc:  # pragma: no cover - external service failure
        LOGGER.warning("Service did not confirm readiness: %s", exc)

    return service


def main() -> None:
    st.set_page_config(page_title="Animal Classifier", layout="centered")
    st.title("🐾 Animal Classifier Demo")
    st.caption(
        "Upload an image and get predictions from the latest deployed MLflow model."
    )

    config_path = "config.yaml"

    try:
        pipeline_config = get_pipeline_config(config_path)
    except Exception as exc:
        st.error(f"Failed to load configuration: {exc}")
        return

    data_config = pipeline_config.data
    inference_config = pipeline_config.inference

    try:
        label_ids, id_to_name = load_label_names(data_config)
    except Exception as exc:
        st.error(f"Unable to load label names: {exc}")
        return

    uploaded_file = st.file_uploader(
        "Select an image",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        st.info("Upload an image to run inference.")
        return

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as exc:
        st.error(f"Unable to open the uploaded image: {exc}")
        return

    st.image(image, caption="Uploaded Image")

    with st.spinner("Contacting prediction service..."):
        image_array = prepare_image(image, data_config.image_size)
        service = fetch_prediction_service(
            pipeline_name=inference_config.pipeline_name,
            step_name=inference_config.pipeline_step_name,
            running=inference_config.running,
            wait_seconds=inference_config.service_wait_seconds,
        )

        if service is None:
            st.error(
                "No running MLflow prediction service was found. Run the deployment pipeline first."
            )
            return

        try:
            predictions = np.asarray(service.predict(image_array))
        except Exception as exc:
            st.error(f"Prediction request failed: {exc}")
            return

    if predictions.ndim != 2 or predictions.shape[1] == 0:
        st.error("Prediction response has unexpected shape.")
        return

    probabilities = predictions[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_label_id = (
        label_ids[predicted_index]
        if predicted_index < len(label_ids)
        else predicted_index
    )
    predicted_label = id_to_name.get(predicted_label_id, str(predicted_label_id))
    confidence = float(probabilities[predicted_index])

    st.success(
        f"Predicted class: **{predicted_label}** ({confidence * 100:.2f}% confidence)"
    )

    st.subheader("Raw probabilities")
    prob_table = {
        id_to_name.get(label_id, str(label_id)): f"{prob * 100:.2f}%"
        for label_id, prob in zip(label_ids, probabilities)
    }
    st.json(prob_table)


if __name__ == "__main__":
    main()
