import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import joblib
import numpy as np

# Clases del modelo (ajusta si usaste otros nombres)
CLASS_NAMES = ["F0", "F1", "F2", "F3", "F4"]

# Dispositivo (normalmente CPU en servidor/PC)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_model_enb0(num_classes: int) -> nn.Module:
    """
    Crea la misma arquitectura EfficientNet-B0 que se usó para entrenar el modelo.
    IMPORTANTE: debe coincidir con la definición usada en Colab cuando generaste
    modelo_fibrosis_state_dict_cpu_enb0.joblib.
    """
    # Si al entrenar usaste weights=EfficientNet_B0_Weights.IMAGENET1K_V1,
    # puedes poner lo mismo aquí o None (el state_dict va a sobrescribir todo).
    model = models.efficientnet_b0(weights=None)

    # Reemplazar la capa final tal como lo hiciste al entrenar
    # (esta es la forma estándar que te propuse antes):
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


@st.cache_resource
def load_model():
    """
    Carga el modelo EfficientNet-B0 y aplica el state_dict guardado en CPU (.joblib).
    """
    num_classes = len(CLASS_NAMES)

    # MISMA arquitectura que en entrenamiento (EfficientNet-B0)
    model = create_model_enb0(num_classes)

    # Cargar state_dict ya en CPU
    state_dict = joblib.load("modelo_fibrosis_state_dict_cpu_enb0.joblib")
    model.load_state_dict(state_dict, strict=True)

    model.to(DEVICE)
    model.eval()

    return model


# Transformaciones iguales a las usadas en validación
TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # ecografía gris → 3 canales
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]),
])


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Recibe una imagen PIL y devuelve un tensor listo para el modelo.
    """
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    img = TRANSFORM(image)
    img = img.unsqueeze(0)  # batch size = 1
    return img.to(DEVICE)


def predict(image: Image.Image, model: torch.nn.Module):
    """
    Ejecuta la predicción sobre una imagen y devuelve:
    - índice de clase predicha
    - vector de probabilidades
    """
    tensor = preprocess_image(image)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    return pred_idx, probs


def main():
    st.set_page_config(page_title="Clasificador de fibrosis hepática", layout="centered")

    st.title("🩺 Clasificador de fibrosis hepática en ecografías")
    st.write(
        """
        Esta aplicación permite cargar una **imagen ecográfica hepática** y obtener una
        predicción del estadio de fibrosis (F0–F4) usando un modelo de **red neuronal convolucional (CNN)** basado en EfficientNet-B0.
        """
    )

    # Cargar modelo (solo la primera vez gracias a @st.cache_resource)
    with st.spinner("Cargando modelo..."):
        model = load_model()
    st.success("Modelo cargado correctamente.")

    uploaded_file = st.file_uploader(
        "Sube una ecografía hepática (formato JPG, PNG o JPEG):",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Imagen original")
            st.image(image, use_column_width=True)

        if st.button("Realizar predicción"):
            with st.spinner("Procesando imagen y ejecutando el modelo..."):
                pred_idx, probs = predict(image, model)
                pred_class = CLASS_NAMES[pred_idx]

            with col2:
                st.subheader("Resultado de la predicción")
                st.markdown(f"### Estadio predicho: **{pred_class}**")

                st.write("Probabilidades por clase:")
                prob_dict = {cls: float(p) for cls, p in zip(CLASS_NAMES, probs)}
                st.bar_chart(prob_dict)

            st.write("Detalle de probabilidades:")
            st.table(
                {
                    "Clase": CLASS_NAMES,
                    "Probabilidad": [f"{p * 100:.2f} %" for p in probs]
                }
            )


if __name__ == "__main__":
    main()
