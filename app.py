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


@st.cache_resource
def load_model():
    """
    Carga el modelo ResNet18 y aplica el state_dict guardado en CPU (.joblib).
    """
    num_classes = len(CLASS_NAMES)

    # Misma arquitectura que usaste al entrenar
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, num_classes),
    )

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
    # Asegurarse de que viene en PIL.Image
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    # Aplicar transformaciones
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
        predicción del estadio de fibrosis (F0–F4) usando un modelo de **red neuronal convolucional (CNN)**.
        """
    )

    # Cargar modelo (solo la primera vez gracias a @st.cache_resource)
    with st.spinner("Cargando modelo..."):
        model = load_model()
    st.success("Modelo cargado correctamente.")

    # Carga de imagen
    uploaded_file = st.file_uploader(
        "Sube una ecografía hepática (formato JPG, PNG o JPEG):",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Leer imagen con PIL
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Imagen original")
            st.image(image, use_column_width=True)

        # Botón para predecir
        if st.button("Realizar predicción"):
            with st.spinner("Procesando imagen y ejecutando el modelo..."):
                pred_idx, probs = predict(image, model)
                pred_class = CLASS_NAMES[pred_idx]

            with col2:
                st.subheader("Resultado de la predicción")
                st.markdown(f"### Estadio predicho: **{pred_class}**")

                # Mostrar probabilidades por clase
                st.write("Probabilidades por clase:")
                prob_dict = {cls: float(p) for cls, p in zip(CLASS_NAMES, probs)}
                st.bar_chart(prob_dict)

            # Mostrar tabla de probabilidades
            st.write("Detalle de probabilidades:")
            st.table(
                {
                    "Clase": CLASS_NAMES,
                    "Probabilidad": [f"{p * 100:.2f} %" for p in probs]
                }
            )


if __name__ == "__main__":
    main()
