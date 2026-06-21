import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2

from gradcam import GradCAM
from model import load_model

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------
# CSS
# -----------------------------
st.markdown("""
<style>
.prediction-box {
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #444;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.title("🧠 Brain Tumor AI")

    st.markdown("---")

    st.markdown("""
### Model Information

- EfficientNet-B0
- Transfer Learning
- PyTorch
- Streamlit
- Grad-CAM

### Classes

- Glioma
- Meningioma
- No Tumor
- Pituitary
""")

    st.markdown("---")

    st.info(
        "Upload a Brain MRI image to get prediction and Grad-CAM visualization."
    )

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# -----------------------------
# MODEL
# -----------------------------
@st.cache_resource
def load_brain_model():

    model = load_model()

    model.load_state_dict(
        torch.load(
            "models/best_brain_tumor_model.pth",
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    return model


model = load_brain_model()

# -----------------------------
# GRADCAM
# -----------------------------
target_layer = model.features[-1]
cam = GradCAM(model, target_layer)

# -----------------------------
# CLASSES
# -----------------------------
classes = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

# -----------------------------
# TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# -----------------------------
# HEADER
# -----------------------------
st.title("🧠 Brain Tumor Classification System")

st.write(
    "Upload an MRI image and get prediction with Grad-CAM explainability."
)

st.markdown("---")

# -----------------------------
# FILE UPLOADER
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# PREDICTION
# -----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded MRI")
        st.image(
            image,
            use_container_width=True
        )

    with col2:

        st.subheader("Prediction")

        if st.button("Predict"):

            image_tensor = (
                transform(image)
                .unsqueeze(0)
                .to(device)
            )

            with torch.no_grad():

                output = model(image_tensor)

                probabilities = torch.softmax(
                    output,
                    dim=1
                )

                confidence, pred = torch.max(
                    probabilities,
                    dim=1
                )

            predicted_class = classes[pred.item()]
            confidence = confidence.item() * 100

            st.markdown(
                f"""
                <div class="prediction-box">
                    <h2>{predicted_class.upper()}</h2>
                    <h3>Confidence: {confidence:.2f}%</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(confidence / 100)

            st.markdown("### Class Probabilities")

            for idx, cls in enumerate(classes):

                prob = probabilities[0][idx].item() * 100

                st.write(
                    f"{cls}: {prob:.2f}%"
                )

            # -----------------------------
            # GRAD-CAM
            # -----------------------------
            heatmap = cam.generate(
                image_tensor,
                pred.item()
            )

            img_np = np.array(
                image.resize((224, 224))
            )

            heatmap = cv2.applyColorMap(
                np.uint8(255 * heatmap),
                cv2.COLORMAP_JET
            )

            overlay = cv2.addWeighted(
                img_np,
                0.6,
                heatmap,
                0.4,
                0
            )

            st.markdown("---")

            st.subheader(
                "🔥 Grad-CAM Visualization"
            )

            st.image(
                overlay,
                use_container_width=True
            )

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")

st.markdown(
    "Built with ❤️ using PyTorch, Streamlit, EfficientNet-B0 and Grad-CAM"
)



