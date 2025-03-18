import torch
import torchvision.transforms as transforms
import torchvision.models as models
import gradio as gr
from flask import Flask, request, jsonify
from PIL import Image
from datasets import load_dataset
import threading

# Charger le dataset pour récupérer les noms des classes
dataset = load_dataset("PedroSampaio/fruits-360")
class_names = dataset["train"].features["label"].names  # Liste des noms de fruits

# Charger le modèle pré-entraîné avec la dernière couche modifiée
num_classes = len(class_names)
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Charger les poids du modèle entraîné avec `strict=False` pour éviter les erreurs de clés
try:
    model.load_state_dict(torch.load("model_fruits360.pth", map_location=torch.device("cpu")), strict=False)
    print("Modèle chargé avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")

model.eval()

# Transformation d'image
def transform_image(image: Image):
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Ajouter une dimension batch

# Fonction de prédiction
def predict(image: Image):
    image = transform_image(image)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return class_names[predicted.item()]

# Initialiser Flask
app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"message": "Bienvenue sur l'API Flask de classification des fruits"})

@app.route("/predict", methods=["POST"])
def predict_api():
    if "file" not in request.files:
        return jsonify({"error": "Aucune image envoyée"}), 400
    
    file = request.files["file"]
    try:
        image = Image.open(file).convert("RGB")
        prediction = predict(image)
        return jsonify({"fruit": prediction})
    except Exception as e:
        return jsonify({"error": f"Erreur lors de la prédiction : {e}"}), 500

# Interface Gradio
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(),
    title="Classification des Fruits",
    description="Téléversez une image de fruit pour obtenir sa classification."
)

def launch_gradio():
    iface.launch(share=True)

# Lancer Gradio dans un thread séparé
thread = threading.Thread(target=launch_gradio)
thread.start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
