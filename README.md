### Classification des Fruits avec ResNet50

## Description

Cette application utilise un modèle de deep learning basé sur ResNet50 pour classifier les images de fruits. Elle propose une API Flask pour la prédiction ainsi qu'une interface interactive avec Gradio.

## Fonctionnalités

Classification d'images de fruits : Utilisation d'un modèle pré-entraîné adapté au dataset Fruits-360.

API Flask : Permet d'envoyer une image et d'obtenir une prédiction via une requête POST.

Interface Gradio : Permet une interaction utilisateur conviviale via un navigateur.

Multithreading : Exécution simultanée de Flask et Gradio sans blocage.

## Prérequis
pip install torch torchvision gradio flask datasets pillow
python app.py

## ""Accéder à l'API Flask""

L'API est disponible à l'adresse suivante : http://localhost:5000/

Endpoint principal : GET / - Retourne un message de bienvenue.

Endpoint de prédiction : POST /predict

Envoie une image via le champ file.

Retourne un JSON contenant le fruit prédit.

## Interface Gradio

Gradio permet de téléverser une image et de voir la prédiction en temps réel. Un lien sera généré lors de l'exécution de l'application.

## Structure du projet

app.py        # Code principal de l'application
model_fruits360.pth # Poids du modèle entraîné
requirements.txt   # Liste des dépendances
README.md          # Documentation# computer-vision-examen
