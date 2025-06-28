from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import torch
from io import BytesIO
import json
import numpy as np
import requests
from pydantic import BaseModel
import os

# Firebase Admin
import firebase_admin
from firebase_admin import credentials, firestore

# Inicializa Firebase
cred = credentials.Certificate("firebase-adminsdk.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://almanaque-d6ba0.web.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=False, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

class ImagemURL(BaseModel):
    imagemUrl: str

@app.post("/gerar-embedding")
async def gerar_embedding(data: ImagemURL):
    try:
        response = requests.get(data.imagemUrl)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Erro ao baixar a imagem")

        img = Image.open(BytesIO(response.content)).convert("RGB")
        face = mtcnn(img)

        if face is None:
            raise HTTPException(status_code=422, detail="Nenhum rosto detectado")

        embedding = model(face.unsqueeze(0).to(device)).detach().cpu().numpy()[0]
        return {"embedding": embedding.tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@app.post("/salvar-embedding")
async def salvar_embedding(request: Request):
    try:
        dados = await request.json()
        embedding = dados.get("embedding")
        url = dados.get("url")

        if not embedding or not url:
            return {"success": False, "message": "Faltando embedding ou url"}

        if not isinstance(embedding, list) or len(embedding) != 512:
            return {"success": False, "message": "Embedding inválido"}

        print("📥 Novo embedding recebido:", url)

        db.collection("embeddings").add({
            "url": url,
            "embedding": embedding
        })

        return {"success": True}
    except Exception as e:
        return {"success": False, "message": str(e)}

@app.post("/reconhecer")
async def reconhecer(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        img = Image.open(BytesIO(img_bytes)).convert("RGB")

        face = mtcnn(img)
        if face is None:
            return {"success": False, "message": "Nenhum rosto detectado"}

        face = face.unsqueeze(0).to(device)
        input_embedding = model(face).detach().cpu().numpy()[0]

        def cosine_similarity(v1, v2):
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

        resultados = []

        docs = db.collection("embeddings").stream()
        for doc in docs:
            data = doc.to_dict()
            vetor = data.get("embedding", [])
            url = data.get("url")

            print("🔎 Verificando embedding:", url)

            if isinstance(vetor, list) and len(vetor) == 512 and url:
                sim = cosine_similarity(input_embedding, vetor)
                resultados.append({"url": url, "similaridade": sim})

        resultados.sort(key=lambda x: x["similaridade"], reverse=True)
        return {"success": True, "resultados": resultados[:5]}
    except Exception as e:
        return {"success": False, "message": str(e)}

# Endpoint opcional de debug
@app.get("/debug/total-embeddings")
async def contar_embeddings():
    docs = db.collection("embeddings").stream()
    total = sum(1 for _ in docs)
    return {"total": total}
