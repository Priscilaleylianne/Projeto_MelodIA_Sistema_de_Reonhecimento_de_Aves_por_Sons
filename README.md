# MelodIA_Sistema_de_Reonhecimento_de_Aves_por_Sons

MelodIA – Identificação de Aves por Áudio (BirdNET + k‑NN)
Aplicação web de ciência de dados para identificação automática de espécies de aves a partir de gravações de áudio, combinando embeddings geradas pelo BirdNET com um classificador k‑vizinhos mais próximos (k‑NN) e servindo tudo via FastAPI.​

# Visão geral
O MelodIA recebe arquivos MP3/WAV contendo vocalizações de aves, extrai embeddings especializadas usando o modelo BirdNET (via birdnetlib) e utiliza um modelo k‑NN treinado em dados do Xeno‑Canto para prever as espécies mais prováveis, retornando o top‑5 com escores de confiança. O objetivo é oferecer uma ferramenta simples e interpretável para experimentos de bioacústica e monitoramento de biodiversidade.​

---
Devido ao seu tamanho final superior 100 MB, ele não pôde ser hospedado diretamente neste repositório GitHub.

Você pode baixá-lo através do link abaixo:

[Clique aqui para acessar arquivos no Google Drive](https://drive.google.com/drive/folders/1zthTp8BoSVZsOonrc9ZSqdFkkdAtFTQr?usp=sharing)

# Aplicação 
[Clique aqui para acesso on-line da Aplicação](https://melod-ia.interativoti.app/)

---

# Arquitetura do projeto

## - Extração de embeddings:

birdnetlib.Analyzer + Recording para processar o áudio e gerar embeddings em alta dimensão.​

Função extract_embedding(...) agrega embeddings de vários segmentos em um único vetor médio por gravação.​

## - Classificação:

Modelo k‑NN treinado offline sobre embeddings rotuladas (espécie) e serializado em model/knn_birdnet.pkl.​

LabelEncoder salvo em model/label_encoder.pkl para mapear índices de classes para nomes de espécies.​

Função predict_topk(...) retorna rótulos e probabilidades das top‑k espécies (padrão k=5).​

## - API Web (FastAPI):

Rota GET / renderiza a interface HTML (template/app.html).​

Rota POST /predict recebe o arquivo de áudio, salva em static/uploads, extrai embedding, chama o k‑NN e devolve um JSON com resultado da predição.

---

## Pré‑requisitos
    
    Python 3.9+

- Instalação (exemplo):

    pip install fastapi uvicorn numpy joblib birdnetlib jinja2


## Como treinar o k‑NN

O treinamento do k‑NN é feito em um script separado, usando embeddings extraídas do BirdNET a partir das gravações do Xeno‑Canto. Fluxo típico:​

1 - Carregar metadados e áudios do Xeno‑Canto (xenocanto_meta.csv + arquivos MP3/WAV).​

2 - Para cada gravação, usar o Analyzer/Recording para extrair embeddings e agregá‑las (média por gravação), armazenando o vetor e o rótulo da espécie.​

3 - Separar treino/teste, treinar o k‑NN (por exemplo, KNeighborsClassifier com métrica euclidiana) e um LabelEncoder para converter nomes de espécies em índices.​

4 -Salvar os objetos treinados em model/knn_birdnet.pkl e model/label_encoder.pkl com joblib.dump.​

## Como executar a API

## 1- Certifique‑se de que:

knn_birdnet.pkl e label_encoder.pkl estão em model/.
app.html está em template/.
As pastas static/ e static/uploads/ existem (o código cria uploads se faltar).​

## 2- Rodar o servidor local:

uvicorn app:app --reload

## 3 - Acessar no navegador:

– interface MelodIA para upload de áudios.

Endpoint de predição (para testes via API):
curl -X POST "http://localhost:8000/predict" \
  -F "file=@exemplo.wav"

Resposta esperada (JSON):

{
  "ok": true,
  "species_main": "Nome_da_Especie",
  "confidence_main": 0.87,
  "labels": ["Esp1", "Esp2", "Esp3", "Esp4", "Esp5"],
  "scores": [0.87, 0.06, 0.03, 0.02, 0.02],1
  "audio_url": "/static/uploads/audio_upload.wav"
}


## Equipe

- Alexandre Teixeira da Silva
- César Braz de Oliveira
- Ícaro Guimarães Canto
- Priscila Leylianne da Silva Gonçalves
