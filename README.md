# MelodIA_Sistema_de_Reonhecimento_de_Aves_por_Sons

MelodIA – Identificação de Aves por Áudio (BirdNET + k‑NN)
Aplicação web de ciência de dados para identificação automática de espécies de aves a partir de gravações de áudio, combinando embeddings geradas pelo BirdNET com um classificador k‑vizinhos mais próximos (k‑NN) e servindo tudo via FastAPI.​

# Visão geral
O MelodIA recebe arquivos MP3/WAV contendo vocalizações de aves, extrai embeddings especializadas usando o modelo BirdNET (via birdnetlib) e utiliza um modelo k‑NN treinado em dados do Xeno‑Canto para prever as espécies mais prováveis, retornando o top‑5 com escores de confiança. O objetivo é oferecer uma ferramenta simples e interpretável para experimentos de bioacústica e monitoramento de biodiversidade.​

---

# Download arquivos no link abaixo:

[Clique aqui para acessar arquivos no Google Drive](https://drive.google.com/drive/folders/1zthTp8BoSVZsOonrc9ZSqdFkkdAtFTQr?usp=sharing)

# Aplicação Web:
[Clique aqui para acesso on-line da Aplicação](https://melod-ia.interativoti.app/)

---

# **Resumo do Projeto**

O **MelodIA** é um sistema de **classificação automática de aves por meio de gravações de áudio**.
Ele utiliza:

* **BirdNET** para extração de *embeddings* especializados em bioacústica
* **k-Nearest Neighbors (k-NN)** para classificação
* **FastAPI** para servir o modelo via interface Web
* **Xeno-Canto** como base de dados de referência

O projeto foi desenvolvido como parte da disciplina **Machine Learning Aplicado II**, sob orientação do **Prof. Juan Gabriel Colonna**.

---

#  **Arquitetura Geral**

```
             Arquivo de áudio (WAV/MP3)
                         |
                         v
                Extração de Embeddings
                    (BirdNET)
                         |
                         v
                Classificador KNN
                         |
                         v
             Top-5 espécies + probabilidades
                         |
                         v
                Interface Web (FastAPI)
```

---

#  **Objetivos**

### **Objetivo Geral**

Construir e avaliar um sistema de IA para identificação de aves através de áudio, combinando embeddings do BirdNET com um classificador KNN.

### **Objetivos Específicos**

* Extrair **embeddings bioacústicos** com BirdNET
* Treinar um classificador KNN para espécies de aves
* Desenvolver uma API web com FastAPI
* Criar uma interface simples para upload de áudio
* Avaliar métricas de desempenho e discutir limitações

---

#  **Estrutura do Repositório**

```
Projeto_MelodIA/
│
├── api/                     # Arquivos da aplicação FastAPI
│   ├── main.py
│   ├── utils.py
│   └── templates/
│
├── src/                     # Código de processamento e modelo
│   ├── extract_embeddings.py
│   ├── train_knn.py
│   └── predict.py
│
├── notebooks/               # Notebooks dos experimentos
│   ├── 01_exploracao_dataset.ipynb
│   ├── 02_extracao_embeddings.ipynb
│   └── 03_treinamento_knn.ipynb
│
├── models/                  # Modelos salvos
│   ├── knn_birdnet.pkl
│   ├── label_encoder.pkl
│   └── metadata.json
│
├── docs/                    # PDFs, artigos, imagens, relatório final
│   ├── MelodIA.pdf
│   └── Projeto_Final_ML2.pdf
│
├── data/                    # Metadados / links da base Xeno-Canto
│   └── README.md
│
├── requirements.txt         # Dependências
├── README.md                # Este arquivo
└── .gitignore
```

---

#  **Base de Dados – Xeno-Canto**

O projeto utiliza gravações do repositório **Xeno-Canto**:

* +700.000 gravações
* ~10.000 espécies
* Metadados completos (localização, datas, qualidade)

As gravações são **segmentadas** e enviadas ao **BirdNET**, que retorna vetores numéricos (embeddings 1024-d).

---

#  **Metodologia & Pipeline**

## 1. **Extração de Embeddings com BirdNET**

* Uso da biblioteca `birdnetlib`
* Segmentação automática do áudio
* Extração de vetores representando características bioacústicas
* Média dos embeddings de todas as janelas

## 2. **Treinamento do Classificador K-NN**

* Algoritmo *instance-based*, ideal para poucos dados por classe
* Distância Euclidiana
* `k = 5` selecionado via validação

## 3. **API com FastAPI**

* Upload de arquivos (MP3/WAV)
* Processamento no servidor
* Retorno em JSON com:

  * top-5 espécies
  * probabilidade
  * nome científico
  * link da espécie no XC

---

# **Como Executar Localmente**

## **Clone o repositório**

```bash
git clone https://github.com/Priscilaleylianne/Projeto_MelodIA_Sistema_de_Reonhecimento_de_Aves_por_Sons
cd Projeto_MelodIA_Sistema_de_Reonhecimento_de_Aves_por_Sons
```

## **Crie um ambiente virtual**

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

## **Instale dependências**

```bash
pip install -r requirements.txt
```

## **Execute a API**

```bash
uvicorn api.main:app --reload
```

Acesse no navegador:

```
http://127.0.0.1:8000
```

---

# **Uso da API**

### **Endpoint Principal:**

```
POST /upload_audio
```

### Exemplo com `curl`:

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@meu_audio.wav"
```

Retorno esperado:

```json
{
  "top_predictions": [
    {"species": "Turdus leucomelas", "confidence": 0.78},
    {"species": "Turdus rufiventris", "confidence": 0.12},
    ...
  ]
}
```

---

# **Resultados e Análises**

### **Abordagem clássica (MFCC + SVM)**

* Otimizada para *voz humana*
* Baixo desempenho em cantos de aves
* 20–50% de acurácia no Xeno-Canto

### **Abordagem final (BirdNET + KNN)**

* Embeddings aprendidos com milhões de gravações
* Muito menos dependente de grande volume de dados locais
* Desempenho **significativamente superior**

| Abordagem     | Tipo de Feature           | Desempenho | Observações                               |
| ------------- | ------------------------- | ---------- | ----------------------------------------- |
| MFCC + SVM    | MFCC genérico             | Baixo      | Sensível a ruído e classes desbalanceadas |
| BirdNET + KNN | Embeddings especializados | Alto       | Robustez e melhor generalização           |

---

# **Limitações**

* Dataset extremamente desbalanceado (Xeno-Canto)
* Dependência de qualidade da gravação
* Ruídos ambientais podem confundir o modelo
* BirdNET não cobre 100% das espécies amazônicas
* KNN exige boa organização das embeddings no espaço vetorial

---

# **Trabalhos Futuros**

* Incorporar CNNs finamente ajustadas (fine-tuning)
* Expansão com gravações locais próprias
* Redução de ruído em pré-processamento
* Modelo híbrido (BirdNET + SVM ou Random Forest)
* Dashboard com gráficos e espectrogramas
* API autenticada para produção

---

# **Licença**

Este projeto está licenciado sob a **MIT License**.

---
## Equipe do Projeto

- Alexandre Teixeira da Silva
- César Braz de Oliveira
- Ícaro Guimarães Canto
- Priscila Leylianne da Silva Gonçalves
