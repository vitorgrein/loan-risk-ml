# Credit Risk Scoring API

Projeto de score de inadimplência e priorização de cobrança, utilizando dados públicos do desafio **Home Credit Default Risk (Kaggle)**.  
O objetivo é construir um pipeline completo de dados, desde ingestão e feature engineering até modelagem e disponibilização do score via API, seguindo práticas utilizadas em ambientes reais de crédito e cobrança.

---

## 🎯 Objetivo do Projeto

Desenvolver uma solução analítica robusta para:

- Prever risco de inadimplência  
- Gerar um score de crédito/cobrança  
- Priorizar clientes em estratégias de cobrança  
- Simular um fluxo de decisão próximo ao contexto bancário  

O foco do projeto não é apenas o modelo final, mas principalmente:

- Qualidade e consistência das features  
- Organização e versionamento do pipeline  
- Reprodutibilidade  
- Estrutura preparada para produção  

---

## 🧠 Abordagem

O projeto segue uma abordagem modular:

1. Ingestão e tratamento dos dados  
2. Feature engineering por fonte de dados  
3. Construção do dataset analítico  
4. Treinamento e avaliação do modelo  
5. Exposição do score via API (FastAPI)  

O modelo retorna:

- Probabilidade de inadimplência  
- Score normalizado  
- Classificação de risco  
- Prioridade de cobrança  

---

## 🧱 Estrutura do Projeto

credit-risk-scoring-api/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│ ├── raw/ # Dados originais (não versionados)
│ ├── processed/ # Dados tratados e intermediários
│ └── features/ # Features finais utilizadas no modelo
│
├── models/
│ └── credit_model.pkl
│
├── pipeline/
│ ├── build_features.py
│ ├── build_dataset.py
│ └── train_model.py
│
├── scoring/
│ ├── scorer.py # Lógica de score e classificação de risco
│ └── init.py
│
├── service/
│ ├── main.py # API FastAPI
│ └── init.py
│
└── evaluation/
└── model_analysis.ipynb

---

## 🚀 API de Scoring

A API foi construída com **FastAPI** e expõe endpoints para geração de score de crédito e priorização de cobrança.

### Endpoints

- `GET /health`  
- `GET /v1/features`  
- `POST /v1/score`  

---

## 📥 Exemplo de Request

```json
{
  "loan_amnt": 10000,
  "term": "36 months",
  "int_rate": 13.5,
  "emp_length": "10+ years",
  "home_ownership": "RENT",
  "annual_inc": 60000,
  "verification_status": "Verified",
  "purpose": "credit_card",
  "dti": 18.5,
  "delinq_2yrs": 0,
  "revol_bal": 8000,
  "revol_util": 45.2,
  "total_acc": 20
}

## 🛠️ Tecnologias Utilizadas

- Python 3.12  
- Pandas e NumPy para manipulação e análise de dados  
- Scikit-learn para modelagem preditiva  
- FastAPI para construção da API de scoring  
- Pydantic para validação e tipagem de dados  
- Uvicorn como servidor ASGI  
- Joblib para serialização e versionamento do modelo  
- Kaggle – Home Credit Default Risk (fonte de dados)

---

## 📌 Observações

- Os dados brutos não são versionados no repositório  
- A estrutura do projeto foi pensada para facilitar deploy em ambientes cloud (ex.: Railway, Fly.io)  
- A API simula um cenário real de consumo por aplicações front-end e sistemas de cobrança  
- O foco do projeto está na organização do pipeline, qualidade das features e reprodutibilidade