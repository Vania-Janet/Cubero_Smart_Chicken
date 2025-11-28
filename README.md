# SmartPantry: Intelligent Gastronomic Analysis Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)

> **A Hybrid Framework for Recipe Recommendation and Price Forecasting**  
> Combining RAG (Retrieval-Augmented Generation), Computer Vision, and Time-Series Forecasting for data-driven culinary operations.

---

## 📋 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Modules](#modules)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Research Context](#research-context)
- [Contributing Team](#contributing-team)
- [License](#license)

---

## 🎯 Overview

**SmartPantry** is an intelligent platform designed to optimize culinary operations through advanced machine learning techniques. The system addresses two critical challenges in the food industry:

1. **Operational Efficiency**: Recipe recommendation based on available inventory using semantic search and RAG
2. **Financial Intelligence**: Real-time ingredient pricing and cost forecasting using time-series models

### Problem Statement

Traditional recipe recommendation systems lack:
- **Semantic understanding** of ingredient relationships (e.g., "poultry" ≈ "chicken")
- **Economic integration** with real-time market prices
- **Dual optimization** for both cost-effectiveness and customer satisfaction

### Solution

A hybrid AI platform that combines:
- **Voyage AI embeddings** (1024-dimensional) for semantic recipe retrieval
- **FAISS vector database** with K-means clustering for sub-second search
- **Cohere rerank-v3.5** for result refinement
- **Google Gemini** for profit-focused analysis in Spanish
- **PROFECO API** integration for real-time pricing across 34 Mexican cities
- **Facebook Prophet** for ingredient price forecasting to 2026

---

## 🏗️ System Architecture

```
SmartPantry/
├── RAG_System/           # Recipe Retrieval & Pricing Module
│   ├── app.py            # FastAPI service (deployed on Render)
│   ├── FAISS_recetas/    # Vector index (45 recipes, 1024-dim embeddings)
│   ├── requirements.txt  # Dependencies (Voyage, Cohere, Gemini, FAISS)
│   └── Vision/           # Computer Vision Module
│       ├── src/          # ML pipelines (CLIP embeddings, XGBoost)
│       ├── models/       # Trained models
│       └── data/         # Processed datasets & embeddings
└── Scraping_Precios_Actuales.ipynb  # PROFECO data collection
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Embeddings** | Voyage AI (voyage-3-large) | Semantic recipe representation |
| **Vector DB** | FAISS (K-means clustering) | Fast similarity search |
| **Reranking** | Cohere rerank-v3.5 | Result refinement |
| **LLM** | Google Gemini 2.5 Pro | Profit analysis generation |
| **Vision** | CLIP + XGBoost | Image-based recipe retrieval |
| **Forecasting** | Facebook Prophet | Price prediction (2026) |
| **Pricing API** | PROFECO QQP | Real-time prices (34 cities) |
| **Web Framework** | FastAPI | Production API |
| **Deployment** | Render.com | Cloud hosting |

---

## 📦 Modules

### 1. RAG_System (Recipe Recommendation & Pricing)

**Core Functionality:**
- Semantic search over 45 recipes with Voyage AI embeddings
- Dual scoring system:
  - `score_popularidad`: Customer satisfaction (health, prep time, tags)
  - `score_ahorro`: Cost-effectiveness (price, ingredient match)
- Real-time PROFECO pricing integration
- Gemini-generated profit analysis in Spanish

**Deployed API:** [`https://food-api-vgt8.onrender.com`](https://food-api-vgt8.onrender.com/docs)

**Key Endpoints:**
- `POST /recommend` - Returns recipe cards with pricing, scores, and profit analysis

**Example Response:**
```json
{
  "ciudad": "Ciudad de México",
  "tarjetas": [
    {
      "nombre_receta": "Pollo Taco Sopa",
      "score_popularidad": 57.9,
      "score_ahorro": 56.67,
      "etiqueta": "❤️ Más Popular",
      "costo_total": 120.68,
      "precio_venta_sugerido": 75.43,
      "ganancia_por_porcion": 45.26,
      "analisis": "Rentabilidad alta con margen del 60%..."
    }
  ]
}
```

### 2. Vision Module (Computer Vision)

**Core Functionality:**
- CLIP embeddings for image-based recipe search
- XGBoost ingredient scoring model
- Multimodal fusion (text + image)
- Streamlit interface for visual queries

**Key Components:**
- `src/vision/inference.py` - CLIP model inference
- `src/recommender/hybrid.py` - Multimodal recommendation engine
- `models/ingredient_scoring/` - Trained XGBoost model

---

## ✨ Key Features

### 1. Semantic Ingredient Understanding
```python
# Automatically recognizes synonyms and related concepts
Query: "protein, carbs, tomato"
→ Matches: "chicken", "rice", "jitomate" (Spanish tomato)
```

### 2. Dual Optimization Strategy
- **💰 Mejor Precio**: Lowest cost per serving
- **❤️ Más Popular**: Highest customer satisfaction potential
- **⚖️ Equilibrada**: Balanced cost-quality tradeoff

### 3. Real-Time Pricing (34 Cities)
Integration with PROFECO's Quién es Quién en los Precios API:
- Ciudad de México, Guadalajara, Monterrey, Puebla, Tijuana, etc.
- Tracks prices across Walmart, Chedraui, Soriana, Bodega Aurrera

### 4. Price Forecasting (2026)
Facebook Prophet models trained on:
- **INEGI** (2011-2018): Historical baseline
- **SEDECO** (PDF extraction): Mexico City focus
- **SNIIM** (Web scraping 2019-2025): Recent trends

**RMSE Performance (2024 validation):**
| Ingredient | RMSE (MXN) |
|------------|------------|
| Rice       | 3.41       |
| Zucchini   | 4.63       |
| Sugar      | 13.03      |
| Avocado    | 33.70      |

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- Git LFS (for large model files)
- API Keys:
  - Voyage AI
  - Cohere
  - Google Gemini

### Setup

```bash
# Clone repository
git clone https://github.com/Vania-Janet/Cubero_Smart_Chicken.git
cd Cubero_Smart_Chicken/RAG_System

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your API keys:
# VOYAGE_API_KEY=your_key
# COHERE_API_KEY=your_key
# GEMINI_API_KEY=your_key
```

---

## 💻 Usage

### Running the API Locally

```bash
cd RAG_System
uvicorn app:app --host 0.0.0.0 --port 8000
```

Visit:
- **Swagger UI**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/

### Example API Call

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "ingredients": ["chicken", "rice", "tomato"],
    "city": "Ciudad de México",
    "max_results": 3
  }'
```

### Using the Vision Module

```bash
cd RAG_System/Vision

# Install vision dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run src/app/streamlit_app.py
```

---

## 📚 API Documentation

### `POST /recommend`

**Request Body:**
```json
{
  "ingredients": ["string"],
  "city": "Ciudad de México",
  "max_results": 3
}
```

**Response Schema:**
- `ciudad`: Selected city
- `ingredientes`: Input ingredients
- `tarjetas`: Array of recipe cards
  - `nombre_receta`: Recipe name (Spanish)
  - `score_popularidad`: Customer satisfaction score (0-100)
  - `score_ahorro`: Cost-effectiveness score (0-100)
  - `etiqueta`: Badge (💰/❤️/⚖️)
  - `ingredientes_con_precio`: Priced ingredients with store info
  - `costo_total`: Total cost (MXN)
  - `precio_venta_sugerido`: Suggested sale price (2.5x markup)
  - `ganancia_por_porcion`: Profit per serving
  - `analisis`: LLM-generated profit analysis
- `resumen`: Summary text

**Available Cities:** Ciudad de México, Guadalajara, Monterrey, Puebla, Tijuana, León, Zapopan, etc. (34 total)

Full documentation: [`/docs`](https://food-api-vgt8.onrender.com/docs)

---

## 🔬 Research Context

This project is part of academic research on **Intelligent Gastronomic Analysis** conducted at IIMAS, UNAM. The work explores:

1. **Semantic Search for Culinary Applications**
   - High-dimensional embeddings (1024-dim Voyage AI)
   - K-means clustering in FAISS for scalable retrieval
   - Evaluation of reranking strategies (Cohere vs. alternatives)

2. **Economic Integration in Recommender Systems**
   - Real-time pricing API integration
   - Dual optimization (cost + satisfaction)
   - LLM-driven financial analysis

3. **Time-Series Forecasting for Agriculture**
   - Seasonal pattern modeling with Prophet
   - Handling missing data and outliers
   - Exogenous shock resilience

### Publications & Presentations

**LaTeX Paper:** Available in repository (`latex.tex`)
- Format: NeurIPS 2022 style
- Title: *Intelligent Gastronomic Analysis: A Hybrid Framework for Recipe Recommendation and Price Forecasting*

### Limitations & Future Work

**Current Limitations:**
- Recipe dataset limited to 45 entries (Spoonacular subset)
- Prophet models sensitive to exogenous shocks (weather, geopolitics)
- Computer vision module requires pre-computed CLIP embeddings

**Planned Enhancements:**
- Real-time inventory tracking integration
- Multi-language support (English, Spanish)
- External regressors for Prophet (climate data, import tariffs)
- Expanded recipe corpus (1000+ recipes)
- Mobile application deployment

---

## 👥 Contributing Team

| Name | Role | Email |
|------|------|-------|
| **Alegre Ventura Roberto Jhoshua** | Data Science & Math | 319257836@ciencias.unam.mx |
| **Arano Bejarano Melisa Asharet** | Data Science & Math | asharer.b@gmail.com |
| **Fonseca González Bruno** | Data Science & Physics | fonsecagonzalezbruno@gmail.com |
| **Pérez Cruz David Leopardo** | Computer Engineering | davidpcleo@gmail.com |
| **Ramirez Nava Alejandro Iram** | Computer Science & Physics | alejandroiram@ciencias.unam.mx |
| **Raya Rios Vania Janet** | Data Science | vaniaraya17@ciencias.unam.mx |

**Institution:** Instituto de Investigaciones en Matemáticas Aplicadas y en Sistemas (IIMAS), UNAM

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- **Open Source Community**: LangChain, FAISS, Prophet maintainers
- **API Providers**: Voyage AI, Cohere, Google Gemini
- **Data Sources**: INEGI, SNIIM, SEDECO, PROFECO
- **Academic Guidance**: IIMAS faculty and research advisors

---

## 📞 Contact

For questions, collaborations, or issues:
- **GitHub Issues**: [Create an issue](https://github.com/Vania-Janet/Cubero_Smart_Chicken/issues)
- **Email**: vaniaraya17@ciencias.unam.mx
- **Project Website**: https://food-api-vgt8.onrender.com/docs

---

<div align="center">

**Built with ❤️ at UNAM | Powered by AI for culinary excellence**

</div>
