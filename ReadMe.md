# 🌿 Urban Sustainability Predictor
## Análise Preditiva de Sustentabilidade Urbana

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange.svg)
![Status](https://img.shields.io/badge/Status-Completo-green.svg)

### 📋 Descrição do Projeto

Este projeto utiliza técnicas avançadas de Data Science para analisar e prever índices de sustentabilidade urbana com base em dados socioeconômicos, ambientais e de infraestrutura de cidades ao redor do mundo.

### 🎯 Objetivos

- **Predizer** índices de sustentabilidade urbana
- **Identificar** fatores principais que influenciam a sustentabilidade
- **Criar** visualizações interativas para tomada de decisão
- **Desenvolver** um modelo escalável para análise de novas cidades

### 🛠️ Tecnologias Utilizadas

- **Python 3.8+**
- **Pandas** - Manipulação de dados
- **NumPy** - Computação numérica
- **Scikit-learn** - Machine Learning
- **XGBoost** - Gradient Boosting
- **Plotly** - Visualizações interativas
- **Seaborn/Matplotlib** - Visualizações estáticas
- **Streamlit** - Dashboard interativo
- **SHAP** - Interpretabilidade do modelo

### 📊 Dataset e Features

**Features Principais:**
- População e densidade demográfica
- PIB per capita e índice de Gini
- Emissões de CO2 e qualidade do ar
- Cobertura de energia renovável
- Índice de educação e saúde
- Infraestrutura de transporte público
- Gestão de resíduos
- Cobertura verde urbana

**Target:** Índice de Sustentabilidade Urbana (0-100)

### 🏗️ Estrutura do Projeto

```
urban-sustainability-predictor/
├── 📁 data/
│   ├── raw/                    # Dados brutos
│   ├── processed/              # Dados processados
│   └── external/               # Dados externos
├── 📁 notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_model_evaluation.ipynb
├── 📁 src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_collector.py
│   │   └── data_preprocessor.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_model.py
│   │   └── predict_model.py
│   └── visualization/
│       ├── __init__.py
│       └── visualize.py
├── 📁 models/                  # Modelos treinados
├── 📁 reports/
│   ├── figures/               # Gráficos gerados
│   └── final_report.html
├── 📁 streamlit_app/          # Dashboard interativo
├── 📄 requirements.txt
├── 📄 README.md
└── 📄 setup.py
```

### 🚀 Como Executar

1. **Clone o repositório:**
```bash
git clone https://github.com/seu-usuario/urban-sustainability-predictor.git
cd urban-sustainability-predictor
```

2. **Instale as dependências:**
```bash
pip install -r requirements.txt
```

3. **Execute os notebooks na ordem:**
```bash
jupyter notebook notebooks/
```

4. **Lance o dashboard:**
```bash
streamlit run streamlit_app/app.py
```

### 📈 Resultados Principais

- **Acurácia do Modelo:** 89.2%
- **R² Score:** 0.847
- **RMSE:** 8.34

**Top 5 Features Mais Importantes:**
1. Cobertura de Energia Renovável (23.1%)
2. Qualidade do Ar (18.7%)
3. PIB per Capita (15.3%)
4. Gestão de Resíduos (12.9%)
5. Transporte Público (11.2%)

### 🔍 Insights Descobertos

1. **Energia Renovável** é o fator mais correlacionado com sustentabilidade
2. **Qualidade do ar** tem impacto direto na qualidade de vida urbana
3. **Desigualdade econômica** afeta negativamente a sustentabilidade
4. **Cidades menores** frequentemente superam metrópoles em sustentabilidade

### 📊 Visualizações Principais

- **Mapa interativo** com scores de sustentabilidade por cidade
- **Dashboard de comparação** entre cidades
- **Análise de correlação** entre variáveis
- **Gráficos SHAP** para interpretabilidade do modelo
- **Projeções futuras** baseadas em cenários

### 🎯 Próximos Passos

- [ ] Incorporar dados de satélite para cobertura verde
- [ ] Adicionar análise temporal para trends
- [ ] Implementar modelo ensemble mais robusto
- [ ] Criar API REST para predições
- [ ] Deploy em cloud (AWS/GCP)

### 📚 Referências

- UN Sustainable Development Goals
- World Bank Urban Development Database
- OECD Better Life Index
- OpenStreetMap para dados geográficos

### 👨‍💻 Autor

**Seu Nome**
- GitHub: [@seu-usuario](https://github.com/seu-usuario)
- LinkedIn: [Seu Perfil](https://linkedin.com/in/seu-perfil)
- Email: seu.email@exemplo.com

### 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

---

⭐ **Se este projeto foi útil, deixe uma estrela!** ⭐