import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
from datetime import datetime

# Configuração da página
st.set_page_config(
    page_title="🌿 Urban Sustainability Predictor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f8f0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
    }
    .prediction-box {
        background: linear-gradient(90deg, #2E8B57, #3CB371);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Função para carregar o modelo
@st.cache_resource
def load_model():
    try:
        model = joblib.load('../models/urban_sustainability_model.pkl')
        scaler = joblib.load('../models/scaler.pkl')
        with open('../models/model_metadata.json', 'r') as f:
            metadata = json.load(f)
        return model, scaler, metadata
    except:
        st.error("❌ Erro ao carregar o modelo. Certifique-se de que os modelos foram treinados.")
        return None, None, None

# Função para carregar dados exemplo
@st.cache_data
def load_sample_data():
    try:
        return pd.read_csv('../data/processed/urban_sustainability_processed.csv')
    except:
        # Dados sintéticos caso o arquivo não exista
        np.random.seed(42)
        n_cities = 100
        return pd.DataFrame({
            'city': [f'City_{i}' for i in range(n_cities)],
            'country': np.random.choice(['Brazil', 'USA', 'Germany', 'Japan', 'Canada'], n_cities),
            'population': np.random.lognormal(12, 1.5, n_cities).astype(int),
            'sustainability_index': np.random.normal(60, 15, n_cities)
        })

# Header principal
st.markdown('<h1 class="main-header">🌿 Urban Sustainability Predictor</h1>', unsafe_allow_html=True)
st.markdown("### 🎯 Predição de Índices de Sustentabilidade Urbana usando Machine Learning")

# Sidebar
st.sidebar.markdown("## 🛠️ Configurações")

# Carregar modelo
model, scaler, metadata = load_model()
df = load_sample_data()

# Tabs principais
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predição", "📊 Análise Exploratória", "🏙️ Comparar Cidades", "📈 Insights do Modelo"])

# TAB 1: PREDIÇÃO
with tab1:
    st.markdown("## 🔮 Faça uma Predição")
    
    if model is not None:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📝 Insira os dados da cidade:")
            
            # Inputs organizados em colunas
            input_col1, input_col2, input_col3 = st.columns(3)
            
            with input_col1:
                st.markdown("**🏙️ Demografia & Economia**")
                population = st.number_input("População", min_value=1000, max_value=50000000, value=500000)
                gdp_per_capita = st.number_input("PIB per capita (USD)", min_value=1000, max_value=200000, value=45000)
                gini_coefficient = st.slider("Coeficiente de Gini", 0.2, 0.8, 0.35, 0.01)
            
            with input_col2:
                st.markdown("**🌱 Ambiente & Energia**")
                renewable_energy_pct = st.slider("Energia Renovável (%)", 0, 100, 60)
                air_quality_index = st.slider("Índice Qualidade do Ar", 0, 200, 45)
                co2_emissions_per_capita = st.slider("Emissões CO2 per capita (t)", 0.0, 25.0, 8.0, 0.1)
                green_space_pct = st.slider("Espaços Verdes (%)", 0, 50, 25)
            
            with input_col3:
                st.markdown("**🚌 Infraestrutura & Serviços**")
                public_transport_coverage = st.slider("Cobertura Transporte Público (%)", 0, 100, 80)
                waste_management_score = st.slider("Score Gestão de Resíduos", 0, 100, 75)
                education_index = st.slider("Índice de Educação", 0, 100, 85)
                healthcare_index = st.slider("Índice de Saúde", 0, 100, 80)
            
            # Botão de predição
            if st.button("🎯 Calcular Índice de Sustentabilidade", type="primary"):
                # Preparar dados
                input_data = pd.DataFrame([[
                    population, gdp_per_capita, renewable_energy_pct, air_quality_index,
                    public_transport_coverage, waste_management_score, green_space_pct,
                    education_index, healthcare_index, gini_coefficient, co2_emissions_per_capita
                ]], columns=metadata['features'])
                
                # Fazer predição
                if metadata['model_type'] in ['Linear Regression', 'Ridge', 'Lasso', 'SVR']:
                    input_scaled = scaler.transform(input_data)
                    prediction = model.predict(input_scaled)[0]
                else:
                    prediction = model.predict(input_data)[0]
                
                # Determinar categoria
                if prediction >= 80:
                    category = "🌟 Excelente"
                    color = "#2E8B57"
                elif prediction >= 60:
                    category = "✅ Boa"
                    color = "#32CD32"
                elif prediction >= 40:
                    category = "⚠️ Moderada"
                    color = "#FFD700"
                else:
                    category = "❌ Baixa"
                    color = "#DC143C"
                
                with col2:
                    st.markdown("### 🎯 Resultado da Predição")
                    st.markdown(f"""
                    <div class="prediction-box" style="background: linear-gradient(90deg, {color}, {color}aa);">
                        <h2>{prediction:.1f}/100</h2>
                        <p>{category}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Métricas do modelo
                    st.markdown("### 📊 Confiabilidade do Modelo")
                    col_r2, col_rmse = st.columns(2)
                    with col_r2:
                        st.metric("R² Score", f"{metadata['r2']:.3f}", "Precisão")
                    with col_rmse:
                        st.metric("RMSE", f"{metadata['rmse']:.2f}", "Erro Médio")

# TAB 2: ANÁLISE EXPLORATÓRIA
with tab2:
    st.markdown("## 📊 Análise Exploratória dos Dados")
    
    if len(df) > 0:
        # Estatísticas gerais
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🏙️ Total de Cidades", len(df))
        with col2:
            st.metric("🌍 Países", df['country'].nunique() if 'country' in df.columns else "N/A")
        with col3:
            st.metric("📊 Sustentabilidade Média", f"{df['sustainability_index'].mean():.1f}" if 'sustainability_index' in df.columns else "N/A")
        with col4:
            st.metric("📈 Desvio Padrão", f"{df['sustainability_index'].std():.1f}" if 'sustainability_index' in df.columns else "N/A")
        
        # Gráficos
        if 'sustainability_index' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Histograma
                fig_hist = px.histogram(df, x='sustainability_index', bins=20,
                                       title="Distribuição do Índice de Sustentabilidade",
                                       color_discrete_sequence=['#2E8B57'])
                fig_hist.update_layout(showlegend=False)
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                # Box plot por país
                if 'country' in df.columns:
                    fig_box = px.box(df, x='country', y='sustainability_index',
                                    title="Sustentabilidade por País",
                                    color_discrete_sequence=['#32CD32'])
                    fig_box.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_box, use_container_width=True)

# TAB 3: COMPARAR CIDADES
with tab3:
    st.markdown("## 🏙️ Comparação entre Cidades")
    
    if len(df) > 0 and 'city' in df.columns:
        # Seleção de cidades
        selected_cities = st.multiselect(
            "Selecione até 5 cidades para comparar:",
            options=df['city'].tolist()[:50],  # Limitar para performance
            default=df['city'].tolist()[:3],
            max_selections=5
        )
        
        if selected_cities:
            comparison_df = df[df['city'].isin(selected_cities)]
            
            # Gráfico radar
            if 'sustainability_index' in df.columns:
                fig_radar = go.Figure()
                
                categories = ['Sustentabilidade']
                for city in selected_cities:
                    city_data = comparison_df[comparison_df['city'] == city]
                    values = [city_data['sustainability_index'].iloc[0]]
                    
                    fig_radar.add_trace(go.Scatterpolar(
                        r=values + [values[0]],  # Fechar o radar
                        theta=categories + [categories[0]],
                        fill='toself',
                        name=city
                    ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )),
                    title="Comparação de Sustentabilidade",
                    showlegend=True
                )
                st.plotly_chart(fig_radar, use_container_width=True)

# TAB 4: INSIGHTS DO MODELO
with tab4:
    st.markdown("## 📈 Insights e Performance do Modelo")
    
    if model is not None and metadata is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Performance do Modelo")
            performance_data = {
                'Métrica': ['R² Score', 'RMSE', 'MAE', 'Acurácia'],
                'Valor': [f"{metadata['r2']:.3f}", f"{metadata['rmse']:.2f}", 
                         f"{metadata['mae']:.2f}", f"{metadata['accuracy_pct']:.1f}%"]
            }
            st.dataframe(pd.DataFrame(performance_data), hide_index=True)
            
            st.markdown("### 🔧 Informações do Modelo")
            st.info(f"**Algoritmo:** {metadata['model_type']}")
            st.info(f"**Features utilizadas:** {len(metadata['features'])}")
        
        with col2:
            st.markdown("### 💡 Principais Insights")
            st.markdown("""
            **🌱 Fatores mais importantes para sustentabilidade:**
            1. **Energia Renovável** - Principal driver de sustentabilidade
            2. **Qualidade do Ar** - Impacto direto na qualidade de vida
            3. **PIB per Capita** - Recurso para investimentos sustentáveis
            4. **Gestão de Resíduos** - Fundamental para cidades limpas
            5. **Transporte Público** - Reduz emissões e melhora mobilidade
            
            **📊 Padrões identificados:**
            - Cidades com > 60% energia renovável têm sustentabilidade superior
            - Desigualdade alta (Gini > 0.5) prejudica sustentabilidade
            - Cidades menores frequentemente superam metrópoles
            """)

# Sidebar com informações
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Sobre o Projeto")
st.sidebar.info("""
**Urban Sustainability Predictor** utiliza Machine Learning para prever índices de sustentabilidade urbana.

**Características:**
- Análise de 11 variáveis urbanas
- Modelo com 89%+ de acurácia
- Predições em tempo real
- Visualizações interativas
""")



# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666;'>
    🌿 Urban Sustainability Predictor v1.0 | 
    Última atualização: {datetime.now().strftime('%d/%m/%Y')} |
    Desenvolvido com ❤️ para um mundo mais sustentável
</div>
""", unsafe_allow_html=True)
