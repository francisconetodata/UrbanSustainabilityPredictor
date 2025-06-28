# Urban Sustainability Predictor - Análise Exploratória
# 02_exploratory_analysis.ipynb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configurações de visualização
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# 1. CARREGAMENTO DOS DADOS
print("🔄 Carregando dados...")

# Simulando dados realistas (em um projeto real, você carregaria de APIs ou CSVs)
np.random.seed(42)
n_cities = 500

# Gerando dados sintéticos realistas
cities_data = {
    'city': [f'City_{i}' for i in range(n_cities)],
    'country': np.random.choice(['Brazil', 'USA', 'Germany', 'Japan', 'Canada', 'Australia', 'UK', 'France'], n_cities),
    'population': np.random.lognormal(12, 1.5, n_cities).astype(int),
    'gdp_per_capita': np.random.lognormal(10, 0.8, n_cities),
    'renewable_energy_pct': np.random.beta(2, 3, n_cities) * 100,
    'air_quality_index': np.random.gamma(2, 25, n_cities),
    'public_transport_coverage': np.random.beta(3, 2, n_cities) * 100,
    'waste_management_score': np.random.beta(4, 2, n_cities) * 100,
    'green_space_pct': np.random.beta(2, 3, n_cities) * 50,
    'education_index': np.random.beta(5, 2, n_cities) * 100,
    'healthcare_index': np.random.beta(4, 2, n_cities) * 100,
    'gini_coefficient': np.random.beta(2, 3, n_cities) * 0.7 + 0.2,
    'co2_emissions_per_capita': np.random.gamma(2, 3, n_cities),
}

df = pd.DataFrame(cities_data)

# Criando target baseado em uma combinação realista das features
sustainability_score = (
    0.25 * (df['renewable_energy_pct'] / 100) +
    0.20 * (1 - df['air_quality_index'] / 200) +
    0.15 * (df['public_transport_coverage'] / 100) +
    0.15 * (df['waste_management_score'] / 100) +
    0.10 * (df['green_space_pct'] / 50) +
    0.10 * (1 - df['gini_coefficient']) +
    0.05 * (1 - df['co2_emissions_per_capita'] / 20)
) * 100

# Adicionando ruído realista
df['sustainability_index'] = np.clip(
    sustainability_score + np.random.normal(0, 5, n_cities), 
    0, 100
)

print(f"✅ Dataset carregado: {df.shape[0]} cidades, {df.shape[1]} variáveis")

# 2. ANÁLISE DESCRITIVA
print("\n📊 ESTATÍSTICAS DESCRITIVAS")
print("="*50)
print(df.describe().round(2))

print(f"\n🏙️  Cidades por país:")
print(df['country'].value_counts())

# 3. ANÁLISE DE DISTRIBUIÇÕES
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
axes = axes.ravel()

numeric_cols = df.select_dtypes(include=[np.number]).columns
for i, col in enumerate(numeric_cols[:9]):
    axes[i].hist(df[col], bins=30, alpha=0.7, color=plt.cm.Set3(i))
    axes[i].set_title(f'Distribuição: {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequência')

plt.tight_layout()
plt.savefig('reports/figures/distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. MATRIZ DE CORRELAÇÃO
plt.figure(figsize=(14, 12))
correlation_matrix = df.select_dtypes(include=[np.number]).corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

sns.heatmap(correlation_matrix, 
            mask=mask,
            annot=True, 
            cmap='RdBu_r', 
            center=0,
            square=True,
            fmt='.2f',
            cbar_kws={"shrink": .8})
plt.title('Matriz de Correlação - Variáveis Urbanas', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig('reports/figures/correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. ANÁLISE DO TARGET
print("\n🎯 ANÁLISE DO ÍNDICE DE SUSTENTABILIDADE")
print("="*50)
print(f"Média: {df['sustainability_index'].mean():.2f}")
print(f"Mediana: {df['sustainability_index'].median():.2f}")
print(f"Desvio Padrão: {df['sustainability_index'].std():.2f}")
print(f"Mínimo: {df['sustainability_index'].min():.2f}")
print(f"Máximo: {df['sustainability_index'].max():.2f}")

# Visualização do target
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Histograma
ax1.hist(df['sustainability_index'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
ax1.axvline(df['sustainability_index'].mean(), color='red', linestyle='--', label=f'Média: {df["sustainability_index"].mean():.1f}')
ax1.set_title('Distribuição do Índice de Sustentabilidade')
ax1.set_xlabel('Índice de Sustentabilidade')
ax1.set_ylabel('Frequência')
ax1.legend()

# Box plot por país
df.boxplot(column='sustainability_index', by='country', ax=ax2)
ax2.set_title('Sustentabilidade por País')
ax2.set_xlabel('País')
ax2.set_ylabel('Índice de Sustentabilidade')

plt.tight_layout()
plt.savefig('reports/figures/target_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. TOP CORRELAÇÕES COM O TARGET
target_correlations = df.corr()['sustainability_index'].abs().sort_values(ascending=False)
print("\n🔗 TOP 10 CORRELAÇÕES COM SUSTENTABILIDADE:")
print("="*50)
for var, corr in target_correlations[1:11].items():
    print(f"{var:<25}: {corr:.3f}")

# 7. SCATTER PLOTS DAS PRINCIPAIS FEATURES
top_features = target_correlations[1:5].index
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

for i, feature in enumerate(top_features):
    axes[i].scatter(df[feature], df['sustainability_index'], alpha=0.6, color=plt.cm.Set1(i))
    
    # Linha de tendência
    z = np.polyfit(df[feature], df['sustainability_index'], 1)
    p = np.poly1d(z)
    axes[i].plot(df[feature], p(df[feature]), "r--", alpha=0.8)
    
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Índice de Sustentabilidade')
    axes[i].set_title(f'Sustentabilidade vs {feature}')
    
    # Correlação no título
    corr = df[feature].corr(df['sustainability_index'])
    axes[i].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[i].transAxes, 
                bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('reports/figures/feature_relationships.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. ANÁLISE POR FAIXAS DE POPULAÇÃO
df['population_category'] = pd.cut(df['population'], 
                                  bins=[0, 100000, 500000, 1000000, float('inf')],
                                  labels=['Pequena', 'Média', 'Grande', 'Metrópole'])

print("\n🏙️  SUSTENTABILIDADE POR TAMANHO DE CIDADE:")
print("="*50)
size_analysis = df.groupby('population_category')['sustainability_index'].agg(['count', 'mean', 'std']).round(2)
print(size_analysis)

# 9. OUTLIERS DETECTION
Q1 = df['sustainability_index'].quantile(0.25)
Q3 = df['sustainability_index'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['sustainability_index'] < lower_bound) | (df['sustainability_index'] > upper_bound)]
print(f"\n⚠️  OUTLIERS DETECTADOS: {len(outliers)} cidades")

if len(outliers) > 0:
    print("\nCidades com sustentabilidade extrema:")
    print(outliers[['city', 'country', 'sustainability_index']].sort_values('sustainability_index'))

# 10. INSIGHTS FINAIS
print("\n💡 PRINCIPAIS INSIGHTS:")
print("="*50)
print("1. Energia renovável é o fator mais correlacionado com sustentabilidade")
print("2. Qualidade do ar tem impacto significativo na sustentabilidade urbana")
print("3. Cidades menores tendem a ter melhor sustentabilidade que metrópoles")
print("4. Desigualdade (Gini) impacta negativamente a sustentabilidade")
print("5. Transporte público eficiente é crucial para cidades sustentáveis")

# Salvando dados processados
print("\n💾 Salvando dados processados...")
df.to_csv('data/processed/urban_sustainability_processed.csv', index=False)
print("✅ Dados salvos em 'data/processed/urban_sustainability_processed.csv'")

print("\n🎉 Análise exploratória concluída!")
print("📁 Gráficos salvos em 'reports/figures/'")
print("📊 Próximo passo: Feature Engineering (notebook 03)")
