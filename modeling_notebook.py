# Urban Sustainability Predictor - Modelagem
# 04_modeling.ipynb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

print("🤖 INICIANDO MODELAGEM DE SUSTENTABILIDADE URBANA")
print("="*60)

# 1. CARREGAMENTO E PREPARAÇÃO DOS DADOS
print("📁 Carregando dados processados...")
df = pd.read_csv('data/processed/urban_sustainability_processed.csv')

# Features para modelagem
feature_columns = [
    'population', 'gdp_per_capita', 'renewable_energy_pct', 'air_quality_index',
    'public_transport_coverage', 'waste_management_score', 'green_space_pct',
    'education_index', 'healthcare_index', 'gini_coefficient', 'co2_emissions_per_capita'
]

X = df[feature_columns]
y = df['sustainability_index']

print(f"✅ Dados carregados: {X.shape[0]} amostras, {X.shape[1]} features")
print(f"🎯 Target: {y.name}")

# 2. DIVISÃO DOS DADOS
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print(f"\n📊 Divisão dos dados:")
print(f"   Treino: {X_train.shape[0]} amostras")
print(f"   Validação: {X_val.shape[0]} amostras") 
print(f"   Teste: {X_test.shape[0]} amostras")

# 3. PADRONIZAÇÃO DAS FEATURES
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("✅ Features padronizadas com RobustScaler")

# 4. DEFINIÇÃO DOS MODELOS
models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
    'SVR': SVR(kernel='rbf', C=1.0)
}

# 5. TREINAMENTO E AVALIAÇÃO INICIAL
print(f"\n🏋️ TREINANDO MODELOS...")
print("="*60)

results = {}
predictions = {}

for name, model in models.items():
    print(f"📈 Treinando {name}...")
    
    # Treinar modelo
    if name in ['Linear Regression', 'Ridge', 'Lasso', 'SVR']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_val_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
    
    # Calcular métricas
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    results[name] = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    }
    
    predictions[name] = y_pred
    
    print(f"   RMSE: {rmse:.3f} | R²: {r2:.3f}")

# 6. COMPARAÇÃO DOS MODELOS
print(f"\n📊 RESULTADOS COMPARATIVOS:")
print("="*60)
results_df = pd.DataFrame(results).T.round(3)
print(results_df.sort_values('R²', ascending=False))

# Visualização dos resultados
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# RMSE
results_df.sort_values('RMSE')['RMSE'].plot(kind='bar', ax=axes[0,0], color='lightcoral')
axes[0,0].set_title('RMSE por Modelo')
axes[0,0].set_ylabel('RMSE')

# R²
results_df.sort_values('R²', ascending=False)['R²'].plot(kind='bar', ax=axes[0,1], color='lightblue')
axes[0,1].set_title('R² por Modelo')
axes[0,1].set_ylabel('R²')

# MAE
results_df.sort_values('MAE')['MAE'].plot(kind='bar', ax=axes[1,0], color='lightgreen')
axes[1,0].set_title('MAE por Modelo')
axes[1,0].set_ylabel('MAE')

# Predições vs Real (melhor modelo)
best_model_name = results_df.sort_values('R²', ascending=False).index[0]
best_predictions = predictions[best_model_name]

axes[1,1].scatter(y_val, best_predictions, alpha=0.6)
axes[1,1].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
axes[1,1].set_xlabel('Valores Reais')
axes[1,1].set_ylabel('Valores Preditos')
axes[1,1].set_title(f'Predições vs Real - {best_model_name}')

plt.tight_layout()
plt.savefig('reports/figures/model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. OTIMIZAÇÃO DO MELHOR MODELO
print(f"\n🔧 OTIMIZANDO O MELHOR MODELO: {best_model_name}")
print("="*60)

if best_model_name == 'XGBoost':
    # Grid Search para XGBoost
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    best_model = xgb.XGBRegressor(random_state=42)
    grid_search = GridSearchCV(best_model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    optimized_model = grid_search.best_estimator_
    print(f"🎯 Melhores parâmetros: {grid_search.best_params_}")

elif best_model_name == 'Random Forest':
    # Grid Search para Random Forest
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    best_model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(best_model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    optimized_model = grid_search.best_estimator_
    print(f"🎯 Melhores parâmetros: {grid_search.best_params_}")

else:
    optimized_model = models[best_model_name]

# 8. AVALIAÇÃO FINAL NO CONJUNTO DE TESTE
print(f"\n🧪 AVALIAÇÃO FINAL NO CONJUNTO DE TESTE")
print("="*60)

if best_model_name in ['Linear Regression', 'Ridge', 'Lasso', 'SVR']:
    final_predictions = optimized_model.predict(X_test_scaled)
else:
    final_predictions = optimized_model.predict(X_test)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
final_mae = mean_absolute_error(y_test, final_predictions)
final_r2 = r2_score(y_test, final_predictions)

print(f"📊 Métricas Finais:")
print(f"   RMSE: {final_rmse:.3f}")
print(f"   MAE: {final_mae:.3f}")
print(f"   R²: {final_r2:.3f}")
print(f"   Acurácia: {(1 - final_mae/y_test.mean())*100:.1f}%")

# 9. IMPORTÂNCIA DAS FEATURES
if hasattr(optimized_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': optimized_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔍 IMPORTÂNCIA DAS FEATURES:")
    print("="*60)
    for idx, row in feature_importance.iterrows():
        print(f"{row['feature']:<25}: {row['importance']:.3f}")
    
    # Visualização da importância
    plt.figure(figsize=(12, 8))
    sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
    plt.title('Importância das Features - Modelo Final')
    plt.xlabel('Importância')
    plt.tight_layout()
    plt.savefig('reports/figures/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

# 10. ANÁLISE DE RESÍDUOS
residuals = y_test - final_predictions

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Resíduos vs Predições
axes[0].scatter(final_predictions, residuals, alpha=0.6)
axes[0].axhline(y=0, color='r', linestyle='--')
axes[0].set_xlabel('Valores Preditos')
axes[0].set_ylabel('Resíduos')
axes[0].set_title('Resíduos vs Predições')

# QQ-plot dos resíduos
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[1])
axes[1].set_title('Q-Q Plot dos Resíduos')

# Histograma dos resíduos
axes[2].hist(residuals, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
axes[2].set_xlabel('Resíduos')
axes[2].set_ylabel('Frequência')
axes[2].set_title('Distribuição dos Resíduos')

plt.tight_layout()
plt.savefig('reports/figures/residual_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 11. SALVANDO O MODELO FINAL
print(f"\n💾 SALVANDO MODELO E SCALER...")
joblib.dump(optimized_model, 'models/urban_sustainability_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

# Salvando metadados do modelo
model_metadata = {
    'model_type': best_model_name,
    'features': feature_columns,
    'rmse': final_rmse,
    'mae': final_mae,
    'r2': final_r2,
    'accuracy_pct': (1 - final_mae/y_test.mean())*100
}

import json
with open('models/model_metadata.json', 'w') as f:
    json.dump(model_metadata, f, indent=2)

print("✅ Modelo salvo em 'models/urban_sustainability_model.pkl'")
print("✅ Scaler salvo em 'models/scaler.pkl'")
print("✅ Metadados salvos em 'models/model_metadata.json'")

# 12. PREDIÇÃO DE EXEMPLO
print(f"\n🔮 EXEMPLO DE PREDIÇÃO:")
print("="*60)

# Cidade exemplo
example_city = {
    'population': 500000,
    'gdp_per_capita': 45000,
    'renewable_energy_pct': 60,
    'air_quality_index': 45,
    'public_transport_coverage': 80,
    'waste_management_score': 75,
    'green_space_pct': 25,
    'education_index': 85,
    'healthcare_index': 80,
    'gini_coefficient': 0.35,
    'co2_emissions_per_capita': 8
}

example_df = pd.DataFrame([example_city])
if best_model_name in ['Linear Regression', 'Ridge', 'Lasso', 'SVR']:
    example_scaled = scaler.transform(example_df)
    prediction = optimized_model.predict(example_scaled)[0]
else:
    prediction = optimized_model.predict(example_df)[0]

print("Características da cidade exemplo:")
for key, value in example_city.items():
    print(f"   {key}: {value}")
print(f"\n🎯 Índice de Sustentabilidade Predito: {prediction:.1f}/100")

print(f"\n🎉 MODELAGEM CONCLUÍDA!")
print(f"🏆 Melhor modelo: {best_model_name}")
print(f"📊 Performance final: R² = {final_r2:.3f}, RMSE = {final_rmse:.2f}")
print(f"📁 Arquivos salvos em 'models/' e gráficos em 'reports/figures/'")
