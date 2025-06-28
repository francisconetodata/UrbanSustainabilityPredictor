#!/usr/bin/env python3
"""
Urban Sustainability Predictor - Pipeline Principal
Executa todo o pipeline de machine learning de ponta a ponta
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
import json

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class UrbanSustainabilityPipeline:
    """Pipeline completo para predição de sustentabilidade urbana"""
    
    def __init__(self, config_path='config/config.json'):
        """Inicializa o pipeline com configurações"""
        self.config = self._load_config(config_path)
        self.model = None
        self.scaler = None
        self.feature_columns = [
            'population', 'gdp_per_capita', 'renewable_energy_pct', 'air_quality_index',
            'public_transport_coverage', 'waste_management_score', 'green_space_pct',
            'education_index', 'healthcare_index', 'gini_coefficient', 'co2_emissions_per_capita'
        ]
        
    def _load_config(self, config_path):
        """Carrega configurações do projeto"""
        default_config = {
            "data_path": "data/",
            "model_path": "models/",
            "reports_path": "reports/",
            "random_seed": 42,
            "test_size": 0.2,
            "val_size": 0.2,
            "cv_folds": 5
        }
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return {**default_config, **config}
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using default config.")
            return default_config
    
    def create_directory_structure(self):
        """Cria estrutura de diretórios do projeto"""
        directories = [
            'data/raw', 'data/processed', 'data/external',
            'models', 'reports/figures', 'logs',
            'notebooks', 'src/data', 'src/features', 
            'src/models', 'src/visualization'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Directory created: {directory}")
    
    def generate_synthetic_data(self, n_cities=1000):
        """Gera dados sintéticos para demonstração"""
        logger.info(f"Generating synthetic data for {n_cities} cities...")
        
        np.random.seed(self.config['random_seed'])
        
        # Gerando dados sintéticos realistas
        data = {
            'city': [f'City_{i:04d}' for i in range(n_cities)],
            'country': np.random.choice([
                'Brazil', 'USA', 'Germany', 'Japan', 'Canada', 
                'Australia', 'UK', 'France', 'Sweden', 'Netherlands'
            ], n_cities),
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
        
        df = pd.DataFrame(data)
        
        # Criando target baseado em combinação realista das features
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
        
        return df
    
    def preprocess_data(self, df):
        """Preprocessa os dados para modelagem"""
        logger.info("Preprocessing data...")
        
        # Verificar valores ausentes
        missing_values = df.isnull().sum()
        if missing_values.any():
            logger.warning(f"Missing values found: {missing_values[missing_values > 0].to_dict()}")
            df = df.fillna(df.mean(numeric_only=True))
        
        # Remover outliers extremos
        for col in self.feature_columns:
            Q1 = df[col].quantile(0.01)
            Q3 = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=Q1, upper=Q3)
        
        # Adicionar features engineered
        df['population_density_log'] = np.log1p(df['population'])
        df['economic_sustainability'] = df['gdp_per_capita'] / (1 + df['gini_coefficient'])
        df['environmental_score'] = (df['renewable_energy_pct'] + 
                                    (100 - df['air_quality_index']) + 
                                    df['green_space_pct']) / 3
        
        logger.info(f"Data preprocessing completed. Shape: {df.shape}")
        return df
    
    def train_model(self, df):
        """Treina o modelo de machine learning"""
        logger.info("Training model...")
        
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import RobustScaler
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, r2_score
        import xgboost as xgb
        
        # Preparar dados
        X = df[self.feature_columns]
        y = df['sustainability_index']
        
        # Divisão dos dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config['test_size'], 
            random_state=self.config['random_seed']
        )
        
        # Padronização
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Treinar múltiplos modelos
        models = {
            'RandomForest': RandomForestRegressor(
                n_estimators=200, max_depth=10, 
                random_state=self.config['random_seed']
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                random_state=self.config['random_seed']
            )
        }
        
        best_score = 0
        best_model_name = ""
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            if name == 'XGBoost':
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
            else:
                model.fit(X_train_scaled, y_train)
                predictions = model.predict(X_test_scaled)
            
            r2 = r2_score(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            
            logger.info(f"{name} - R²: {r2:.3f}, RMSE: {rmse:.3f}")
            
            if r2 > best_score:
                best_score = r2
                best_model_name = name
                self.model = model
                self.best_predictions = predictions
                self.test_y = y_test
        
        logger.info(f"Best model: {best_model_name} with R²: {best_score:.3f}")
        return self.model, self.scaler
    
    def save_model(self):
        """Salva o modelo treinado"""
        logger.info("Saving model...")
        
        model_path = os.path.join(self.config['model_path'], 'urban_sustainability_model.pkl')
        scaler_path = os.path.join(self.config['model_path'], 'scaler.pkl')
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        # Salvar metadados
        metadata = {
            'model_type': type(self.model).__name__,
            'features': self.feature_columns,
            'r2': float(r2_score(self.test_y, self.best_predictions)),
            'rmse': float(np.sqrt(mean_squared_error(self.test_y, self.best_predictions))),
            'mae': float(np.mean(np.abs(self.test_y - self.best_predictions))),
            'accuracy_pct': float((1 - np.mean(np.abs(self.test_y - self.best_predictions))/self.test_y.mean())*100),
            'training_date': datetime.now().isoformat(),
            'training_samples': len(self.test_y)
        }
        
        metadata_path = os.path.join(self.config['model_path'], 'model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Scaler saved to {scaler_path}")
        logger.info(f"Metadata saved to {metadata_path}")
    
    def predict_single(self, city_data):
        """Faz predição para uma única cidade"""
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained or loaded")
        
        # Converter para DataFrame
        if isinstance(city_data, dict):
            city_df = pd.DataFrame([city_data])
        else:
            city_df = city_data
        
        # Selecionar features corretas
        X = city_df[self.feature_columns]
        
        # Aplicar mesmo preprocessing
        if type(self.model).__name__ == 'XGBRegressor':
            prediction = self.model.predict(X)
        else:
            X_scaled = self.scaler.transform(X)
            prediction = self.model.predict(X_scaled)
        
        return prediction[0] if len(prediction) == 1 else prediction
    
    def generate_report(self, df):
        """Gera relatório de análise"""
        logger.info("Generating analysis report...")
        
        report = {
            'dataset_info': {
                'total_cities': len(df),
                'countries': df['country'].nunique(),
                'avg_sustainability': df['sustainability_index'].mean(),
                'std_sustainability': df['sustainability_index'].std()
            },
            'top_countries': df.groupby('country')['sustainability_index'].mean().sort_values(ascending=False).head().to_dict(),
            'correlations': df[self.feature_columns + ['sustainability_index']].corr()['sustainability_index'].sort_values(ascending=False).to_dict()
        }
        
        report_path = os.path.join(self.config['reports_path'], 'analysis_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Report saved to {report_path}")
        return report
    
    def run_full_pipeline(self, n_cities=1000):
        """Executa pipeline completo"""
        logger.info("🚀 Starting Urban Sustainability Predictor Pipeline")
        
        # 1. Criar estrutura de diretórios
        self.create_directory_structure()
        
        # 2. Gerar/carregar dados
        df = self.generate_synthetic_data(n_cities)
        
        # 3. Preprocessar dados
        df = self.preprocess_data(df)
        
        # 4. Salvar dados processados
        processed_path = os.path.join(self.config['data_path'], 'processed', 'urban_sustainability_processed.csv')
        df.to_csv(processed_path, index=False)
        logger.info(f"Processed data saved to {processed_path}")
        
        # 5. Treinar modelo
        self.train_model(df)
        
        # 6. Salvar modelo
        self.save_model()
        
        # 7. Gerar relatório
        self.generate_report(df)
        
        logger.info("✅ Pipeline completed successfully!")
        
        # 8. Exemplo de predição
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
        
        prediction = self.predict_single(example_city)
        logger.info(f"🎯 Example prediction: {prediction:.1f}/100")
        
        return df

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description='Urban Sustainability Predictor Pipeline')
    parser.add_argument('--n-cities', type=int, default=1000, help='Number of cities to generate')
    parser.add_argument('--config', type=str, default='config/config.json', help='Config file path')
    
    args = parser.parse_args()
    
    # Criar diretório de logs
    os.makedirs('logs', exist_ok=True)
    
    # Executar pipeline
    pipeline = UrbanSustainabilityPipeline(args.config)
    df = pipeline.run_full_pipeline(args.n_cities)
    
    print(f"\n🎉 Pipeline concluído com sucesso!")
    print(f"📊 {len(df)} cidades processadas")
    print(f"📁 Arquivos salvos em: data/, models/, reports/")
    print(f"🚀 Para executar o dashboard: streamlit run streamlit_app/app.py")

if __name__ == "__main__":
    main()
