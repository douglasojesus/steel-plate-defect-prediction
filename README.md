# Steel Plate Defect Prediction - Machine Learning 

A Supervised Learning project to the IA Curricular Unit.

## Pipeline

1. Data Loading and Preprocessing
2. Problem Definition and Target Identification
3. Model Selection and Parameter Tuning

Problema: Classificação Multi-Rótulo de Defeitos em Placas de Aço

Descrição:
O objetivo é prever a presença de vários tipos de defeitos em placas de aço com base em 
características geométricas e medidas de luminosidade. Cada placa pode ter um ou mais 
tipos de defeitos simultaneamente, caracterizando um problema de classificação multi-rótulo.

Tipos de Defeitos (Target Variables):
1. Pastry - Manchas irregulares paralelas à direção de laminação
2. Z_Scratch - Arranhões na direção Z (paralelos à laminação)
3. K_Scatch - Arranhões na direção K (perpendiculares à laminação)
4. Stains - Áreas descoloridas
5. Dirtiness - Presença de sujeira ou partículas
6. Bumps - Áreas elevadas na superfície
7. Other_Faults - Outros defeitos não categorizados

Características (Features):
O dataset contém 27 features numéricas que descrevem:
- Localização do defeito (coordenadas X e Y)
- Tamanho e área do defeito
- Características de luminosidade
- Propriedades do material (tipo de aço, espessura)
- Índices geométricos e estatísticos

Estratégia de Modelagem:

Estratégia Adotada:
1. Abordagem Multi-Rótulo: 
    - Vamos tratar isso como um problema de classificação multi-rótulo, onde cada instância pode pertencer a múltiplas classes simultaneamente.
    - Uma mesma placa pode apresentar múltiplos defeitos simultaneamente (por exemplo, uma placa pode ter tanto "Z_Scratch" quanto "Stains")

2. Transformação do Problema: 
    - Usaremos a abordagem Binary Relevance, que trata cada classe de defeito como um 
     problema de classificação binária independente.
    - Isso nos permite usar classificadores binários padrão enquanto ainda capturamos 
     as relações entre os defeitos.
    - Treinamento independente: Cada modelo aprende a identificar um defeito específico, ignorando inicialmente as correlações entre defeitos
    - Combinação das previsões: As saídas dos modelos individuais são combinadas para formar a previsão multi-rótulo final

3. Métricas de Avaliação:
   - Como temos múltiplas classes desbalanceadas (alguns defeitos são raros (como "Bumps") enquanto outros são mais comuns), focaremos em:
     * Precision (para minimizar falsos positivos)
     * Recall (para capturar o máximo de defeitos verdadeiros)
     * F1-score (média harmônica de precision e recall)
     * ROC-AUC (para avaliar a qualidade geral da classificação)

Balanceamento de classes (SMOTE (Synthetic Minority Over-sampling Technique)):
- Algumas classes de defeitos têm poucas amostras (como visto nos gráficos de distribuição)
- Modelos tradicionais tendem a ser enviesados para classes majoritárias 
- Geração sintética: Para cada classe minoritária, SMOTE cria novas instâncias artificiais interpolando exemplos reais
- Aplicação controlada: O balanceamento é feito apenas nos dados de treino (para evitar vazamento de dados)
- Por classe: Cada defeito é balanceado independentemente, mantendo as relações multi-rótulo originais

Modelos:

Selecionamos três algoritmos com características diferentes:
1. Random Forest: 
   - Robustos a outliers e features não normalizadas
   - Boa performance em problemas complexos
   - Pouco sensível a overfitting

2. Support Vector Machine (SVM):
   - Efetivo em espaços de alta dimensão
   - Versátil com diferentes kernels
   - Bom para problemas com margens de decisão claras

3. k-Nearest Neighbors (k-NN):
   - Simples e intuitivo
   - Baseado em instâncias, bom para problemas com clusters naturais
   - Sensível à escala dos dados (já normalizamos)



ANALISAR ONDE ENTRAR:


   # 2.1 Definição do Problema
"""
Problema: Classificação Multi-Rótulo de Defeitos em Placas de Aço

Descrição:
O objetivo é prever a presença de vários tipos de defeitos em placas de aço com base em 
características geométricas e medidas de luminosidade. Cada placa pode ter um ou mais 
tipos de defeitos simultaneamente, caracterizando um problema de classificação multi-rótulo.

Tipos de Defeitos (Target Variables):
1. Pastry - Manchas irregulares paralelas à direção de laminação
2. Z_Scratch - Arranhões na direção Z (paralelos à laminação)
3. K_Scatch - Arranhões na direção K (perpendiculares à laminação)
4. Stains - Áreas descoloridas
5. Dirtiness - Presença de sujeira ou partículas
6. Bumps - Áreas elevadas na superfície
7. Other_Faults - Outros defeitos não categorizados

Características (Features):
O dataset contém 27 features numéricas que descrevem:
- Localização do defeito (coordenadas X e Y)
- Tamanho e área do defeito
- Características de luminosidade
- Propriedades do material (tipo de aço, espessura)
- Índices geométricos e estatísticos
"""




# 2.2 Análise das Variáveis Alvo
# Plot da distribuição de cada tipo de defeito
plt.figure(figsize=(15, 10))
for i, defect in enumerate(defect_columns, 1):
    plt.subplot(3, 3, i)
    sns.countplot(x=steel_data[defect])
    plt.title(f'Distribuição de {defect}')
    plt.xlabel('')
    plt.ylabel('Contagem')
plt.tight_layout()
plt.show()

# Matriz de correlação entre os defeitos
plt.figure(figsize=(10, 8))
corr_matrix = steel_data[defect_columns].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlação entre Tipos de Defeitos')
plt.show()

# 2.3 Estratégia de Modelagem
"""
Estratégia Adotada:
1. Abordagem Multi-Rótulo: Vamos tratar isso como um problema de classificação multi-rótulo, 
   onde cada instância pode pertencer a múltiplas classes simultaneamente.

2. Transformação do Problema: 
   - Usaremos a abordagem Binary Relevance, que trata cada classe de defeito como um 
     problema de classificação binária independente.
   - Isso nos permite usar classificadores binários padrão enquanto ainda capturamos 
     as relações entre os defeitos.

3. Métricas de Avaliação:
   - Como temos múltiplas classes desbalanceadas, focaremos em:
     * Precision (para minimizar falsos positivos)
     * Recall (para capturar o máximo de defeitos verdadeiros)
     * F1-score (média harmônica de precision e recall)
     * ROC-AUC (para avaliar a qualidade geral da classificação)
"""

# 2.4 Balanceamento de Classes
from imblearn.over_sampling import SMOTE

"""
Observação sobre Balanceamento:
O dataset mostra desbalanceamento significativo em algumas classes de defeitos.
Vamos aplicar SMOTE (Synthetic Minority Over-sampling Technique) para cada classe 
de defeito individualmente durante o treinamento, usando um pipeline.
"""



# 3.4 Grade de Parâmetros para Tuning
param_grids = {
    'rf': {
        'multioutputclassifier__estimator__n_estimators': [100, 200],
        'multioutputclassifier__estimator__max_depth': [None, 10, 20],
        'multioutputclassifier__estimator__min_samples_split': [2, 5]
    },
    'svm': {
        'multioutputclassifier__estimator__C': [0.1, 1, 10],
        'multioutputclassifier__estimator__kernel': ['linear', 'rbf'],
        'multioutputclassifier__estimator__gamma': ['scale', 'auto']
    },
    'knn': {
        'multioutputclassifier__estimator__n_neighbors': [3, 5, 7],
        'multioutputclassifier__estimator__weights': ['uniform', 'distance'],
        'multioutputclassifier__estimator__metric': ['euclidean', 'manhattan']
    }
}

# 3.5 Treinamento e Tuning dos Modelos
best_models = {}
for name in pipelines.keys():
    print(f"\nIniciando GridSearchCV para {name.upper()}")
    
    # Cria o GridSearchCV para o modelo atual
    grid_search = GridSearchCV(
        estimator=pipelines[name],
        param_grid=param_grids[name],
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    # Ajusta o modelo aos dados de treino
    grid_search.fit(X_train, y_train)
    
    # Armazena o melhor modelo
    best_models[name] = grid_search.best_estimator_
    
    # Exibe os melhores parâmetros
    print(f"Melhores parâmetros para {name.upper()}:")
    print(grid_search.best_params_)
    
    # Avaliação no conjunto de teste
    y_pred = best_models[name].predict(X_test)
    y_prob = best_models[name].predict_proba(X_test)
    
    # Calcula ROC-AUC para cada classe
    print("\nRelatório de Classificação:")
    for i, defect in enumerate(defect_columns):
        auc = roc_auc_score(y_test[defect], y_prob[i][:, 1])
        print(f"{defect}: AUC = {auc:.4f}")
    
    print("\n" + "="*80 + "\n")

# 3.6 Seleção do Melhor Modelo
# Compara o desempenho dos modelos e seleciona o melhor
best_model_name = max(best_models.keys(), 
                     key=lambda x: roc_auc_score(y_test, 
                                               [best_models[x].predict_proba(X_test)[i][:,1] for i in range(7)], 
                                               multi_class='ovr'))
best_model = best_models[best_model_name]

print(f"\nMelhor modelo: {best_model_name.upper()}")
print("Parâmetros do melhor modelo:")
print(best_model.get_params())

# 3.7 Avaliação Final do Melhor Modelo
print("\nAvaliação detalhada do melhor modelo:")
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)

# Relatório de classificação para cada classe
for i, defect in enumerate(defect_columns):
    print(f"\nDefeito: {defect}")
    print(classification_report(y_test[defect], y_pred[:, i]))
    
    # Plot da curva ROC
    from sklearn.metrics import RocCurveDisplay
    RocCurveDisplay.from_estimator(
        best_model.named_steps['multioutputclassifier'].estimators_[i],
        X_test, 
        y_test[defect]
    )
    plt.title(f'Curva ROC para {defect}')
    plt.show()

# 3.8 Feature Importance (para modelos baseados em árvores)
if hasattr(best_model.named_steps['multioutputclassifier'].estimators_[0], 'feature_importances_'):
    plt.figure(figsize=(12, 8))
    for i, defect in enumerate(defect_columns):
        importances = best_model.named_steps['multioutputclassifier'].estimators_[i].feature_importances_
        indices = np.argsort(importances)[-10:]  # Top 10 features
        plt.subplot(3, 3, i+1)
        plt.title(f'Feature Importance - {defect}')
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [selected_features[j] for j in indices])
    plt.tight_layout()
    plt.show()