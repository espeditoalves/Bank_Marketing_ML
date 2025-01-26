import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def outlier_analysis(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Análise e filtra os outliers de uma coluna específica de um DataFrame.

    Parâmetros:
    df (pd.DataFrame): O DataFrame contendo os dados.
    column (str): O nome da coluna a ser analisada.

    Retorna:
    pd.Series: Uma série contendo os valores outliers da coluna especificada.
    """
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    print(f'Coluna: {column}')
    print(f'Limite inferior: {lower_bound}')
    print(f'Limite superior: {upper_bound}')
    print(f'Quantidade de outliers: {df[column][(df[column] < lower_bound) | (df[column] > upper_bound)].count()}')
    print(f'Percentual de outliers: {df[column][(df[column] < lower_bound) | (df[column] > upper_bound)].count() / df.shape[0] * 100:.2f}%')

    result = df[column][(df[column] < lower_bound) | (df[column] > upper_bound)]
    return result

def analyze_column_transform(df: pd.DataFrame, column: str) -> None:
    """
    Análise descritiva e visual de uma coluna específica de um DataFrame.
    Aplica transformação logarítmica e exibe histogramas lado a lado.

    Parâmetros:
    df (pd.DataFrame): O DataFrame contendo os dados.
    column (str): O nome da coluna a ser analisada.
    """
    # Transformação logarítmica
    log_transformed = np.log1p(df[column])

    # Análise descritiva
    descritiva = pd.concat(
        [df[column].describe(), log_transformed.describe()], axis=1
    )
    descritiva.columns = ['Original', 'Log Transformada']
    print(descritiva)

    # -----------GRAFICO----------------#
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    # Histograma original
    axes[0].hist(df[column], bins=50)
    axes[0].set_title('Histograma Original')
    axes[0].set_xlabel(column)
    axes[0].set_ylabel('Frequência')

    # Histograma transformado
    axes[1].hist(log_transformed, bins=50)
    axes[1].set_title('Histograma Transformado (Log)')
    axes[1].set_xlabel(f'{column} (Transformada)')
    axes[1].set_ylabel('Frequência')

    plt.tight_layout()
    plt.show()
    
import numpy as np
import pandas as pd
from typing import Tuple
# Definindo as categorias de preditividade com base no IV
def categorize_iv(iv_value):
    if iv_value < 0.02:
        return "Not useful for prediction"
    elif 0.02 <= iv_value < 0.1:
        return "Weak predictive Power"
    elif 0.1 <= iv_value < 0.3:
        return "Medium predictive Power"
    elif 0.3 <= iv_value < 0.5:
        return "Strong predictive Power"
    else:
        return "Suspicious Predictive Power"

# Aplicando a função ao DataFrame iv
# iv['Predictiveness'] = iv['IV'].apply(categorize_iv)


def iv_woe(
    data: pd.DataFrame, 
    target: str, 
    bins: int = 10, 
    show_woe: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculates the Weight of Evidence (WOE) and Information Value (IV) for all independent variables in a DataFrame.

    Args:
        data (pd.DataFrame): DataFrame containing the data.
        target (str): Column name of the binary target variable.
        bins (int): Number of bins to use for WOE calculation (default is 10).
        show_woe (bool): If True, print the WOE table for each variable (default is True).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: 
            - DataFrame with IV values for each independent variable.
            - DataFrame with WOE values for each bin of each independent variable.

    Reference:  https://lucastiagooliveira.github.io/datascience/iv/woe/python/2020/12/15/iv_woe.html      
    Reference: https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html
    Reference: https://www.linkedin.com/pulse/information-value-uma-excelente-t%C3%A9cnica-para-de-j%C3%BAlia-de-moura-ertel/
    Refernce: https://www.analyticsvidhya.com/blog/2021/06/understand-weight-of-evidence-and-information-value/
    """
    
    # Empty Dataframe
    newDF, woeDF = pd.DataFrame(), pd.DataFrame()
    
    # Extract Column Names
    cols = data.columns
    
    # Run WOE and IV on all the independent variables
    for ivars in cols[~cols.isin([target])]:
        if (data[ivars].dtype.kind in 'bifc') and (len(np.unique(data[ivars])) > 10):
            binned_x = pd.qcut(data[ivars], bins, duplicates='drop')
            d0 = pd.DataFrame({'x': binned_x, 'y': data[target]})
        else:
            d0 = pd.DataFrame({'x': data[ivars], 'y': data[target]})
        d = d0.groupby("x", as_index=False, observed=False).agg({"y": ["count", "sum"]})
        d.columns = ['Cutoff', 'N', 'Events']
        # Adicionamos um pequeno valor (neste caso, 0.5) para garantir que não haja divisões por zero e evitar que o logaritmo de zero seja calculado.
        d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()
        d['Non-Events'] = d['N'] - d['Events']
        d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()
        d['WoE'] = np.log(d['% of Events'] / d['% of Non-Events'])
        d['IV'] = d['WoE'] * (d['% of Events'] - d['% of Non-Events'])
        d.insert(loc=0, column='Variable', value=ivars)

        temp = pd.DataFrame({"Variable": [ivars], "IV": [d['IV'].sum()]}, columns=["Variable", "IV"])
        newDF = pd.concat([newDF, temp], axis=0)
        woeDF = pd.concat([woeDF, d], axis=0)
        
        newDF = newDF.sort_values(by='IV', ascending=False)
        newDF['Predictiveness'] = newDF['IV'].apply(categorize_iv)

        # Show WOE Table
        if show_woe:
            print(d)
    
    return newDF, woeDF

import pandas as pd
from typing import List

def corr_features_target(
    df: pd.DataFrame, 
    target_column: str, 
    numerical_features: List[str],
    corr_limit: float = 0.1
) -> pd.DataFrame:
    """
    Calcula as features relevantes com base na correlação com a coluna alvo.

    Parâmetros:
    df (pd.DataFrame): O DataFrame contendo os dados.
    target_column (str): O nome da coluna alvo.
    numerical_features (List[str]): A lista de nomes das colunas numéricas a serem analisadas.
    corr_limit (float): O valor mínimo de correlação para uma feature ser considerada relevante.

    Retorna:
    pd.DataFrame: Um DataFrame contendo as features relevantes e suas correlações com a coluna alvo.
    """
    
    # Calcular a matriz de correlação de Pearson
    corr = df[numerical_features].corr(method='pearson')
    
    # Calcular a correlação absoluta com a coluna alvo
    corr_target = abs(corr[target_column])
    
    # Selecionar features com correlação superior ao limite
    relevant_features = corr_target[corr_target > corr_limit].index.tolist()
    
    # Criar uma tabela com features relevantes e suas correlações com a coluna alvo
    relevant_corr_table = pd.DataFrame({
        'Feature': relevant_features,
        'Correlation': corr.loc[relevant_features, target_column]
    }).reset_index(drop=True)
    relevant_corr_table = relevant_corr_table.sort_values(by='Correlation', ascending=False)
    
    return relevant_corr_table