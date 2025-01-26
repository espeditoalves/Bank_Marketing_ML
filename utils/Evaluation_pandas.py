import pandas as pd
from typing import Dict

def confusion_matriz(df: pd.DataFrame, target: str, y_pred: str) -> Dict[str, int]:
    """
    Calcula a matriz de confusão para um conjunto de dados fornecido.

    Parâmetros:
    df (pd.DataFrame): DataFrame contendo os dados reais e as previsões.
    target (str): Nome da coluna que contém os valores reais.
    y_pred (str): Nome da coluna que contém os valores previstos.

    Retorna:
    Dict[str, int]: 
            Um dicionário contendo os valores de 
            Verdadeiros Positivos (VP), Falsos Positivos (FP), 
            Verdadeiros Negativos (VN) e Falsos Negativos (FN).
    """
    VP = df.query(f'{target} == 1 and {y_pred} == 1').shape[0]
    VN = df.query(f'{target} == 0 and {y_pred} == 0').shape[0]
    FP = df.query(f'{target} == 0 and {y_pred} == 1').shape[0]
    FN = df.query(f'{target} == 1 and {y_pred} == 0').shape[0]
    
    return {
        'Verdadeiros Positivos (VP ou TP)': VP,
        'Falsos Positivos (FP)': FP,
        'Verdadeiros Negativos (VN ou TN)': VN,
        'Falsos Negativos (FN)': FN
    }

import pandas as pd
import numpy as np
from typing import Tuple

def calculate_ks(
    df: pd.DataFrame, 
    target_col: str,
    prob_col: str, 
    ) -> Tuple[pd.DataFrame, float]:
    """
    Calcula a tabela KS e o valor KS a partir de um DataFrame com colunas de probabilidade e target.
    Ref: https://www.listendata.com/2019/07/KS-Statistics-Python.html
    Ref: https://medium.com/data-hackers/kolmogorov-smirnov-fb90394ca122

    Args:
    df (pd.DataFrame): DataFrame contendo as colunas de probabilidade e target.
    target_col (str): Nome da coluna target.
    prob_col (str): Nome da coluna de probabilidade.

    Returns:
    Tuple[pd.DataFrame, float]: Retorna a tabela KS e o valor KS.
    """
    
    # Defina os limites dos intervalos
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = ['[0-0.1]', '(0.1-0.2]', '(0.2-0.3]', '(0.3-0.4]', '(0.4-0.5]', '(0.5-0.6]', '(0.6-0.7]', '(0.7-0.8]', '(0.8-0.9]', '(0.9-1.0]']
    df['Range'] = pd.cut(df[prob_col], bins=bins, labels=labels, include_lowest=True, right=True)

    df_group = df.groupby(['Range', target_col], observed=False).count().reset_index()
    kstable = df_group.pivot(index='Range', columns=target_col, values=prob_col).fillna(0)
    kstable = kstable.rename(columns={0: 'nonevents' , 1: 'events'})
    kstable = kstable[['events', 'nonevents']]

    # Remova o nível de índice extra e o rótulo do eixo
    kstable.reset_index(inplace=True)
    kstable.columns.name = None

    kstable['Total'] = kstable['events'] + kstable['nonevents']

    # Calcular as taxas de eventos e não eventos
    sum_eventos = kstable['events'].sum()
    sum_nonevents = kstable['nonevents'].sum()
    kstable['event_rate %'] = kstable['events'] / sum_eventos
    kstable['nonevent_rate %'] = kstable['nonevents'] / sum_nonevents
    kstable['cum_event_rate %'] = kstable['event_rate %'].cumsum()
    kstable['cum_nonevent_rate %'] = kstable['nonevent_rate %'].cumsum()
    kstable['KS'] = abs(np.round(kstable['cum_event_rate %'] - kstable['cum_nonevent_rate %'], 4) * 100)

    # Formatar as colunas para porcentagens
    kstable['event_rate %']= kstable['event_rate %'].apply('{0:.2%}'.format)
    kstable['nonevent_rate %']= kstable['nonevent_rate %'].apply('{0:.2%}'.format)
    kstable['cum_event_rate %']= kstable['cum_event_rate %'].apply('{0:.2%}'.format)
    kstable['cum_nonevent_rate %']= kstable['cum_nonevent_rate %'].apply('{0:.2%}'.format)
    
    kstable.index = range(1, 11)
    kstable.index.rename('Decile', inplace=True)

    # Display KS
    ks_value = max(kstable['KS'])
    ks_decile = kstable.index[kstable['KS'] == ks_value][0]
    # print(f'KS: {ks_value}, Decile {ks_decile}')
    print('----')

    return kstable, ks_value

import matplotlib.pyplot as plt
def plot_ks(df: pd.DataFrame, target_col: str, prob_col: str):
    """
    Plota as distribuições acumuladas de eventos e não eventos com base na tabela KS.

    Args:
        df (pd.DataFrame): DataFrame contendo as colunas de probabilidade e target.
        target_col (str): Nome da coluna target.
        prob_col (str): Nome da coluna de probabilidade.
    """
    # Calcular a tabela KS e o valor KS
    kstable, ks_value = calculate_ks(df, target_col, prob_col)

    # Converter as colunas de taxa acumulada de volta para números
    kstable['cum_event_rate %'] = kstable['cum_event_rate %'].str.rstrip('%').astype('float') / 100.0
    kstable['cum_nonevent_rate %'] = kstable['cum_nonevent_rate %'].str.rstrip('%').astype('float') / 100.0

    # Criar a figura e os eixos do gráfico
    fig, ax = plt.subplots(figsize=(7, 3))

    # Plotar as linhas separadamente
    ax.plot(kstable.index, kstable['cum_event_rate %'], color='red', linestyle='-')
    ax.plot(kstable.index, kstable['cum_nonevent_rate %'], color='blue', linestyle='-')

    # Plotar os círculos contornados sem preenchimento
    ax.scatter(kstable.index, kstable['cum_event_rate %'], color='red', edgecolor='red', facecolor='none', s=50, label='Eventos Ac.')
    ax.scatter(kstable.index, kstable['cum_nonevent_rate %'], color='blue', edgecolor='blue', facecolor='none', s=50, label='Não eventos Ac.')

    # Adicionar uma linha vertical no ponto KS
    ax.axvline(x=kstable.index[kstable['KS'] == ks_value][0], color='green', linestyle='--', label=f'KS = {ks_value}')

    # Definir rótulos e título
    ax.set_xlabel('Decil')
    ax.set_ylabel('Taxa acumulada')
    ax.set_title('Distribuição acumulada de eventos e não eventos')

    # Definir ticks e limites do eixo y
    ax.set_yticks(ticks=[i / 10 for i in range(11)])
    ax.set_yticklabels([f'{i*10}%' for i in range(11)])
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True)
    plt.show()
    
import numpy as np
import pandas as pd
from typing import Tuple

def column_event_distribution(
    df: pd.DataFrame, 
    target_col: str, 
    prob_col: str
) -> pd.DataFrame:
    """
    Calculates the distribution of events and non-events by probability range.

    Args:
        df (pd.DataFrame): DataFrame containing the probability and target columns.
        target_col (str): Name of the target column.
        prob_col (str): Name of the probability column.

    Returns:
        pd.DataFrame: A DataFrame with the distribution of events and non-events by probability range.
    """
    
    # Define the probability range intervals
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = [
        '[0-0.1]', '(0.1-0.2]', '(0.2-0.3]', '(0.3-0.4]', '(0.4-0.5]', 
        '(0.5-0.6]', '(0.6-0.7]', '(0.7-0.8]', '(0.8-0.9]', '(0.9-1.0]'
    ]
    df['Range'] = pd.cut(
        df[prob_col], bins=bins, labels=labels, include_lowest=True, right=True
        )

    df_group = df.groupby(['Range', target_col], observed=False).count().reset_index()
    table = df_group.pivot(index='Range', columns=target_col, values=prob_col).fillna(0)
    table = table.rename(columns={0: 'nonevents', 1: 'events'})
    table = table[['events', 'nonevents']]

    # Remove the extra index level and axis label
    table.reset_index(inplace=True)
    table.columns.name = None
    table.index = range(1, 11)
    
    sum_events = table['events'].sum()
    sum_nonevents = table['nonevents'].sum()
    
    table['event_rate %'] = np.round(table['events'] / sum_events, 4) * 100
    table['nonevent_rate %'] = np.round(table['nonevents'] / sum_nonevents, 4) * 100
    return table

def plot_event_distribution(df: pd.DataFrame, 
    target_col: str, 
    prob_col: str):
    """
    Plota um gráfico de barras mostrando a distribuição de eventos e não eventos por faixa de probabilidade.

    Args:
        table (pd.DataFrame): DataFrame contendo a distribuição de eventos e não eventos por faixa.
    """
    table = column_event_distribution(df = df,  target_col = target_col, prob_col = prob_col)
    # Configurar os dados para o gráfico
    faixas = table['Range']
    event_rate = table['event_rate %']
    nonevent_rate = table['nonevent_rate %']

    # Criar a figura e os eixos do gráfico
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plotar o gráfico de barras
    bar_width = 0.4
    index = range(len(faixas))

    bars1 = ax.bar(index, event_rate, bar_width, label='Eventos %', color='red', edgecolor='black')
    bars2 = ax.bar([i + bar_width for i in index], nonevent_rate, bar_width, label='Não eventos %', color='blue', edgecolor='black')

    # Definir rótulos e título
    ax.set_xlabel('Faixa de probabilidade')
    ax.set_ylabel('Percentual')
    ax.set_title('Distribuição de eventos e não eventos por faixa')
    ax.set_xticks([i + bar_width / 2 for i in index])
    ax.set_xticklabels(faixas, rotation=45)
    
    # Ajustar o eixo y para exibir percentuais e limitar entre 0 e 100%
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}%'))

    # Adicionar legenda e grade
    ax.legend()
    ax.grid(True)

    # Exibir o gráfico
    plt.tight_layout()
    plt.show()
    



import numpy as np
def line_event_distribution(
    df: pd.DataFrame, 
    target_col: str, 
    prob_col: str
) -> pd.DataFrame:
    """
    Calculates the distribution of events and non-events by probability range in line.

    Args:
        df (pd.DataFrame): DataFrame containing the probability and target columns.
        target_col (str): Name of the target column.
        prob_col (str): Name of the probability column.

    Returns:
        pd.DataFrame: A DataFrame with the distribution of events and non-events by probability range in line.
    """
    
    # Define the probability range intervals
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = [
        '[0-0.1]', '(0.1-0.2]', '(0.2-0.3]', '(0.3-0.4]', '(0.4-0.5]', 
        '(0.5-0.6]', '(0.6-0.7]', '(0.7-0.8]', '(0.8-0.9]', '(0.9-1.0]'
    ]
    df['Range'] = pd.cut(
        df[prob_col], bins=bins, labels=labels, include_lowest=True, right=True
        )

    df_group = df.groupby(['Range', target_col], observed=False).count().reset_index()
    table = df_group.pivot(index='Range', columns=target_col, values=prob_col).fillna(0)
    table = table.rename(columns={0: 'nonevents', 1: 'events'})
    table = table[['events', 'nonevents']]
    # Remove the extra index level and axis label
    table.reset_index(inplace=True)
    table.columns.name = None
    table.index = range(1, 11)
    
    table['total_line'] = table['events'] + table['nonevents']
    table['event_rate_line %'] = np.round(table['events'] / table['total_line'], 4)*100
    table['nonevents_rate_line %'] = np.round(table['nonevents'] / table['total_line'], 4)*100
    
    table['eventos/nonevents'] = np.round(table['events'] / table['nonevents'], 4)*100
    return table

import matplotlib.pyplot as plt

def plot_stacked_bar(
    df: pd.DataFrame, 
    target_col: str, 
    prob_col: str
) -> None:
    """
    Plota um gráfico de barras empilhadas mostrando o percentual de eventos e não eventos por faixa de probabilidade.
    Adiciona uma linha tracejada para o percentual de eventos/não eventos em uma segunda escala de eixo y.

    Args:
        df (pd.DataFrame): DataFrame contendo as colunas de probabilidade e target.
        target_col (str): Nome da coluna target.
        prob_col (str): Nome da coluna de probabilidade.
    """
    table = line_event_distribution(df=df, target_col=target_col, prob_col=prob_col)
    # Configurar os dados para o gráfico
    faixas = table['Range']
    event_rate = table['event_rate_line %']
    nonevent_rate = table['nonevents_rate_line %']
    evento_nonevent_ratio = table['eventos/nonevents']

    # Criar a figura e os eixos do gráfico
    fig, ax1 = plt.subplots(figsize=(8, 4))

    # Plotar o gráfico de barras empilhadas no eixo primário
    bar_width = 0.5
    index = range(len(faixas))

    bars1 = ax1.bar(index, event_rate, bar_width, label='Eventos %', color='red', edgecolor='black')
    bars2 = ax1.bar(index, nonevent_rate, bar_width, bottom=event_rate, label='Não eventos %', color='blue', edgecolor='black')

    # Adicionar a segunda escala de eixo y para a linha tracejada
    ax2 = ax1.twinx()
    ax2.plot(index, evento_nonevent_ratio, color='green', linestyle='--', marker='o', label='Evento/Não evento %')

    # Definir rótulos e título
    ax1.set_xlabel('Faixa de probabilidade')
    ax1.set_ylabel('Percentual')
    ax1.set_title('Distribuição empilhada de eventos e não eventos por faixa')
    ax1.set_xticks(index)
    ax1.set_xticklabels(faixas, rotation=45)

    # Ajustar o eixo y para exibir percentuais e limitar entre 0 e 100% no eixo primário
    ax1.set_ylim(0, 100)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}%'))

    # Ajustar o eixo y secundário para a linha tracejada
    ax2.set_ylabel('Evento/Não evento %')
    ax2.set_ylim(0, max(evento_nonevent_ratio) * 1.1)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}%'))

    # Adicionar legendas
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    ax1.grid(True)
    plt.tight_layout()
    plt.show()

import pandas as pd
import numpy as np

def population_stability_index(
    reference_df: pd.DataFrame, 
    monitored_df: pd.DataFrame,
    reference_prob: str,
    monitored_prob: str,
    epsilon: float = 1e-10
) -> pd.DataFrame:
    """
    Calculates the Population Stability Index (PSI) between two samples.
    Em português Indice de Estabilidade Populacional (IEP).


    Args:
        reference_df (pd.DataFrame): DataFrame containing the reference sample.
        monitored_df (pd.DataFrame): DataFrame containing the monitored sample.
        reference_prob (str): Column name for probabilities in the reference sample.
        monitored_prob (str): Column name for probabilities in the monitored sample.
        epsilon (float): Pequeno valor adicionado para evitar log(0).

    Returns:
        Tuple[pd.DataFrame, float]: DataFrame with PSI calculations for each bin and the total PSI value.
        
    Reference values:
    PSI < 0.1 — No change. You can continue using existing model.
    PSI >=0.1 and PSI< 0.25 — Slight change is required.
    PSI >=0.25 — Significant change is required. Ideally, you should not use this model anymore, retraining is required.
    
    Reference: https://www.listendata.com/2015/05/population-stability-index.html
    Reference: https://medium.com/@parthaps77/population-stability-index-psi-and-characteristic-stability-index-csi-in-machine-learning-6312bc52159d
    """
    
    # Define the probability range intervals
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = [
        '[0-0.1]', '(0.1-0.2]', '(0.2-0.3]', '(0.3-0.4]', '(0.4-0.5]', 
        '(0.5-0.6]', '(0.6-0.7]', '(0.7-0.8]', '(0.8-0.9]', '(0.9-1.0]'
    ]
    
    # Create DataFrames to hold the counts
    reference_df['Reference_Range'] = pd.cut(reference_df[reference_prob], bins=bins, labels=labels, include_lowest=True, right=True)
    reference_group = reference_df['Reference_Range'].value_counts().reset_index()
    reference_group.columns = ['Reference_Range', 'Reference_Count']
    reference_group = reference_group.sort_values(by='Reference_Range')
    reference_group['Reference_Perc'] = reference_group['Reference_Count'] / reference_df.shape[0] + epsilon

    monitored_df['Monitored_Range'] = pd.cut(monitored_df[monitored_prob], bins=bins, labels=labels, include_lowest=True, right=True)
    monitored_group = monitored_df['Monitored_Range'].value_counts().reset_index()
    monitored_group.columns = ['Monitored_Range', 'Monitored_Count']
    monitored_group = monitored_group.sort_values(by='Monitored_Range')
    monitored_group['Monitored_Perc'] = monitored_group['Monitored_Count'] / monitored_df.shape[0] + epsilon

    # Calculate the PSI for each bin
    df_psi = pd.merge(reference_group, monitored_group, left_on='Reference_Range', right_on='Monitored_Range')
    df_psi.rename(columns={'Reference_Range': 'Range'}, inplace=True)
    df_psi.drop(columns=['Monitored_Range'], inplace=True)
    df_psi['PSI'] = (df_psi['Monitored_Perc'] - df_psi['Reference_Perc']) * np.log(df_psi['Monitored_Perc'] / df_psi['Reference_Perc'])
    df_psi['PSI'] = df_psi['PSI'].round(8)
    psi_value = df_psi['PSI'].sum().round(8)
    
    return df_psi, psi_value

import pandas as pd
from typing import List

def characteristic_stability_index(
    reference_df: pd.DataFrame, 
    monitored_df: pd.DataFrame, 
    feature_columns: List[str]
) -> pd.DataFrame:
    """
    Calculates the Characteristic Stability Index (CSI) for specified feature columns using the Population Stability Index (PSI) function.

    Args:
        reference_df (pd.DataFrame): DataFrame containing the reference sample.
        monitored_df (pd.DataFrame): DataFrame containing the monitored sample.
        feature_columns (List[str]): List of column names for the features to calculate CSI.

    Returns:
        pd.DataFrame: DataFrame with CSI values for each feature.
    """
    csi_values = {}
    for column in feature_columns:
        df_psi, csi_value = population_stability_index(
            reference_df=reference_df, 
            monitored_df=monitored_df, 
            reference_prob=column, 
            monitored_prob=column
        )
        csi_values[column] = csi_value
    csi_df = pd.DataFrame(list(csi_values.items()), columns=['Variable', 'CSI'])
    return csi_df
    
#######################################################################################

# Versão Original Ajustada
import pandas as pd 
import numpy as np
from typing import Optional

def ks(data: pd.DataFrame, target: str, prob: str) -> pd.DataFrame:
    """
    Calcula a tabela KS (Kolmogorov-Smirnov) e a métrica KS a partir de um DataFrame com colunas de probabilidade e target.

    Args:
    data (pd.DataFrame): DataFrame contendo as colunas de probabilidade e target.
    target (str): Nome da coluna target.
    prob (str): Nome da coluna de probabilidade.

    Returns:
    pd.DataFrame: DataFrame contendo as estatísticas KS e a métrica KS.
    """

    # Calcular o inverso do target
    data['target0'] = 1 - data[target]
    
    # Criar os buckets (decis)
    data['bucket'] = pd.qcut(data[prob], 10, duplicates='drop')
    grouped = data.groupby('bucket', as_index=False)
    
    # Criar o DataFrame para armazenar as estatísticas
    kstable = pd.DataFrame()
    kstable['min_prob'] = grouped.min()[prob]
    kstable['max_prob'] = grouped.max()[prob]
    kstable['events'] = grouped.sum()[target]
    kstable['nonevents'] = grouped.sum()['target0']
    
    # Ordenar pelo menor valor de probabilidade
    kstable = kstable.sort_values(by="min_prob", ascending=False).reset_index(drop=True)
    
    # Calcular as taxas de eventos e não eventos
    kstable['event_rate'] = kstable['events'] / data[target].sum()
    kstable['nonevent_rate'] = kstable['nonevents'] / data['target0'].sum()
    
    # Calcular as taxas acumuladas de eventos e não eventos
    kstable['cum_eventrate'] = kstable['event_rate'].cumsum()
    kstable['cum_noneventrate'] = kstable['nonevent_rate'].cumsum()
    
    # Calcular a métrica KS
    kstable['KS'] = np.round(kstable['cum_eventrate'] - kstable['cum_noneventrate'], 3) * 100
    
    # Formatar as colunas para porcentagens
    kstable['event_rate'] = kstable['event_rate'].apply('{0:.2%}'.format)
    kstable['nonevent_rate'] = kstable['nonevent_rate'].apply('{0:.2%}'.format)
    kstable['cum_eventrate'] = kstable['cum_eventrate'].apply('{0:.2%}'.format)
    kstable['cum_noneventrate'] = kstable['cum_noneventrate'].apply('{0:.2%}'.format)
    
    # Ajustar o índice para Decis
    kstable.index = range(1, 11)
    kstable.index.rename('Decile', inplace=True)
    
    # Exibir o DataFrame
    pd.set_option('display.max_columns', 9)
    print(kstable)
    
    # Exibir a métrica KS
    ks_value = max(kstable['KS'])
    ks_decile = kstable.index[kstable['KS'] == ks_value][0]
    from colorama import Fore
    print(Fore.RED + f"KS is {ks_value}% at decile {ks_decile}")
    
    return kstable

import seaborn as sns
import matplotlib.pyplot as plt

def plot_event_distribution_seaborn(table: pd.DataFrame):
    """
    Plota um gráfico de barras mostrando a distribuição de eventos e não eventos por faixa de probabilidade usando seaborn.

    Args:
        table (pd.DataFrame): DataFrame contendo a distribuição de eventos e não eventos por faixa.
    """
    # Configurar os dados para o gráfico
    table_melted = table.melt(id_vars='Range', value_vars=['event_rate %', 'nonevent_rate %'], 
                              var_name='Tipo', value_name='Percentual')

    # Criar o gráfico de barras
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(x='Range', y='Percentual', hue='Tipo', data=table_melted, palette=['red', 'blue'])

    # Adicionar a grade ao gráfico
    ax.grid(True)

    # Definir rótulos e título
    plt.xlabel('Faixa de probabilidade')
    plt.ylabel('Percentual')
    plt.title('Distribuição de eventos e não eventos por faixa de probabilidade')

    # Ajustar layout e exibir o gráfico
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Exemplo de uso
# table = calculate_events_by_range(test_data_experimento, 'Survived', 'y_proba_forest_clf')
# plot_event_distribution_seaborn(table)
