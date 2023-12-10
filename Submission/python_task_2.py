
import pandas as pd
import networkx as nx
import numpy as np

def calculate_distance_matrix(file_path, replace_inf=np.nan, replace_nan=np.nan):
    df = pd.read_csv(file_path)
    G = nx.DiGraph()
    for _, row in df.iterrows():
         G.add_edge(row['id_start'], row['id_end'], weight=row['distance'])

    distance_matrix = nx.floyd_warshall_numpy(G, weight='weight')

    distance_matrix = np.where(np.isinf(distance_matrix), replace_inf, distance_matrix)

    distance_matrix = np.where(np.isnan(distance_matrix), replace_nan, distance_matrix)

    distance_df = pd.DataFrame(distance_matrix, index=G.nodes, columns=G.nodes)

    return distance_df

file_path = 'dataset-3.csv'
distance_matrix = calculate_distance_matrix(file_path, replace_inf=np.inf, replace_nan=np.nan)
distance_matrix

import pandas as pd

def unroll_distance_matrix(distance_matrix):
    stacked_series = distance_matrix.stack()
    stacked_df = stacked_series.reset_index()
    stacked_df.columns = ['id_start', 'id_end', 'distance']
    stacked_df = stacked_df[stacked_df['id_start'] != stacked_df['id_end']]

    return stacked_df

result_unrolled = unroll_distance_matrix(distance_matrix)
result_unrolled


def find_ids_within_ten_percentage_threshold(df, reference_value):
    subset_df = df[df['id_start'] == reference_value]
    print("Unique id_start values:", df['id_start'].unique())
    if subset_df.empty:
        print(f"No rows found for id_start = {reference_value}")
        return []

    average_distance = subset_df['distance'].mean()
    print("Average distance:", average_distance)

    threshold_range = 0.1 * average_distance
    print("Threshold range:", threshold_range)

    result_ids = df[(df['id_start'] != reference_value) &
                    (df['distance'] >= (average_distance - threshold_range)) &
                    (df['distance'] <= (average_distance + threshold_range))]['id_start'].unique()

    result_ids.sort()

    return result_ids

reference_value = 123  # Replace with the desired reference value
result_within_threshold = find_ids_within_ten_percentage_threshold(result_unrolled, reference_value)
result_within_threshold


def calculate_toll_rate(df):
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}
    for vehicle_type, rate_coefficient in rate_coefficients.items():
        column_name = f'{vehicle_type}_toll'
        df[column_name] = df['distance'] * rate_coefficient

    return df

result_with_toll_rates = calculate_toll_rate(result_unrolled)
result_with_toll_rates





