import pandas as pd
import numpy as np

def generate_car_matrix(data):
    df = pd.read_csv(data)
    matrix_df = df.pivot(index='id_1', columns='id_2', values='car').fillna(0)
    for idx in matrix_df.index:
        matrix_df.at[idx, idx] = 0

    return matrix_df

data = 'dataset-1.csv'
result_matrix = generate_car_matrix(file_path)
result_matrix



def get_type_count(df):
    conditions = [
        (df['car'] <= 15),
        (df['car'] > 15) & (df['car'] <= 25),
        (df['car'] > 25) ]
    choices = ['low', 'medium', 'high']
    df['car_type'] = np.select(conditions, choices, default='unknown')
    type_counts = df['car_type'].value_counts().to_dict()
    sorted_type_counts = {k: v for k, v in sorted(type_counts.items())}

    return sorted_type_counts

result_type_counts = get_type_count(df)
result_type_counts



def get_bus_indexes(df):
    bus_mean = df['bus'].mean()
    bus_indexes = df[df['bus'] > 2 * bus_mean].index.tolist()
    sorted_bus_indexes = sorted(bus_indexes)

    return sorted_bus_indexes

result_bus_indexes = get_bus_indexes(df)
print(result_bus_indexes)



import pandas as pd

def filter_routes(df):
    route_avg_truck = df.groupby('route')['truck'].mean()
    selected_routes = route_avg_truck[route_avg_truck > 7].index.tolist()
    sorted_selected_routes = sorted(selected_routes)

    return sorted_selected_routes

result_routes = filter_routes(df)
print(result_routes)




def multiply_matrix(result_matrix):
    modified_matrix = result_matrix.applymap(lambda x: x * 0.75 if x > 20 else x * 1.25)
    modified_matrix = modified_matrix.round(1)

    return modified_matrix

result_matrix = generate_car_matrix('dataset-1.csv')
modified_result_matrix = multiply_matrix(result_matrix)
modified_result_matrix





