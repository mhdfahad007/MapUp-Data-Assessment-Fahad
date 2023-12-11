import pandas as pd


def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Write your logic here

    def calculate_distance_matrix(df):
    # Read the CSV file into a DataFrame
    data = pd.read_csv('dataset-3.csv')

    # Create a dictionary to store distances between toll locations
    distances = {}

    # Iterate through the data to populate the distances dictionary
    for index, row in data.iterrows():
        loc1, loc2, distance = row['id_start'], row['id_end'], row['distance']

        # Store distances bidirectionally
        distances[(loc1, loc2)] = distance
        distances[(loc2, loc1)] = distance

    # Get unique toll locations
    locations = sorted(list(set(data['id_start'].unique()) | set(data['id_end'].unique())))

    # Initialize the distance matrix with zeros
    distance_matrix = pd.DataFrame(np.zeros((len(locations), len(locations))), index=locations, columns=locations)

    # Populate the distance matrix with cumulative distances
    for i in range(len(locations)):
        for j in range(len(locations)):
            if i != j:
                loc1, loc2 = locations[i], locations[j]
                # Check if distance between loc1 and loc2 is known
                if (loc1, loc2) in distances:
                    distance_matrix.loc[loc1, loc2] = distances[(loc1, loc2)]
                else:
                    # Find intermediate locations to calculate cumulative distances
                    intermediate_locs = [k for k in locations if (loc1, k) in distances and (k, loc2) in distances]
                    if intermediate_locs:
                        # Calculate cumulative distance via intermediate locations
                        cumulative_distance = sum(distances[(loc1, k)] + distances[(k, loc2)] for k in intermediate_locs)
                        distance_matrix.loc[loc1, loc2] = cumulative_distance

    # Fill diagonal values with zeros
    np.fill_diagonal(distance_matrix.values, 0)

    return distance_matrix


    resulting_matrix = calculate_distance_matrix('dataset-3.csv')
  print(resulting_matrix)


    return df


def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Write your logic here

    import pandas as pd

    def unroll_distance_matrix(df):
    # Extract the upper triangular matrix excluding the diagonal
    upper_triangular = df.where(np.triu(np.ones(df.shape), k=1).astype(np.bool))

    # Reset index to convert the DataFrame to long format
    unrolled = upper_triangular.stack().reset_index()
    unrolled.columns = ['id_start', 'id_end', 'distance']

    return unrolled

  # assuming 'dataset-13.csv' is the DataFrame obtained from the previous calculation
   
  unrolled_distances = unroll_distance_matrix('dataset-13.csv')
    print(unrolled_distances)

    return df


def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here

    def find_ids_within_ten_percentage_threshold(df, reference_id):
    # Filter DataFrame for the reference_id in id_start column
    reference_df = df[df['id_start'] == reference_value]

    # Calculate the average distance for the reference value
    reference_avg_distance = reference_df['distance'].mean()

    # Calculate the threshold values (within 10% range)
    threshold = 0.1 * reference_avg_distance
    lower_bound = reference_avg_distance - threshold
    upper_bound = reference_avg_distance + threshold

    # Filter ids within the 10% threshold
    filtered_ids = df[(df['id_start'] != reference_value) & 
                      (df['distance'] >= lower_bound) & 
                      (df['distance'] <= upper_bound)]['id_start'].unique()

    # Sort the filtered IDs
    filtered_ids = sorted(filtered_ids)

    return filtered_ids

# Assuming 'unrolled_distances' is the DataFrame obtained from the previous function

 reference_value = 5  # Asuming reference value
 result_ids_within_threshold = find_ids_within_ten_percentage_threshold(unrolled_distances, reference_value)
   print(result_ids_within_threshold)


    return df


def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Wrie your logic here

    def calculate_toll_rate(df):
    # Define rate coefficients for different vehicle types
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    # Calculate toll rates for each vehicle type
    for vehicle, rate in rate_coefficients.items():
        df[vehicle] = df['distance'] * rate

    return df

# Assuming 'unrolled_distances' is the DataFrame obtained from the previous question

result_with_toll_rates = calculate_toll_rate(unrolled_distances)
print(result_with_toll_rates)


    return df


def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here


    import datetime

def calculate_time_based_toll_rates(df):
    # Create a DataFrame to cover the time intervals for each day of the week
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    time_intervals = [
        ('00:00:00', '10:00:00', 0.8),
        ('10:00:00', '18:00:00', 1.2),
        ('18:00:00', '23:59:59', 0.8)
    ]
    weekend_discount = 0.7

    # Generate combinations of time intervals for each day
    time_intervals_weekdays = []
    for day in days_of_week[:5]:  # Weekdays
        for interval in time_intervals:
            start_time = datetime.datetime.strptime(interval[0], '%H:%M:%S').time()
            end_time = datetime.datetime.strptime(interval[1], '%H:%M:%S').time()
            time_intervals_weekdays.append((day, start_time, day, end_time, interval[2]))

    time_intervals_weekends = []
    for day in days_of_week[5:]:  # Weekends
        for interval in [(datetime.time(0, 0, 0), datetime.time(23, 59, 59))]:
            time_intervals_weekends.append((day, interval[0], day, interval[1], weekend_discount))

    # Concatenate intervals for both weekdays and weekends
    all_time_intervals = time_intervals_weekdays + time_intervals_weekends

    # Create a DataFrame to store time-based toll rates
    time_based_toll_rates = pd.DataFrame(columns=['id_start', 'id_end', 'start_day', 'start_time', 'end_day', 'end_time'] + list(df.columns[5:]))

    # Iterate over unique combinations of id_start and id_end
    for (id_start, id_end), group in df.groupby(['id_start', 'id_end']):
        for interval in all_time_intervals:
            start_day, start_time, end_day, end_time, discount_factor = interval
            temp_row = [id_start, id_end, start_day, start_time, end_day, end_time]

            # Apply discount factor to the vehicle columns based on the time interval
            temp_row += list(group.iloc[:, 5:].apply(lambda x: x * discount_factor))

            time_based_toll_rates = time_based_toll_rates.append(pd.Series(temp_row, index=time_based_toll_rates.columns), ignore_index=True)

    return time_based_toll_rates

# Assuming 'result_ids_within_threshold' is the DataFrame obtained from the previous question

time_based_rates = calculate_time_based_toll_rates(result_ids_within_threshold)
print(time_based_rates)


    return df
