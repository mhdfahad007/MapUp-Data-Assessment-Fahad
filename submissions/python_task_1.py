import pandas as pd


def generate_car_matrix(df)->pd.DataFrame:
    """
    Creates a DataFrame  for id combinations.
    
    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Matrix generated with 'car' values, 
                          where 'id_1' and 'id_2' are used as indices and columns respectively.
    """
    # Write your logic here

#logic that takes the `dataset-1.csv` as a DataFrame
dataframe = pd.read_csv('dataset-1.csv')

# Pivot the DataFrame to create the desired matrix
df = dataframe.pivot(index='id_1', columns='id_2', values='car')

#for 0 diagonal values
df[[range(len(df))], [range(len(df))]] = 0

    return df


def get_type_count(df)->dict:
    """
    Categorizes 'car' values into types and returns a dictionary of counts.

    Args:
        df (pandas.DataFrame)

    Returns:
        dict: A dictionary with car types as keys and their counts as values.
    """
    # Write your logic here

    
    
    #Adding new categorical column `car_type` based on values of the column `car`:
     
    def get_type_count(df):
    #Defining  bins and labels for car types

    bins= [float('-inf'),15,25, float('inf')]
    labels=['low', 'medium','high']

    # create car_type column
    df['car_type']= pd.cut(df['car'], bins=bins, labels=labels, right=False)

    # Count occurrences of each car_type category and return as a sorted dictionary
    return(
     
    df['car_type'].value_counts().sort_index().to_dict()

    )

    # Read dataset-1.csv into a DataFrame
    dataframe= pd.read_csv('dataset-1.csv')
     
    # call the function and store result 
    result=get_type_count(dataframe)
    print(result)  # Display the count of occurrences for each car_type category
    
    return dict()


def get_bus_indexes(df)->list:
    """
    Returns the indexes where the 'bus' values are greater than twice the mean.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of indexes where 'bus' values exceed twice the mean.
    """
    # Write your logic here

    def get_bus_indexes(df)

    # calculate the mean of the bus column
 
    bus_mean=df['bus'].mean()

    # Indicies where bus values are greater than twice the mean value of the `bus` column in the DataFrame.

    bus_indexes = df[df['bus'] > 2 * bus_mean].index.tolist()
    
    #sort the Indicies in ascending order

    bus_indexes.sort()

  # read the dataset-1.csv
   dataframe=pd.read_csv('dataset-1.csv')

   #call the function and store the result
   result=get_bus_indexes(dataframe)

   print(result) # display the indicies where bus values are greater 

    

    return list()


def filter_routes(df)->list:
    """
    Filters and returns routes with average 'truck' values greater than 7.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of route names with average 'truck' values greater than 7.
    """
    # Write your logic here

    def filter_routes(df):
    # Group the data by 'route' and calculate the mean of 'truck' values for each route
    route_means=df.groupby('route')['truck'].mean()

    # Filter routes where the average of 'truck' values is greater than 7
    filtered_routes = route_means[route_means > 7].index.tolist()

    #sort the list of routes in Ascending order
    filtered_routes.sort()
    
    return filtered_routes

    # read the dataset-1.csv
   dataframe=pd.read_csv('dataset-1.csv')

   # Call the function and store the result
     result = filter_routes(dataframe)

     print(result)

    return list()


def multiply_matrix(matrix)->pd.DataFrame:
    """
    Multiplies matrix values with custom conditions.

    Args:
        matrix (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Modified matrix with values multiplied based on custom conditions.
    """
    # Write your logic here

    def multiply_matrix(matrix)
    #Modify values based on condition 
    modified_data= matrix.applymap(lambda x: x * 0.75 if x > 20 else x * 1.25)

    # Round the modified values to 1 decimal place
    modified_data_rounded = modified_data.round(1)
       
    return modified_data_rounded

    # Assuming 'matric2' contains the DataFrame from the question 1
    # Call the function
    modified_result = multiply_matrix(matric2)
    print(modified_result)


    return matrix


def time_check(df)->pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here

    def time_check(df):
    # Combine startDay and startTime to create a start datetime column
    df['start'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])

    # Combine endDay and endTime to create an end datetime column
    df['end'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])

    # Calculate duration in seconds between start and end timestamps
    df['duration'] = (df['end'] - df['start']).dt.total_seconds()

    # Group by id, id_2 and check if each pair spans a full 24-hour period for all 7 days
    completeness_check = dataframe.groupby(['id', 'id_2']).apply(
        lambda x: (
            (x['duration'].sum() >= 604800) and  # 7 days in seconds
            (x['start'].min().time() == pd.Timestamp('00:00:00').time()) and
            (x['end'].max().time() == pd.Timestamp('23:59:59').time())
        )
    )
    return completeness_check

    # Load the CSV into a DataFrame
    data = pd.read_csv('dataset-2.csv')

   completeness_result = time_check(data)
   print(completeness_result)


    return pd.Series()
