import numpy as np
import pandas as pd


def dataset_load(df):
    temp = pd.read_csv(df, low_memory=False)
    temp.info()

    # df2 = temp[ (temp['city_or_county'].isna()) | (temp['n_participants'].isna())]
    df2 = temp[
        (temp['latitude'] >= 0) & (temp['latitude'] <= 90) & (temp['longitude'] <= 0) & (temp['longitude'] >= -180)]
    df2 = df2.drop_duplicates(subset=['longitude', 'latitude', 'date', 'state', 'city_or_county', 'n_participants'],
                              keep='first')
    df.describe()
    df.corr()
    # df2 = df2[df2['n_participants'] == (df2['n_males'] + df2['n_females'])]
    print(df2)


# Converting non-integer values to integers and handle missing values, then check for age range
def clean_age(x):
    try:
        converted = int(x)
    except (ValueError, TypeError):
        return pd.NA  # Set a default value for non-convertible or missing values
    if (converted < 0) or (converted > 115):
        return pd.NA
    return converted


def to_int(x):
    try:
        converted = int(x)
    except (ValueError, TypeError):
        return pd.NA  # Set a default value for non-convertible or missing values
    return converted


def non_negative_int(x):
    try:
        converted = int(x)
    except (ValueError, TypeError):
        return pd.NA  # Set a default value for non-convertible or missing values
    if converted < 0:
        return pd.NA
    return converted


def sturges_bins(dataframe, column):
    N = dataframe[column].dropna().count()  # Count non-null values in the specified column
    k = int(np.ceil(np.log2(N) + 1))
    return k


def clean_incidents(incidents):
    df = incidents

    incidents.drop_duplicates(subset=['longitude', 'latitude', 'date', 'state', 'city_or_county', 'n_participants'],
                              keep='first')

    # 1. Handling Missing Data
    df.dropna(subset=['date'], inplace=True)

    # 2. Data Type Validation
    # Converting 'date' to datetime data type
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Removing Duplicates
    df.drop_duplicates(inplace=True)

    # Handling age-related columns
    # Convert to int format and handle missing and irrealistic values
    age_columns = ['participant_age1', 'min_age_participants', 'avg_age_participants', 'max_age_participants']

    df[age_columns] = df[age_columns].apply(pd.to_numeric, errors='coerce')
    df[age_columns] = df[age_columns].map(clean_age)

    people_columns = ['n_participants', 'n_killed', 'n_injured', 'n_arrested']
    df[people_columns] = df[people_columns].astype('Int64')

    df['n_participants'] = df['n_participants'].apply(to_int)
    df['n_females'] = df['n_females'].apply(non_negative_int)
    df['n_males'] = df['n_males'].apply(non_negative_int)
    df['n_killed'] = df['n_killed'].apply(to_int)
    df['n_injured'] = df['n_injured'].apply(to_int)
    df['n_arrested'] = df['n_arrested'].apply(to_int)

    age_groups = ['n_participants_child', 'n_participants_teen', 'n_participants_adult']
    df[age_groups] = df[age_groups].apply(pd.to_numeric, errors='coerce')

    df['n_participants_child'] = df['n_participants_child'].apply(non_negative_int)
    df['n_participants_teen'] = df['n_participants_teen'].apply(non_negative_int)
    df['n_participants_adult'] = df['n_participants_adult'].apply(non_negative_int)

    # remove wrong data
    df = df[(df['n_participants'] >= 0) & (df['n_killed'] >= 0) & (df['n_injured'] >= 0) & (df['n_arrested'] >= 0)]
    df = df[(df['n_females'] + df['n_males']) <= df['n_participants']]
    df = df[df['n_killed'] <= df['n_participants']]
    df = df[df['n_injured'] <= df['n_participants']]
    df = df[df['n_arrested'] <= df['n_participants']]
    df = df[df['n_participants_child'] + df['n_participants_teen'] + df['n_participants_adult'] <= df['n_participants']]

    # Standardizing incident characteristics
    incident_characteristics_columns = ['incident_characteristics1', 'incident_characteristics2']
    for column in incident_characteristics_columns:
        df[column] = df[column].str.lower()  # Standardize to lowercase

    # 26. Cleaning 'notes' (example: removing special characters)
    df['notes'] = df['notes'].str.replace(r'[^\w\s]', '')

    df = df[(df['date'].dt.year >= 2010) & (df['date'].dt.year <= 2020)]

    # filter out the latitude and longitude that are not in the US
    df = df[(df['latitude'] >= 20) & (df['latitude'] <= 50) & (df['longitude'] <= -50) & (df['longitude'] >= -125)]

    df = df[df['congressional_district'] <= 435]

    # write year and month to the dataframe from date
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    return df


def den_incidents(year, incidents, population):
    incidents = incidents[incidents['year'] == year]
    population = population[population['Year'] == year]
    state_incidents = incidents['state'].value_counts().rename_axis('State').reset_index(name='counts')
    total_incidents = state_incidents['counts'].sum()
    state_incidents.loc[len(state_incidents)] = ["U.S.", total_incidents]
    merge = pd.merge(state_incidents, population, left_on='State', right_on='Description', how='inner')
    df_den_incidents = pd.DataFrame(columns=['State', 'Incident Density per 100k'])
    df_den_incidents['State'] = merge['State']
    df_den_incidents['Incident Density per 100k'] = (merge['counts'] / merge['Total Population']) * 100000
    df_den_incidents.sort_values('Incident Density per 100k', inplace=True, ascending=False)
    df_den_incidents.reset_index(inplace=True, drop=True)
    print("Density of incidents in the year " + str(year))
    print(df_den_incidents)
    print("------------------------------------------")
    return df_den_incidents
    # df_den_incidents.plot(kind='bar', title='title')
    # plt.show()
    # return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pd.set_option('display.max_rows', 50)
    pd.set_option('display.max_columns', None)
    dataset_load("data/incidents.csv")


def group_parties(x, parties=['Republican', 'Democrat', 'Other']):
    if x in parties:
        return x
    else:
        return 'Other'
