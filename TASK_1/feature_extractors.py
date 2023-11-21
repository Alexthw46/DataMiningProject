import pandas as pd


# calculates the ratio of participants in each incident to the total number of participants in all incidents for that
# specific month and year.
def ratio_par_to_total(df):
    total_participants = pd.DataFrame()
    years = [2014, 2015, 2016, 2017, 2018]
    months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    for x in years:
        incidents = df[df['year'] == x]
        for y in months:
            month_incidents = incidents[incidents['month'] == y]
            temp = month_incidents.groupby('state', as_index=False).sum()
            total_involved = temp[['state', 'n_participants']]
            total_involved['year'] = x
            total_involved['month'] = y
            total_participants = pd.concat([total_participants, total_involved])
    total_participants = total_participants.rename(columns={'n_participants': 'tot_participants'})
    df = pd.merge(df, total_participants, how='inner')
    df['par_to_tot'] = df['n_participants'] / df['tot_participants']
    df = df.drop(columns=['tot_participants'])
    return df


# calculates the ratio of killed to participants in each incident
def ktp(df):
    df['kil_to_p'] = df['n_killed'] / df['n_participants']
    return df


# calculates the ratio of people killed to the poverty percentage of the state in that year
def crime_to_poverty(df, df2):
    temp = pd.merge(df, df2, left_on=['state', 'year'], right_on=['state', 'year'], how='inner')
    temp['kills_to_pov'] = temp['n_killed'] / temp['povertyPercentage']
    temp = temp.drop(columns=['povertyPercentage'])
    df = pd.merge(df, temp, how='inner')
    return df


# calculates ratio of average age of participants in each incident to the average age of participants in all incidents
# for that specific state
def average_age(df):
    avg_age = df.groupby('state')['avg_age_participants'].transform('mean')
    df['avg_age'] = avg_age
    df['age_to_average'] = df['avg_age_participants'] / df['avg_age']
    df = df.drop(columns=['avg_age'])
    return df


# calculates ratio of number of participants in each incident to the average number of participants in all incidents
# for that specific state
def average_participants(df):
    avg_participants = df.groupby('state')['n_participants'].transform('mean')
    df['avg_participants'] = avg_participants
    df['par_to_average'] = df['n_participants'] / df['avg_participants']
    return df


# sum of killed and injured in each incident
def involved(df):
    df['n_involved'] = df['n_killed'] + df['n_injured']
    return df


def minors_percentage(df, df4):
    merged_df = pd.merge(df, df4, on=['state', 'year'], how='inner')
    merged_df['minors_to_pop'] = ((merged_df['n_participants_child'] + merged_df['n_participants_teen']) / merged_df[
        'Population Under 18']) * 100000
    df = merged_df.drop(
        columns=['Total Population', 'Population Under 18', 'Population 18-54', 'Population 55+', 'Male Population',
                 'Female Population'])
    return df


def par_to_pop(df, df4):
    temp = pd.merge(df, df4, left_on=['state', 'year'], right_on=['state', 'year'], how='inner')
    temp['par_to_pop'] = (temp['n_participants'] / temp['Total Population']) * 100000
    temp = temp.drop(
        columns=['Total Population', 'Population Under 18', 'Population 18-54',
                 'Population 55+', 'Male Population', 'Female Population'])
    df = pd.merge(df, temp, how='inner')
    return df


# ratio of males participant to total participants
def mtp(df):
    df['man_to_p'] = df['n_males'] / df['n_participants']
    return df


# ratio of females participant to total participants
def ftp(df):
    df['fem_to_p'] = df['n_females'] / df['n_participants']
    return df


# ratio of arrested participant to total participants
def atp(df):
    df['arr_to_p'] = df['n_arrested'] / df['n_participants']
    return df


# ratio of unharmed participant to total participants
def utp(df):
    df['unh_to_p'] = df['n_unharmed'] / df['n_participants']
    return df


# ratio of injured participant to total participants
def itp(df):
    df['inj_to_p'] = df['n_injured'] / df['n_participants']
    return df


# sum of participants under 20
def num_minors(df):
    df['n_minors'] = df['n_participants_child'] + df['n_participants_teen']
    return df


# ratio between participants under 20 and male participants
def teen_to_m(df):
    df['teen_to_m'] = df['n_participants_teen'] / df['n_males'].replace(0, 1)
    df['teen_to_m'].fillna(0, inplace=True)
    return df


# ratio between participants under 20 and female participants
def teen_to_f(df):
    df['teen_to_f'] = df['n_participants_teen'] / df['n_females'].replace(0, 1)
    df['teen_to_f'].fillna(0, inplace=True)
    return df


def incidents_to_win(df, df3):
    incidents_win = pd.DataFrame()
    years = [2014, 2015, 2016, 2017, 2018]
    for x in years:
        incidents = df[df['year'] == x]
        state_incidents = incidents['state'].value_counts().rename_axis('state').reset_index(name='counts')
        state_incidents['year'] = x
        year_results = df3
        if x % 2 == 0:
            year_results = year_results[year_results['year'] == x - 2]
            year_results['year'] = x
        else:
            year_results = year_results[year_results['year'] == x - 1]
            year_results['year'] = x
        year_results['win_percentage'] = (year_results['candidatevotes'] / year_results['totalvotes']) * 100
        year_results = year_results.drop(columns=['congressional_district', 'party', 'candidatevotes', 'totalvotes'])
        year_results = year_results.groupby('state', as_index=False).mean()
        temp = pd.merge(state_incidents, year_results)
        incidents_win = pd.concat([incidents_win, temp])
    incidents_win['incidents_to_win'] = incidents_win['counts'] / incidents_win['win_percentage']
    incidents_win = incidents_win.drop(columns=['counts', 'win_percentage'])
    df = pd.merge(df, incidents_win, how='outer')
    return df


def involved_to_voters(df, df3):
    involved_voters = pd.DataFrame()
    years = [2014, 2015, 2016, 2017, 2018]
    for x in years:
        incidents = df[df['year'] == x]
        temp = incidents.groupby('state', as_index=False).sum()
        temp['n_involved'] = temp['n_killed'] + temp['n_injured']
        total_involved = temp[['state', 'n_involved']]
        total_involved['year'] = x
        year_results = df3
        if x % 2 == 0:
            year_results = year_results[year_results['year'] == x - 2]
            year_results['year'] = x
        else:
            year_results = year_results[year_results['year'] == x - 1]
            year_results['year'] = x
        year_results = year_results.groupby('state', as_index=False).sum()
        year_results = year_results.drop(columns=['congressional_district', 'party', 'year', 'totalvotes'])
        temp = pd.merge(total_involved, year_results)
        involved_voters = pd.concat([involved_voters, temp])
    involved_voters['involved_to_voters'] = (involved_voters['n_involved'] / involved_voters['candidatevotes']) * 100000
    involved_voters = involved_voters.drop(columns=['n_involved', 'candidatevotes'])
    df = pd.merge(df, involved_voters, how='outer')
    return df


def kill_to_gun(df, df5):
    kills_to_gun = pd.DataFrame()
    months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    years = [2014, 2015, 2016]
    for x in years:
        incidents = df[df['year'] == x]
        for y in months:
            temp = incidents[incidents['month'] == y]
            temp = temp.groupby('state', as_index=False).sum()
            state_kills = temp[['state', 'n_killed']]
            state_kills['year'] = x
            state_kills['month'] = y
            gun_ownership = df5[df5['year'] == x]
            temp = pd.merge(state_kills, gun_ownership, how='inner')
            kills_to_gun = pd.concat([kills_to_gun, temp])
    kills_to_gun['kill_to_gun'] = kills_to_gun['n_killed'] / (kills_to_gun['HFR'] * 100)
    kills_to_gun = kills_to_gun.drop(columns=['n_killed', 'HFR', 'universl', 'permit'])
    df = pd.merge(df, kills_to_gun, how='outer')
    return df


def poverty_to_gun(df, df2, df5):
    temp = df5.rename(columns={'Year': 'year', 'STATE': 'state'})
    merge = pd.merge(df2, temp, how='inner')
    merge['poverty_to gun'] = merge['povertyPercentage'] / (merge['HFR'] * 100)
    merge = merge.drop(columns=['povertyPercentage', 'HFR', 'universl', 'permit'])
    df = pd.merge(df, merge, how='outer')
    return df
