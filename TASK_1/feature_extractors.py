import pandas as pd


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


def crime_to_poverty(df, df2):
    incidents_to_poverty = pd.DataFrame()
    months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    years = [2014, 2015, 2016, 2017, 2018]
    for x in years:
        incidents = df[df['year'] == x]
        for y in months:
            temp = incidents[incidents['month'] == y]
            state_incidents = temp['state'].value_counts().rename_axis('state').reset_index(name='counts')
            poverty_year = df2[df2['year'] == x]
            state_incidents['month'] = y
            temp = pd.merge(state_incidents, poverty_year, left_on='state', right_on='state', how='inner')
            incidents_to_poverty = pd.concat([incidents_to_poverty, temp])
    incidents_to_poverty['crime_ratio'] = incidents_to_poverty['counts'] / incidents_to_poverty['povertyPercentage']
    incidents_to_poverty = incidents_to_poverty.drop(columns=['counts', 'povertyPercentage'])
    df = pd.merge(df, incidents_to_poverty, how='inner')
    return df


def average_age(df):
    avg_age = df.groupby('state', as_index=False)['avg_age_participants'].mean()
    avg_age = avg_age.rename(columns={'avg_age_participants': 'avg_age'})
    df = pd.merge(df, avg_age, how='inner')
    return df


def average_participants(df):
    avg_participants = df.groupby('state', as_index=False)['n_participants'].mean()
    avg_participants = avg_participants.rename(columns={'n_participants': 'avg_participants'})
    df = pd.merge(df, avg_participants, how='inner')
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


def minors_percentage(df, df4):
    minors = pd.DataFrame()
    years = [2014, 2015, 2016, 2017]
    for x in years:
        incidents = df[df['year'] == x]
        temp = incidents.groupby('state', as_index=False).sum()
        temp['n_participants_minors'] = temp['n_participants_child'] + temp['n_participants_teen']
        total_minors = temp[['state', 'n_participants_minors']]
        population_year = df4[df4['year'] == x]
        temp = pd.merge(total_minors, population_year)
        minors = pd.concat([minors, temp])
    minors['minors_percentage'] = minors['n_participants_minors'] / minors['Population Under 18']
    minors = minors.drop(
        columns=['n_participants_minors', 'Total Population', 'Population Under 18', 'Population 18-54',
                 'Population 55+', 'Male Population', 'Female Population'])
    df = pd.merge(df, minors, how='outer')
    return df


def poverty_to_gun(df, df2, df5):
    temp = df5.rename(columns={'Year': 'year', 'STATE': 'state'})
    merge = pd.merge(df2, temp, how='inner')
    merge['poverty_to gun'] = merge['povertyPercentage'] / (merge['HFR'] * 100)
    merge = merge.drop(columns=['povertyPercentage', 'HFR', 'universl', 'permit'])
    df = pd.merge(df, merge, how='outer')
    return df
