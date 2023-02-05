import fastf1 as ff1
import numpy as np
import pandas as pd
import dataclasses
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

ff1.Cache.enable_cache('./fastf1_cache')

rounds = [
    (2020, 'Austrian Grand Prix', 'Red Bull Ring'),
    (2020, 'Styrian Grand Prix', 'Red Bull Ring'),
    (2020, 'Hungarian Grand Prix', 'Hungaroring'),
    (2020, 'British Grand Prix', 'Silverstone Circuit'),
    (2020, '70th Anniversary Grand Prix', 'Silverstone Circuit'),
    (2020, 'Spanish Grand Prix', 'Circuit de Barcelona-Catalunya'),
    (2020, 'Belgian Grand Prix', 'Circuit de Spa-Francorchamps'),
    (2020, 'Italian Grand Prix', 'Monza Circuit'),
    (2020, 'Tuscan Grand Prix', 'Mugello Circuit'),
    (2020, 'Russian Grand Prix', 'Sochi Autodrom'),
    (2020, 'Eifel Grand Prix', 'Nürburgring'),
    (2020, 'Portuguese Grand Prix', 'Algarve International Circuit'),
    (2020, 'Emilia Romagna Grand Prix', 'Imola Circuit'),
    (2020, 'Turkish Grand Prix', 'Istanbul Park'),
    (2020, 'Bahrain Grand Prix', 'Bahrain International Circuit'),
    (2020, 'Sakhir Grand Prix', 'Bahrain International Circuit'),
    (2020, 'Abu Dhabi Grand Prix', 'Yas Marina Circuit'),
    # 2021
    (2021, 'Bahrain Grand Prix', 'Bahrain International Circuit'),
    (2021, 'Emilia Romagna Grand Prix', 'Imola Circuit'),
    (2021, 'Portuguese Grand Prix', 'Algarve International Circuit'),
    (2021, 'Spanish Grand Prix', 'Circuit de Barcelona-Catalunya'),
    (2021, 'Monaco Grand Prix', 'Circuit de Monaco'),
    (2021, 'Azerbaijan Grand Prix', 'Baku City Circuit'),
    (2021, 'French Grand Prix', 'Circuit Paul Ricard'),
    (2021, 'Styrian Grand Prix', 'Red Bull Ring'),
    (2021, 'Austrian Grand Prix', 'Red Bull Ring'),
    (2021, 'British Grand Prix', 'Silverstone Circuit'),
    (2021, 'Hungarian Grand Prix', 'Hungaroring'),
    (2021, 'Belgian Grand Prix', 'Circuit de Spa-Francorchamps'),
    (2021, 'Dutch Grand Prix', 'Circuit Zandvoort'),
    (2021, 'Italian Grand Prix', 'Monza Circuit'),
    (2021, 'Russian Grand Prix', 'Sochi Autodrom'),
    (2021, 'Turkish Grand Prix', 'Istanbul Park'),
    (2021, 'United States Grand Prix', 'Circuit of the Americas'),
    (2021, 'Mexico City Grand Prix', 'Autódromo Hermanos Rodríguez'),
    (2021, 'São Paulo Grand Prix', 'Interlagos Circuit'),
    (2021, 'Qatar Grand Prix', 'Losail International Circuit'),
    (2021, 'Saudi Arabian Grand Prix', 'Jeddah Corniche Circuit'),
    (2021, 'Abu Dhabi Grand Prix', 'Yas Marina Circuit'),
    # 2022
    (2022, 'Bahrain Grand Prix', 'Bahrain International Circuit'),
    (2022, 'Saudi Arabian Grand Prix', 'Jeddah Corniche Circuit'),
    (2022, 'Australian Grand Prix', 'Albert Park Circuit'),
    (2022, 'Emilia Romagna Grand Prix', 'Imola Circuit'),
    (2022, 'Miami Grand Prix', 'Miami International Autodrome'),
    (2022, 'Spanish Grand Prix', 'Circuit de Barcelona-Catalunya'),
    (2022, 'Monaco Grand Prix', 'Circuit de Monaco'),
    (2022, 'Azerbaijan Grand Prix', 'Baku City Circuit'),
    (2022, 'Canadian Grand Prix', 'Circuit Gilles Villeneuve'),
    (2022, 'British Grand Prix', 'Silverstone Circuit'),
    (2022, 'Austrian Grand Prix', 'Red Bull Ring'),
    (2022, 'French Grand Prix', 'Circuit Paul Ricard'),
    (2022, 'Hungarian Grand Prix', 'Hungaroring'),
    (2022, 'Belgian Grand Prix', 'Circuit de Spa-Francorchamps'),
    (2022, 'Dutch Grand Prix', 'Circuit Zandvoort'),
    (2022, 'Italian Grand Prix', 'Monza Circuit'),
    (2022, 'Singapore Grand Prix', 'Marina Bay Street Circuit'),
    (2022, 'Japanese Grand Prix', 'Suzuka International Racing Course'),
    (2022, 'United States Grand Prix', 'Circuit of the Americas'),
    (2022, 'Mexico City Grand Prix', 'Autódromo Hermanos Rodríguez'),
    (2022, 'São Paulo Grand Prix', 'Interlagos Circuit'),
    (2022, 'Abu Dhabi Grand Prix', 'Yas Marina Circuit'),
]

def get_training_set_for_rounds(round_num_start, round_num_end):
    training_set = pd.DataFrame()
    
    for round_num in range(round_num_start, round_num_end+1):
        training_set = pd.concat([training_set, get_training_set_for_round(round_num)])

    return training_set

def get_training_set_for_round(round_num):
    round_index = round_num-1
    round = rounds[round_index]
    
    training_set_round_session = ff1.get_session(round[0], round[1], 'FP2')
    traning_set_session_col_names = ['Driver', 'Team', 'LapNumber', 'Compound', 'TyreLife', 'TrackStatus', 'LapTime']
    training_set_session_laps = training_set_round_session.load_laps()
    training_set_session_laps['LapTime'] = training_set_session_laps['LapTime'] / np.timedelta64(1, 's')
    training_set_session_best_laps = training_set_session_laps[(training_set_session_laps['IsPersonalBest'] == True)]
    training_set_session_best_time = training_set_session_best_laps['LapTime'].min()

    print(training_set_session_best_laps)

    training_set_round = training_set_session_best_laps.loc[:, traning_set_session_col_names]

    training_set_round['Year'] = rounds[round_index][0]
    training_set_round['Race'] = rounds[round_index][1]
    training_set_round['Track'] = rounds[round_index][2]

    training_set_round['PracticeLapTimeDelta'] = training_set_round['LapTime'] / training_set_session_best_time
    training_set_round = training_set_round.drop(columns=['LapTime'])

    target_variable_session = ff1.get_session(round[0], round[1], 'Q')
    target_variable_session_laps = target_variable_session.load_laps()
    target_variable_session_laps['LapTime'] = target_variable_session_laps['LapTime'] / np.timedelta64(1, 's')
    target_variable_session_col_names = ['Driver', 'Team', 'LapNumber', 'Compound', 'TyreLife', 'TrackStatus', 'LapTime']
    target_variable_session_best_laps = target_variable_session_laps[(target_variable_session_laps['IsPersonalBest'] == True)]
    target_variable_set_session_best_time = target_variable_session_best_laps['LapTime'].min()
    target_variable_set = target_variable_session_best_laps.loc[:, target_variable_session_col_names]
    target_variable_set['LapTimeDelta'] = target_variable_set['LapTime'] / target_variable_set_session_best_time

    training_set_round['QualifyingLapTimeDelta'] = training_set_round.apply(lambda x: target_variable_set.loc[x['Driver'] == target_variable_set['Driver'], 'LapTimeDelta'].reset_index(drop=True), axis=1)
    
    return training_set_round[['Year','Race','Track','Driver','Team','LapNumber','Compound','TyreLife','TrackStatus','PracticeLapTimeDelta','QualifyingLapTimeDelta']]


training_set = get_training_set_for_rounds(3, 3)

print(training_set.to_string())
