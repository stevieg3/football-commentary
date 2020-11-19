import pandas as pd
import numpy as np

FEATURES_WITH_NULLS = [
    'event_type2',
    'player',
    'player2',
    'player_in',
    'player_out',
    'shot_place',
    'shot_outcome',
    'location',
    'bodypart',
    'situation',
    'odd_over',
    'odd_under',
    'odd_bts',
    'odd_bts_n'
]
"""
Features which contain nulls
"""

COLUMNS_TO_DROP = FEATURES_WITH_NULLS + [
    'id_event',
    'link_odsp',
    'date',
    'league',
    'odd_h',
    'odd_d',
    'odd_a',
    'ht',
    'at',
    'side',
    'fthg',
    'ftag',
    'adv_stats'
]
"""
Full list of features to drop
"""


EVENT_TYPE_MAP = {
    0: 'Announcement',
    1: 'Attempt',
    2: 'Corner',
    3: 'Foul',
    4: 'Yellow card',
    5: 'Second yellow card',
    6: 'Red card',
    7: 'Substitution',
    8: 'Free kick won',
    9: 'Offside',
    10: 'Hand ball',
    11: 'Penalty conceded'
}
"""
Map of encoded event types to description
"""

ASSIST_METHOD_MAP = {
    0: 'None',
    1: 'Pass',
    2: 'Cross',
    3: 'Headed pass',
    4: 'Through ball'
}
"""
Map of encoded assist method to description
"""


def preprocess():
    """
    Pre-process raw data files

    :return: None. Saves processed file as parquet.
    """

    # Load and join data
    events = pd.read_csv('data/raw/events.csv')

    ginf = pd.read_csv('data/raw/ginf.csv')

    combined_df = events.merge(
        ginf,
        on='id_odsp',
        how='inner'
    )

    # Create home team feature
    combined_df['event_team_was_home'] = np.where(
        combined_df['event_team'] == combined_df['ht'],
        1,
        0
    )

    # Un-numerate categorical features
    combined_df['assist_method'] = combined_df['assist_method'].replace(ASSIST_METHOD_MAP)
    combined_df['event_type'] = combined_df['event_type'].replace(EVENT_TYPE_MAP)

    # Drop redundant columns
    combined_df.drop(COLUMNS_TO_DROP, axis=1, inplace=True)

    # Save data
    combined_df.to_parquet('data/processed/full_data_processed.parquet', index=False)
    print('Saved data to parquet')


if __name__ == '__main__':
    preprocess()
