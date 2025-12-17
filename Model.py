import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# DATA
df = pd.read_csv("match_data.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')

cols_to_drop = ['Unnamed: 0', 'Comp', 'Notes', 'Match Report', 'Attendance', 'Captain']
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

team_to_opp = {
    'Tottenham Hotspur': 'Tottenham', 'Manchester United': 'Manchester Utd',
    'Manchester City': 'Manchester City', 'Newcastle United': 'Newcastle Utd',
    'Wolverhampton Wanderers': 'Wolves', 'Brighton and Hove Albion': 'Brighton',
    'West Ham United': 'West Ham', 'Nottingham Forest': "Nott'ham Forest"
}
opp_to_team = {v: k for k, v in team_to_opp.items()}
for t in df['team'].unique():
    if t not in team_to_opp:
        opp_to_team[t] = t

df['Opponent_Std'] = df['Opponent'].map(opp_to_team).fillna(df['Opponent'])

# ELO RATING
current_elo = {team: 1500 for team in df['team'].unique()}
processed_matches = set()
k_factor = 32

def get_expected_score(rating, opp_rating):
    return 1 / (1 + 10 ** ((opp_rating - rating) / 400))

def update_elo(rating, expected, actual, k=32):
    return rating + k * (actual - expected)

feature_team_elo = []
feature_opp_elo = []

for index, row in df.iterrows():
    team = row['team']
    opp = row['Opponent_Std']
    
    t_elo = current_elo[team]
    o_elo = current_elo[opp]
    feature_team_elo.append(t_elo)
    feature_opp_elo.append(o_elo)
    
    match_id = tuple(sorted([team, opp])) + (row['Date'],)
    if match_id not in processed_matches:
        if row['Result'] == 'W': score = 1; opp_score = 0
        elif row['Result'] == 'D': score = 0.5; opp_score = 0.5
        else: score = 0; opp_score = 1
            
        exp_team = get_expected_score(t_elo, o_elo)
        exp_opp = get_expected_score(o_elo, t_elo)
        
        current_elo[team] = update_elo(t_elo, exp_team, score, k_factor)
        current_elo[opp] = update_elo(o_elo, exp_opp, opp_score, k_factor)
        processed_matches.add(match_id)

df['elo'] = feature_team_elo
df['opp_elo'] = feature_opp_elo

# ROLLING STATS
cols_to_roll = ['GF', 'GA', 'xG', 'xGA', 'Poss', 'Sh', 'SoT', 'Dist', 'FK', 'PK', 'PKatt']
rolling_cols = [f'rolling_{c}' for c in cols_to_roll]

def get_rolling_stats(group):
    return group.sort_values('Date')[cols_to_roll].shift(1).rolling(3, min_periods=1).mean()

rolling_stats = df.groupby('team', group_keys=False).apply(get_rolling_stats)
rolling_stats.columns = rolling_cols
df = df.join(rolling_stats)

# MERGING OPPONENT STATS
stats_lookup = df[['Date', 'team'] + rolling_cols].copy()
stats_lookup = stats_lookup.rename(columns={'team': 'Opponent_Std'})
for c in rolling_cols:
    stats_lookup = stats_lookup.rename(columns={c: f'opp_{c}'})

df = pd.merge(df, stats_lookup, on=['Date', 'Opponent_Std'], how='left')
df_final = df.dropna().copy()

# TRAINING MODEL
le_team = LabelEncoder()
le_team.fit(df['team'].unique())

df_final['team_code'] = le_team.transform(df_final['team'])
df_final['Opp_code'] = le_team.transform(df_final['Opponent_Std'])
df_final['Venue_code'] = df_final['Venue'].astype('category').cat.codes
df_final['Hour'] = df_final['Time'].str.replace(':.+', '', regex=True).astype(int)
df_final['Target'] = df_final['Result'].map({'W': 2, 'D': 1, 'L': 0})

features = ['Venue_code', 'team_code', 'Opp_code', 'Hour', 'elo', 'opp_elo'] + rolling_cols + [f'opp_{c}' for c in rolling_cols]

X = df_final[features]
y = df_final['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=1)
rf.fit(X_train, y_train)

def get_model_accuracy(X_data, y_data):
    """
    Calculates and returns the accuracy of the model.
    Prints a detailed classification report.
    """
    predictions = rf.predict(X_data)
    acc = accuracy_score(y_data, predictions)
    report = classification_report(y_data, predictions)
    print(f"Model Accuracy: {acc:.2%}")
    print("\nDetailed Report:\n", report)
    return acc

get_model_accuracy(X_test, y_test)

# MATCHUP
def predict_match_final(team1, team2):
    if team1 not in current_elo or team2 not in current_elo:
        return "Error: Team not found."
        
    t1_latest = df[df['team'] == team1].iloc[-1]
    t2_latest = df[df['team'] == team2].iloc[-1]
    
    input_data = {
        'Venue_code': 1, # Home
        'team_code': le_team.transform([team1])[0],
        'Opp_code': le_team.transform([team2])[0],
        'Hour': 15,
        'elo': current_elo[team1],     
        'opp_elo': current_elo[team2]
    }
    
    # Fill rolling stats
    for c in rolling_cols:
        input_data[c] = t1_latest[c]
        input_data[f'opp_{c}'] = t2_latest[c]
        
    X_new = pd.DataFrame([input_data])[features]
    probs = rf.predict_proba(X_new)[0]
    pred = rf.predict(X_new)[0]
    
    res_map = {2: 'Win', 1: 'Draw', 0: 'Loss'}
    return f"Prediction for {team1}: {res_map[pred]} (Win: {probs[2]:.0%}, Draw: {probs[1]:.0%}, Loss: {probs[0]:.0%})"

# Example Usage
print(predict_match_final('Arsenal', 'Tottenham Hotspur'))