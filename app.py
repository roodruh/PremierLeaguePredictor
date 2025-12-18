from flask import Flask, render_template, request
from Model import *

app = Flask(__name__)

# Form
TEAMS = [
    'Arsenal',
    'Aston Villa',
    'Bournemouth',
    'Brentford',
    'Brighton and Hove Albion',
    'Chelsea',
    'Crystal Palace',
    'Everton',
    'Fulham',
    'Ipswich Town',
    'Leicester City',
    'Liverpool',
    'Manchester City',
    'Manchester United',
    'Newcastle United',
    'Nottingham Forest',
    'Southampton',
    'Tottenham Hotspur',
    'West Ham United',
    'Wolverhampton Wanderers'
]

def match_sim(t1, t2):
    if t1 != t2:
        prediction = predict_matchup(t1, t2)
        return prediction
    
    return "Error: Please select different teams"

@app.route("/", methods=["GET","POST"])
def index():

    if request.method == "POST":
        team1 = request.form.get("team1")
        team2 = request.form.get("team2")
        result = match_sim(team1, team2)
        return render_template("index.html", teams = TEAMS, result = result)
    
    return render_template('index.html', teams = TEAMS)