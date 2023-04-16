import copy

import numpy as np
import pandas as pd
import xgboost as xgb
from colorama import Fore, Style, init, deinit
from src.Utils import Expected_Value


# from src.Utils.Dictionaries import team_index_current
# from src.Utils.tools import get_json_data, to_data_frame, get_todays_games_json, create_todays_games
init()
xgb_ml = xgb.Booster()
xgb_ml.load_model('Models/XGBoost_Models/XGBoost_68.6%_ML-2.json')
xgb_uo = xgb.Booster()
xgb_uo.load_model('Models/XGBoost_Models/XGBoost_54.8%_UO-8.json')



def xgb_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds):
    ml_predictions_array = []

    for row in data:
        ml_predictions_array.append(xgb_ml.predict(xgb.DMatrix(np.array([row]))))

    frame_uo = copy.deepcopy(frame_ml)
    frame_uo['OU'] = np.asarray(todays_games_uo)
    data = frame_uo.values
    data = data.astype(float)

    ou_predictions_array = []

    for row in data:
        ou_predictions_array.append(xgb_uo.predict(xgb.DMatrix(np.array([row]))))

    output = {}
    output["predictions"] = []
    count = 0
    for game in games:
        home_team = game[0]
        away_team = game[1]
        winner = int(np.argmax(ml_predictions_array[count]))
        under_over = int(np.argmax(ou_predictions_array[count]))
        winner_confidence = ml_predictions_array[count]
        un_confidence = ou_predictions_array[count]
        if winner == 1:
            winner_confidence = round(winner_confidence[0][1] * 100, 1)
            if under_over == 0:
                un_confidence = round(ou_predictions_array[count][0][0] * 100, 1)
                ev_home = float(Expected_Value.expected_value(ml_predictions_array[count][0][1], int(home_team_odds[count])))
                ev_away = float(Expected_Value.expected_value(ml_predictions_array[count][0][0], int(away_team_odds[count])))
                prediction = {
                    "home_team": home_team,
                    "away_team": away_team,
                    "winner": "home",
                    "confidence": winner_confidence,
                    "over_under": "under",
                    "odds": todays_games_uo[count],
                    "ou_confidence": un_confidence,
                    "ev_home": f"{ev_home:+.2f}",
                    "ev_away": f"{ev_away:+.2f}"
                }
                output["predictions"].append(prediction)
            else:
                un_confidence = round(ou_predictions_array[count][0][1] * 100, 1)
                ev_home = float(Expected_Value.expected_value(ml_predictions_array[count][0][1], int(home_team_odds[count])))
                ev_away = float(Expected_Value.expected_value(ml_predictions_array[count][0][0], int(away_team_odds[count])))
                prediction = {
                    "home_team": home_team,
                    "away_team": away_team,
                    "winner": "home",
                    "confidence": winner_confidence,
                    "over_under": "over",
                    "odds": todays_games_uo[count],
                    "ou_confidence": un_confidence,
                    "ev_home": f"{ev_home:+.2f}",
                    "ev_away": f"{ev_away:+.2f}"
                }
                output["predictions"].append(prediction)
        else:
            winner_confidence = round(winner_confidence[0][0] * 100, 1)
            if under_over == 0:
                un_confidence = round(ou_predictions_array[count][0][0] * 100, 1)
                ev_home = float(
                    Expected_Value.expected_value(ml_predictions_array[count][0][0], int(away_team_odds[count])))
                ev_away = float(
                    Expected_Value.expected_value(ml_predictions_array[count][0][1], int(home_team_odds[count])))
                prediction = {
                    "home_team": home_team,
                    "away_team": away_team,
                    "winner": "away",
                    "confidence": winner_confidence,
                    "over_under": "under",
                    "odds": todays_games_uo[count],
                    "ou_confidence": un_confidence,
                    "ev_home": f"{ev_home:+.2f}",
                    "ev_away": f"{ev_away:+.2f}"
                }
                output["predictions"].append(prediction)
            else:
                un_confidence = round(ou_predictions_array[count][0][1] * 100, 1)
                ev_home = float(
                    Expected_Value.expected_value(ml_predictions_array[count][0][0], int(away_team_odds[count])))
                ev_away = float(
                    Expected_Value.expected_value(ml_predictions_array[count][0][1], int(home_team_odds[count])))
                prediction = {
                    "home_team": home_team,
                    "away_team": away_team,
                    "winner": "away",
                    "confidence": winner_confidence,
                    "over_under": "over",
                    "odds": todays_games_uo[count],
                    "ou_confidence": un_confidence,
                    "ev_home": f"{ev_home:+.2f}",
                    "ev_away": f"{ev_away:+.2f}"
                }
                output["predictions"].append(prediction)
        count += 1

    return output

