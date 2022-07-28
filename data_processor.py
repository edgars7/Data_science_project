import pandas as pd
import numpy as np
import csv
import datetime

# Make numpy values easier to read
np.set_printoptions(precision=3, suppress=True)

# import tensorflow as tf
# from tensorflow.keras import layers
print(f"\n######  Libraries have been imported  ######\n")

## Defines the data entires that will be used to train the machine
##                     # 0)  season (starting year)
ENTRIES = ["Date",     # 1)  date at which it was played
           "HomeTeam", # 2)  home team
           "AwayTeam", # 3)  away team
           "FTHG",     # 4)  full time home goals
           "FTAG",     # 5)  full time away goals
           "FTR",      # 6)  full time result (H, D, A)
           "HTHG",     # 7)  half time home goals
           "HTAG",     # 8)  half time away goals
           "HTR",      # 9)  half time result (H, D, A)
           "HS",       # 10) home shots
           "AS",       # 11) away shots
           "HST",      # 12) home team shots on target
           "AST",      # 13) away team shots on target
           "HF",       # 14) home team fouls
           "AF",       # 15) away team fouls
           "HC",       # 16) home team corners
           "AC",       # 17) away team corners
           "HY",       # 18) home yellow cards
           "AY",       # 19) away yellow cards
           "HR",       # 20) home red cards
           "AR"        # 21) away red cards
        ]

## Defines the season interval
BOUNDRIES = (2000, 2019) # includes the last year (min 2000, max 2019)

def main():
    ## Loads the data
    clean_data = []
    for year in range(BOUNDRIES[0], BOUNDRIES[1] + 1):
        path = "soccer_stats_data/england/england-premier-league-"+str(year)+"-to-"+str(year+1)+".csv"
        temp = pd.read_csv(path)

        ## Keeps only the necessary data
        temp1 = temp[ENTRIES]
        ## Adds a column that indicates the season
        temp1.insert(0, "season", year)
        clean_data.append(temp1)

    ## Combines all dataframes into one
    soccer_data = pd.concat(clean_data)

    ## Changes the winning statments from (H, D, A) -> (1, 0, -1)
    soccer_data = soccer_data.replace("H", 1)
    soccer_data = soccer_data.replace("D", 0)
    soccer_data = soccer_data.replace("A", -1)
    
    ## Also loads the historical standings on each team
    team_standings = pd.read_csv("soccer_stats_data/england/EPL_Standings_2000-2022.csv")
    ## Season,Pos,Team,Pld,W,D,L,GF,GA,GD,Pts
    ##    0  , 1 , 2  , 3 ,4,5,6,7 ,8 ,9 ,10

    ## Creates a new data frame for each match with the following
    ## features:
    ##
    ## AWR  -> The avg wins in the last 10 games             (home vs away)
    ## AGR  -> Avg difference in full time goals in 10 games (home vs away)
    ## ASR  -> Avg difference in shots made in 10 games      (home vs away)
    ## PSH  -> Previous 5 season standings, ?distribution?   (    home    )
    ## PSA  -> Previous 5 season standings  ? linear rn  ?   (    away    )
    ## PWHB -> The avg wins in the last 10 home games        (home vs  any)
    ## PGHB -> Avg difference in home full time goals in     (home vs  any)
    ##         10 games
    ## PSHB -> Avg difference in home shots made in 10 games (home vs  any)
    ## PWAB -> The avg wins in the last 10 away games        (away vs  any)
    ## PGAB -> Avg difference in home full time goals in     (away vs  any)
    ##         10 games
    ## PSAB -> Avg difference in away shots made in 10 games (away vs  any)
    ## PWHS -> The avg wins in the last 3 home games         (home vs  any)
    ## PGHS -> Avg difference in home full time goals in     (home vs  any)
    ##         3 games
    ## PSHS -> Avg difference in home shots made in 3 games  (home vs  any)
    ## PWAS -> The avg wins in the last 3 away games         (away vs  any)
    ## PGAS -> Avg difference in home full time goals in     (away vs  any)
    ##         3 games
    ## PSAS -> Avg difference in away shots made in 3 games  (away vs  any)
    ## TGH  -> Time between the previous game for home       (    home    )
    ## TGA  -> Time betweem the previous game for away       (    away    )
    ## DOG  -> Day on which the game is played               (    both    )
    ##
    ## The ouput variable is "res" and it can be either win, draw, loss

    ## Writes all the data in a csv file
    with open('cooked_data/feature_gen_1_list.csv', 'w') as f:
        ## Deals with the csv stuff
        header = ["Season", "Date", "HomeTeam", "AwayTeam",
                  "AWR", "AGR", "ASR", "PSH", "PSA", "PWHB", "PWAB",
                  "PGHB", "PGAB", "PSHB", "PSAB", "PWHS", "PWAS", "PGHS",
                  "PGAS", "PSHS", "PSAS", "TGH", "TGA", "DOG",
                  "Result"]
        writer = csv.writer(f)
        writer.writerow(header)

        games          = soccer_data.to_numpy()
        nice_standings = team_standings.to_numpy()
        max_wait_time  = 0
        
        for game_index in range(len(games)):
            ## Cooking time indicator
            if game_index % 200 == 0:
                print(game_index)
            
            ## AWR & AGR & ASR ####################################################
            this_game  = games[game_index]
            game_count = 0
            home_team  = this_game[2]
            away_team  = this_game[3]
            AWR        = 0
            AGR        = 0
            ASR        = 0

            # Iterates over the previous games
            for prev_index in range(game_index - 1, -1, -1):
                if game_count == 10:
                    break
                else:
                    game_temp = games[prev_index]
                    if home_team == game_temp[2] or home_team ==  game_temp[3]:
                        if away_team == game_temp[3]:
                            # The home is still home
                            game_count += 1
                            AWR += game_temp[6]
                            AGR += game_temp[4] - game_temp[5]
                            ASR += game_temp[10] - game_temp[11]
                        elif away_team == game_temp[2]:
                            # The home is now away
                            game_count += 1
                            AWR += (-1) * game_temp[6]
                            AGR += game_temp[5] - game_temp[4]
                            ASR += game_temp[11] - game_temp[10]

            if game_count != 0:
                AWR = AWR / game_count
                AGR = AGR / game_count
                ASR = ASR / game_count
            

            ## PSH & PSA ##########################################################
            ## PSH and PSA represt the avg standings in past seasons, where
            ## 1 is best and 20 is worst
            this_season = this_game[0]
            PSH         = 0
            PSA         = 0
            if this_season != 2000:
                # Iterates over previous seasons
                diff         = this_season - max(2000, this_season - 5)
                home_seasons = 0
                away_seasons = 0
                for season in range(this_season - 1, this_season - diff -1, -1):
                    # Finds both teams previous standings
                    indicator = (season - 2000) * 20
                    for i in range(indicator, indicator + 20):
                        result_stats = nice_standings[i]
                        if result_stats[2] == home_team:
                            PSH += result_stats[1]
                            home_seasons += 1
                        if result_stats[2] == away_team:
                            PSA += result_stats[1]
                            away_seasons += 1
                if home_seasons != 0:
                    PSH = PSH / home_seasons
                else:
                    PSH = 10.5
                if away_seasons != 0:
                    PSA = PSA / away_seasons
                else:
                    PSA = 10.5
            elif this_season == 2000:
                # There are no records for previous seasons, so it
                # is assumed that they had the mean result 10.5
                PSH = 10.5
                PSA = 10.5
            

            ## PWHB, PGHB, PSHB, PWAB, PGAB & PSAB ################################
            ## PWHS, PGHS, PSHS, PWAS, PGAS & PSAS ################################
            PWHB, PWHS = 0, 0
            PGHB, PGHS = 0, 0
            PSHB, PSHS = 0, 0
            PWAB, PWAS = 0, 0
            PGAB, PGAS = 0, 0
            PSAB, PSAS = 0, 0

            home_game_count = 0
            away_game_count = 0
            # Iterates over previous games
            for prev_index in range(game_index - 1, -1, -1):
                if home_game_count == 10 and away_game_count == 10:
                    break
                else:
                    game_temp = games[prev_index]
                    # Handles the home team
                    if home_team == game_temp[2] and home_game_count < 10:
                        # Home is home
                        if home_game_count < 3:
                            PWHB += game_temp[6]
                            PWHS += game_temp[6]
                            PGHB += game_temp[4] - game_temp[5]
                            PGHS += game_temp[4] - game_temp[5]
                            PSHB += game_temp[10] - game_temp[11]
                            PSHS += game_temp[10] - game_temp[11]
                            home_game_count += 1
                        else:
                            PWHB += game_temp[6]
                            PGHB += game_temp[4] - game_temp[5]
                            PSHB += game_temp[10] - game_temp[11]
                            home_game_count += 1
                    elif home_team == game_temp[3] and home_game_count < 10:
                        # Home is playing as away
                        if home_game_count < 3:
                            PWHB += (-1) * game_temp[6]
                            PWHS += (-1) * game_temp[6]
                            PGHB += game_temp[5] - game_temp[4]
                            PGHS += game_temp[5] - game_temp[4]
                            PSHB += game_temp[11] - game_temp[10]
                            PSHS += game_temp[11] - game_temp[10]
                            home_game_count += 1
                        else:
                            PWHB += (-1) * game_temp[6]
                            PGHB += game_temp[5] - game_temp[4]
                            PSHB += game_temp[11] - game_temp[10]
                            home_game_count += 1
                    
                    # Handles the away team
                    if away_team == game_temp[2] and away_game_count < 10:
                        # Away is home
                        if away_game_count < 3:
                            PWAB += game_temp[6]
                            PWAS += game_temp[6]
                            PGAB += game_temp[4] - game_temp[5]
                            PGAS += game_temp[4] - game_temp[5]
                            PSAB += game_temp[10] - game_temp[11]
                            PSAS += game_temp[10] - game_temp[11]
                            away_game_count += 1
                        else:
                            PWAB += game_temp[6]
                            PGAB += game_temp[4] - game_temp[5]
                            PSAB += game_temp[10] - game_temp[11]
                            away_game_count += 1
                    elif away_team == game_temp[3] and away_game_count < 10:
                        # Away is playing as away
                        if away_game_count < 3:
                            PWAB += (-1) * game_temp[6]
                            PWAS += (-1) * game_temp[6]
                            PGAB += game_temp[5] - game_temp[4]
                            PGAS += game_temp[5] - game_temp[4]
                            PSAB += game_temp[11] - game_temp[10]
                            PSAS += game_temp[11] - game_temp[10]
                            away_game_count += 1
                        else:
                            PWAB += (-1) * game_temp[6]
                            PGAB += game_temp[5] - game_temp[4]
                            PSAB += game_temp[11] - game_temp[10]
                            away_game_count += 1
            
            # Calculates the averages of these values
            if home_game_count < 3 and home_game_count > 0:
                PWHB, PWHS = PWHB / home_game_count, PWHS / home_game_count
                PGHB, PGHS = PGHB / home_game_count, PGHS / home_game_count
                PSHB, PSHS = PSHB / home_game_count, PSHS / home_game_count
            elif home_game_count >= 3:
                PWHB, PWHS = PWHB / home_game_count, PWHS / 3
                PGHB, PGHS = PGHB / home_game_count, PGHS / 3
                PSHB, PSHS = PSHB / home_game_count, PSHS / 3
            
            if away_game_count < 3 and away_game_count > 0:
                PWAB, PWAS = PWAB / away_game_count, PWAS / away_game_count
                PGAB, PGAS = PGAB / away_game_count, PGAS / away_game_count
                PSAB, PSAS = PSAB / away_game_count, PSAS / away_game_count
            elif away_game_count >= 3:
                PWAB, PWAS = PWAB / away_game_count, PWAS / 3
                PGAB, PGAS = PGAB / away_game_count, PGAS / 3
                PSAB, PSAS = PSAB / away_game_count, PSAS / 3
            

            ## TGH, TGA, DOG ######################################################
            TGH = 0
            TGA = 0
            
            # Gets this matches date
            game_date = get_date(this_game[1])
            DOG       = game_date.weekday()
            
            # Finds the last game home team played
            HDNF = True # home date not found
            ADNF = True # away date not found
            for prev_index in range(game_index - 1, -1, -1):
                game_temp = games[prev_index]
                # Checks if the game is in this season
                if game_temp[0] != this_game[0]:
                    if HDNF:
                        TGH = 70
                    if ADNF:
                        TGA = 70
                    break
                else:
                    if (home_team == game_temp[2] or home_team == game_temp[3]) and HDNF:
                        prev_home = get_date(game_temp[1])
                        TGH       = (game_date - prev_home).days
                        HDNF      = False

                    if (away_team == game_temp[2] or away_team == game_temp[3]) and ADNF:
                        prev_away = get_date(game_temp[1])
                        TGA       = (game_date - prev_away).days
                        ADNF      = False
            
            
            ## Sets up the result #################################################
            RESULT = [0, 0, 0]
            if this_game[6] == 1:
                RESULT[0] = 1
            elif this_game[6] == -1:
                RESULT[2] = 1
            else:
                RESULT[1] = 1
            

        
            writer.writerow([this_season, this_game[1], home_team, away_team,
                AWR, AGR, ASR, PSH, PSA, PWHB, PWAB, PGHB, PGAB, PSHB,
                PSAB, PWHS, PWAS, PGHS, PGAS, PSHS, PSAS, TGH, TGA, DOG,
                RESULT])
        

def get_date(date_string):
    """
    Returns the date specified in the string
    'dd/mm/yyyy', where year is >= 2000
    """
    date_temp = date_string.split("/")
    if len(date_temp[2]) == 4:
        date_temp[2] = date_temp[2][2:]
    this_date = datetime.date(int("20"+date_temp[2]), 
                              int(date_temp[1]), int(date_temp[0]))
    return this_date


if __name__ == "__main__":
    main()
