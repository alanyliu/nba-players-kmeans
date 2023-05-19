import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


def data_vis(data):
    # generate PER for all players
    player_pers = []
    for idx in range(data.shape[0]):
        player_pers.append(per_calc(data, idx))

    # 3-point shooting vs. player efficiency
    col = np.where(data['3P%'] < 0.3, 'r', np.where(data['3P%'] < 0.37, 'y', 'g'))
    plt.figure(0)
    plt.scatter(player_pers, data['3P%'], c=col)
    plt.title("3-Point Shooting vs. Player Efficiency")
    plt.xlabel("Efficiency rating (PER)")
    plt.ylabel("3-point percentage")
    plt.show()

    # 2-point shooting vs. player efficiency
    col = np.where(data['2P%'] < 0.45, 'r', np.where(data['2P%'] < 0.55, 'y', 'g'))
    plt.figure(1)
    plt.scatter(player_pers, data['2P%'], c=col)
    plt.title("2-Point Shooting vs. Player Efficiency")
    plt.xlabel("Efficiency rating (PER)")
    plt.ylabel("2-point percentage")
    plt.show()

    # FG percentage vs. player efficiency
    col = np.where(data['FG%'] < 0.4, 'r', np.where(data['FG%'] < 0.47, 'y', 'g'))
    plt.figure(2)
    plt.scatter(player_pers, data['FG%'], c=col)
    plt.title("Field Goal Percentage vs. Player Efficiency")
    plt.xlabel("Efficiency rating (PER)")
    plt.ylabel("Field-goal percentage")
    plt.show()

    # Assist/TO ratio vs. player efficiency
    col = np.where(data['AST'] / data['TOV'] < 1, 'r', np.where(data['AST'] / data['TOV'] < 2, 'y', 'g'))
    plt.figure(3)
    plt.scatter(player_pers, data['AST'] / data['TOV'], c=col)
    plt.title("Field Goal Percentage vs. Player Efficiency")
    plt.xlabel("Efficiency rating (PER)")
    plt.ylabel("Field-goal percentage")
    plt.show()

def reg_analysis(data):
    player_pers = []
    for idx in range(data.shape[0]):
        player_pers.append(per_calc(data, idx))

    # Simple linear regression between FG PCT and player efficiency
    player_pers_valid = np.zeros((596, 1))
    fg_pct = np.zeros((596, 1))
    idx = 0
    for per, fg in zip(player_pers, data['FG%']):
        if -float('inf') <= per <= float('inf') and 0 <= fg <= 100:
            player_pers_valid[idx, 0] = per
            fg_pct[idx, 0] = fg
            idx += 1

    fg_slr = LinearRegression().fit(player_pers_valid, fg_pct)
    slr_rsquared = fg_slr.score(player_pers_valid, fg_pct)
    plt.figure(4)
    plt.scatter(player_pers_valid, fg_pct, c='black')
    print(fg_slr.intercept_)
    print(fg_slr.coef_)
    plt.plot(player_pers_valid, fg_slr.intercept_ + fg_slr.coef_ * player_pers_valid, c='red')
    plt.title("SLR: FG% vs. PER (R^2 = " + str(slr_rsquared) + ")")
    plt.xlabel("Efficiency rating (PER)")
    plt.ylabel("Field-goal percentage")
    plt.show()

    # Multiple linear regression for PER on the regressors G, MP, FG%, 3P%, FT%, TRB, AST, STL, BLK, TOV, PF, PTS
    sub_data = data[['G', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST',
                     'STL', 'BLK', 'TOV', 'PF', 'PTS']]
    sub_data = sub_data.dropna()
    player_pers_mlr = np.zeros((sub_data.shape[0], 1))
    for idx in range(sub_data.shape[0]):
        player_pers_mlr[idx] = per_calc(sub_data, idx)
    sub_data = sub_data[['G', 'MP', 'FG%', '3P%', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']]

    print(sub_data.shape)
    print(player_pers_mlr.shape)
    mlr = LinearRegression().fit(sub_data, player_pers_mlr)
    mlr_rsquared = mlr.score(sub_data, player_pers_mlr)
    print(mlr.coef_)
    print(mlr_rsquared)
    return


def kmeans(data, clusters):
    features = data.columns.values[4:]
    x = data.loc[:, features].values

    # Deal with NaN entries
    nan_cols = [5, 8, 11, 12, 15]
    for col in nan_cols:
        row = 0
        for pct in x[:, col]:
            if not 0 <= pct <= 100:
                x[row, col] = 0.0
            row += 1

    # Perform dimensionality reduction (principal component analysis)
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(x)
    pc_df = pd.DataFrame(data=principal_components, columns=["PC1", "PC2"])

    # Select random centroids from the data
    centroids = pc_df.sample(n=clusters)
    print("Initial centroids randomly chosen:")
    print(centroids)

    # Predict cluster labels
    km = KMeans(n_clusters=clusters, init='random', n_init=10, max_iter=2000, tol=1e-2)
    y_km = km.fit_predict(pc_df)

    # Print players in each cluster
    players_grouped = [[], [], [], [], [], [], [], [], [], []]
    major_players = [[], [], [], [], [], [], [], [], [], []]
    for idx in range(len(y_km)):
        if per_calc(data, idx) > 12 and len(major_players[y_km[idx]]) < 8 and data.iloc[idx, 4] > 50:
            major_players[y_km[idx]].append(data.iloc[idx, 0])
        players_grouped[y_km[idx]].append(data.iloc[idx, 0])
    print("All players and their corresponding cluster:")
    for idx, cluster in enumerate(players_grouped, 1):
        print("Cluster " + str(idx) + ": " + str(cluster))
    print("Major players from each cluster:")
    for idx, cluster in enumerate(major_players, 1):
        print("Cluster " + str(idx) + ": " + str(cluster))

    # Plot the clusters
    plt.figure(5)
    plt.scatter(pc_df.iloc[y_km == 0, 0], pc_df.iloc[y_km == 0, 1], c="orange", label="Cluster 1")
    plt.scatter(pc_df.iloc[y_km == 1, 0], pc_df.iloc[y_km == 1, 1], c="blue", label="Cluster 2")
    plt.scatter(pc_df.iloc[y_km == 2, 0], pc_df.iloc[y_km == 2, 1], c="pink", label="Cluster 3")
    plt.scatter(pc_df.iloc[y_km == 3, 0], pc_df.iloc[y_km == 3, 1], c="green", label="Cluster 4")
    plt.scatter(pc_df.iloc[y_km == 4, 0], pc_df.iloc[y_km == 4, 1], c="lightblue", label="Cluster 5")
    plt.scatter(pc_df.iloc[y_km == 5, 0], pc_df.iloc[y_km == 5, 1], c="purple", label="Cluster 6")
    plt.scatter(pc_df.iloc[y_km == 6, 0], pc_df.iloc[y_km == 6, 1], c="yellow", label="Cluster 7")
    plt.scatter(pc_df.iloc[y_km == 7, 0], pc_df.iloc[y_km == 7, 1], c="brown", label="Cluster 8")
    plt.scatter(pc_df.iloc[y_km == 8, 0], pc_df.iloc[y_km == 8, 1], c="lightgreen", label="Cluster 9")
    plt.scatter(pc_df.iloc[y_km == 9, 0], pc_df.iloc[y_km == 9, 1], c="navy", label="Cluster 10")

    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], marker='*', c='red', edgecolor='black',
                label='centroids')
    plt.legend(scatterpoints=1)
    plt.grid()
    plt.show()

    return


def per_calc(data, player_idx):
    # Calculates the estimated Player Efficiency Rating for an NBA player in the 2021-22 season
    return 1.591 * data['FG'].iloc[player_idx] + 0.998 * data['STL'].iloc[player_idx] + \
        0.958 * data['3P'].iloc[player_idx] + 0.868 * data['FT'].iloc[player_idx] + \
        0.726 * data['BLK'].iloc[player_idx] + 0.726 * data['ORB'].iloc[player_idx] + \
        0.642 * data['AST'].iloc[player_idx] + 0.272 * data['DRB'].iloc[player_idx] - \
        0.318 * data['PF'].iloc[player_idx] - 0.372 * (data['FTA'].iloc[player_idx] - data['FT'].iloc[player_idx]) - \
        0.726 * (data['FGA'].iloc[player_idx] - data['FG'].iloc[player_idx]) - 0.998 * data['TOV'].iloc[player_idx]


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Load 2021-22 season data
    player_data = pd.read_csv("data/2021_22_NBA_Season_Stats.csv")
    print(player_data)

    # Data visualization (random graphs)
    data_vis(player_data)

    # Regression analysis (SLR on FG%, MLR on 14 regressors)
    reg_analysis(player_data)

    # K-means clustering to group players' playing styles based on stats
    kmeans(player_data, 10)