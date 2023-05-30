from shiny import App, render, ui, reactive
import shiny.experimental as x
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from main import kmeans, per_calc

# read the data
player_data = pd.read_csv("data/2021_22_NBA_Season_Stats.csv")

# user interface for Shiny app
app_ui = x.ui.page_fillable(
    ui.navset_tab(
        ui.nav('K-Means Clustering',
               ui.h3('NBA Player Statistics Analysis (2021-22)'),
               ui.p(
                   'A visualization of running the k-means clustering algorithm with the two optimal principal '
                   'components on an axis each. Select a value for the number of clusters, and see what '
                   'happens!'),
               ui.br(),
               ui.input_slider('k', 'Select k:', min=1, max=20, value=10, step=1),
               ui.input_switch('by_clusters', 'Highlight clusters', value=True),
               ui.input_switch('player_labels', 'Show select player labels', value=False),
               ui.input_action_button('run', 'Rerun Algorithm', class_="btn-success btn-med-length"),
               x.ui.output_plot('scatter', fill=True)
               ),
        ui.nav('Visualizations',
               ui.h3('Data Visualizations (2021-22)'),
               x.ui.layout_sidebar(
                   x.ui.sidebar(
                       ui.p(
                           'Enter two variables and visualize the correlations between them!'
                       ),
                       ui.br(),
                       ui.input_radio_buttons(
                           'var1', 'Variable 1', ['G', 'GS', 'FG%', '3P%', '2P%', 'ORB', 'TRB', 'AST', 'STL', 'BLK',
                                                  'TOV', 'PTS', 'PER'], selected='FG%'
                       ),
                       ui.input_radio_buttons(
                           'var2', 'Variable 2', ['G', 'GS', 'FG%', '3P%', '2P%', 'ORB', 'TRB', 'AST', 'STL', 'BLK',
                                                  'TOV', 'PTS', 'PER'], selected='PER'
                       ),
                       ui.input_switch('player_labels2', 'Show select player labels', value=False),
                   ),
                   x.ui.output_plot('reg', fill=True)
               )
               )
    )
)


# server for Shiny app
def server(input, output, session):
    @output
    @render.plot
    @reactive.event(input.run, ignore_none=False)
    def scatter():
        pc_df, clust_num, km_obj = kmeans(player_data, input.k())
        pc_df['clust_num'] = clust_num

        rcParams['figure.figsize'] = 50, 20

        sns.scatterplot(
            data=pc_df,
            x='PC1',
            y='PC2',
            hue='clust_num' if input.by_clusters() else None,
            legend=False
        )
        if input.player_labels():
            temp_df = pd.concat({'PC1': pc_df['PC1'], 'PC2': pc_df['PC2'], 'val': player_data['Player']}, axis=1)
            star_list = ['Nikola Jokić', 'Giannis Antetokounmpo', 'Joel Embiid', 'Luka Dončić', 'Stephen Curry',
                         'LeBron James', 'James Harden', 'Jayson Tatum', 'Trae Young', 'Jaylen Brown', 'Jimmy Butler',
                         'Julius Randle', 'Bradley Beal', 'Donovan Mitchell', 'Domantas Sabonis', 'Rudy Gobert',
                         'Deandre Ayton', 'Pascal Siakam', 'Spencer Dinwiddie', 'Marcus Smart', 'Steven Adams',
                         'Clint Capela', 'Robert Williams', 'Jrue Holiday', "De'Aaron Fox", 'Facundo Campazzo',
                         'Danny Green', 'Joe Ingles', 'Chris Paul', 'Caleb Martin', "Royce O'Neale",
                         'Juancho Hernangomez', 'Markelle Fultz', 'Emmanuel Mudiay', 'Solomon Hill', 'Jordan Bell',
                         'Tyrese Haliburton', 'Tyler Herro', 'Grayson Allen', 'Carmelo Anthony', 'DeMar DeRozan',
                         'Bobby Portis', 'Christian Wood', 'Trey Burke', 'Elfrid Payton', 'LaMelo Ball', 'Kyle Lowry',
                         'Thaddeus Young', 'P.J. Tucker', 'P.J. Washington', 'Christian Wood', 'Jalen Green',
                         'Devin Booker', 'Karl-Anthony Towns', 'Boban Marjanović', 'Enes Freedom', 'Kelly Oubre Jr.']
            for i, row_series in temp_df.iterrows():
                if player_data['Player'].iloc[i] in star_list:
                    plt.gca().text(row_series['PC1'] + .02, row_series['PC2'], str(row_series['val']), size='xx-small')

        # col = []
        # for idx, cluster in enumerate(clust_num, 0):
        #     if cluster == 0:
        #         col.append('red')
        #     elif cluster == 1:
        #         col.append('blue')
        #     elif cluster == 2:
        #         col.append('green')
        #     elif cluster == 3:
        #         col.append('purple')
        #     elif cluster == 4:
        #         col.append('lightblue')
        #     elif cluster == 5:
        #         col.append('orange')
        #     elif cluster == 6:
        #         col.append('yellow')
        #     elif cluster == 7:
        #         col.append('brown')
        #     elif cluster == 8:
        #         col.append('navy')
        #     elif cluster == 9:
        #         col.append('lightgreen')
        #     elif cluster == 10:
        #         col.append('pink')
        #     elif cluster == 11:
        #         col.append('gray')
        #     elif cluster == 12:
        #         col.append('lightsalmon')
        #     elif cluster == 13:
        #         col.append('teal')
        #     elif cluster == 14:
        #         col.append('fuchsia')
        #     elif cluster == 15:
        #         col.append('aqua')
        #     elif cluster == 16:
        #         col.append('mediumpurple')
        #     elif cluster == 17:
        #         col.append('gold')
        #     elif cluster == 18:
        #         col.append('greenyellow')
        #     elif cluster == 19:
        #         col.append('chocolate')
        # pc_df['col'] = col

        # col = ['red', 'blue', 'green', 'purple', 'lightblue', 'orange', 'yellow', 'lightgreen', 'navy', 'brown', 'pink',
        #        'teal', 'fuchsia', 'chocolate', 'gray', 'lightsalmon', 'mediumpurple', 'gold', 'aqua', 'greenyellow']
        # pc_df, y_km = kmeans(player_data, input.k())
        # for cluster in range(1, input.k() + 1):
        #     plt.scatter(pc_df.iloc[y_km == 0, 0], pc_df.iloc[y_km == 0, 1], c=col[cluster - 1],
        #                 label="Cluster " + str(cluster))
        # return plt

    @output
    @render.plot
    @reactive.event(input.var1, input.var2, input.player_labels2, ignore_none=False)
    def reg():
        player_pers = []
        for idx in range(len(player_data)):
            player_pers.append(per_calc(player_data, idx))
        player_data['PER'] = player_pers
        sns.scatterplot(
            data=player_data,
            x=input.var1(),
            y=input.var2(),
            legend='full'
        )
        if input.player_labels2():
            temp_df = pd.concat({input.var1(): player_data[input.var1()], input.var2(): player_data[input.var2()],
                                 'val': player_data['Player']}, axis=1)
            star_list = ['Nikola Jokić', 'Giannis Antetokounmpo', 'Joel Embiid', 'Luka Dončić', 'Stephen Curry',
                         'LeBron James', 'James Harden', 'Jayson Tatum', 'Trae Young', 'Jaylen Brown', 'Jimmy Butler',
                         'Julius Randle', 'Bradley Beal', 'Donovan Mitchell', 'Domantas Sabonis', 'Rudy Gobert',
                         'Deandre Ayton', 'Pascal Siakam', 'Marcus Smart', 'Steven Adams', 'Clint Capela',
                         'Robert Williams', 'Jrue Holiday', "De'Aaron Fox", 'Chris Paul', "Royce O'Neale",
                         'Markelle Fultz', 'Tyrese Haliburton', 'Tyler Herro', 'Carmelo Anthony', 'DeMar DeRozan',
                         'Bobby Portis', 'Christian Wood', 'Elfrid Payton', 'LaMelo Ball', 'P.J. Washington',
                         'Christian Wood', 'Jalen Green', 'Devin Booker', 'Karl-Anthony Towns', 'Kelly Oubre Jr.']
            for i, row_series in temp_df.iterrows():
                if player_data['Player'].iloc[i] in star_list:
                    plt.gca().text(row_series[input.var1()] + .02, row_series[input.var2()], str(row_series['val']),
                                   size='xx-small')


# run the application
app = App(app_ui, server)
