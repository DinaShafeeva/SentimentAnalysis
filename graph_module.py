import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from model_module import ModelModule


class GraphModule:

    def create_graph(self, file_name, column_id, column_text, column_time):
        data = pd.read_csv('upload/' + file_name)
        fig, ax = plt.subplots(nrows=1, ncols=1)
        plt.yticks([-4, -3.5, -3, -2, -1, 0, 2, 5],
                   ['злость', 'отвращ.', 'страх', 'грусть', 'стыд', 'нейтр.', 'счастье', 'удивл.'])
        plt.xlabel('время')

        user_count = data[column_id].nunique(dropna=False)
        dfs = dict(tuple(data.groupby(column_id)))
        print(user_count)
        bert_model = ModelModule()
        for i in range(user_count):
            x = []
            y = []
            for index, row in dfs[i].iterrows():
                print(row[column_text])
                x.append(datetime.strptime(row[column_time], '%d/%m/%y %H:%M:%S'))
                comment = row[column_text]
                y.append(bert_model.predict_emotion_value(comment))
                plt.plot(x, y, marker="o", markersize=5)

        fig.savefig('static/plot.png')
        plt.close(fig)
