import os
import csv
import json

import tqdm
import pandas as pd


def convert_json_to_csv(json_file_path, output_directory):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        datas = json.load(file)

    for idx, data in enumerate(tqdm.tqdm(datas)):
        models = data['models']
        general_figure_info = data['general_figure_info']
        if data['type'] == 'dot_line' or data['type'] == 'line':      

            # 헤더 작성
            title = general_figure_info['title']['text']
            data_dict = {}
            x_label = general_figure_info['x_axis']['label']['text']
            y_label = general_figure_info['y_axis']['label']['text']

            x_ticks_dict = {}
            for x_tick, x_value in zip(general_figure_info['x_axis']['major_ticks']['values'], general_figure_info['x_axis']['major_labels']['values']):
                if isinstance(x_value, float):
                    if x_value.is_integer():
                        x_value = int(x_value)  # float 값을 정수로 변환
                    else:
                        x_value = round(x_value, 3)
                x_ticks_dict[str(x_tick)] = x_value
            
            rows = []
            for model in models:
                if len(models) == 1:
                    model_name = title
                else:
                    model_name = model['name']
                x_values = [x_ticks_dict[str(x)] for x in model['x']]
                y_values = model['y']

                for x, y in zip(x_values, y_values):
                    if isinstance(y, float):
                        if y.is_integer():
                            y = int(y)  # float 값을 정수로 변환
                        else:
                            y = round(y, 3)
                    rows.append({'특성': model_name, 'X': x, 'Y': y})
        elif data['type'] == 'vbar_categorical':
            rows = []
            for model in models:
                if len(models) == 1:
                    label = general_figure_info['title']['text']
                else:
                    label = model['name']
                x_values = general_figure_info['x_axis']['major_labels']['values']
                y_values = model['y']

                for x, y in zip(x_values, y_values):
                    if isinstance(y, float):
                        if y.is_integer():
                            y = int(y)  # float 값을 정수로 변환
                        else:
                            y = round(y, 3)
                    if isinstance(x, float):
                        if x.is_integer():
                            x = int(x)  # float 값을 정수로 변환
                        else:
                            x = round(x, 3)                        
                    rows.append({'특성': label, 'X': x, 'Y': y})


        elif data['type'] == 'hbar_categorical':
            rows = []
            for model in models:
                if len(models) == 1:
                    label = general_figure_info['title']['text']
                else:
                    label = model['name']
                x_values = model['x']
                y_values = model['y']

                for x, y in zip(x_values, y_values):
                    if isinstance(x, float):
                        if x.is_integer():
                            x = int(x)  # float 값을 정수로 변환
                        else:
                            x = round(x, 3)
                    if isinstance(y, float):
                        if y.is_integer():
                            y = int(y)  # float 값을 정수로 변환
                        else:
                            y = round(y, 3)
                    rows.append({'특성': label, 'X': y, 'Y': x})
        else:
            raise ValueError(f"Unknown type: {data['type']}")

        df = pd.DataFrame(rows)
        # Pivot the DataFrame
        df_pivot = df.pivot(index='특성', columns='X', values='Y')

        # Reset the index
        df_pivot.reset_index(inplace=True)

        # Save the DataFrame to CSV
        output_file_path = os.path.join(output_directory, f"{idx}.csv")
        df_pivot.to_csv(output_file_path, index=False, header=True, quoting=csv.QUOTE_NONNUMERIC)

if __name__ == "__main__":
    json_file_path = '/root/PlotQA/data/translated_train/annotations.json'
    output_directory = '/root/PlotQA/data/translated_train/csv/'
    
    # Check if directory exists, if not, create it
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    convert_json_to_csv(json_file_path, output_directory)