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

        # 헤더 작성
        title = general_figure_info['title']['text']
        data_dict = {}
        x_label = general_figure_info['x_axis']['label']['text']
        y_label = general_figure_info['y_axis']['label']['text']

        df = pd.DataFrame(columns=['Model', 'X', 'Y'])

        x_ticks_dict = {}
        for x_tick, x_value in zip(general_figure_info['x_axis']['major_ticks']['values'], general_figure_info['x_axis']['major_labels']['values']):
            if isinstance(x_value, (float, int)) and x_value.is_integer():
                x_value = int(x_value)
            x_ticks_dict[str(x_tick)] = x_value
        
        rows = []
        for model in models:
            model_name = model['label']
            x_values = [x_ticks_dict[str(x)] for x in model['x']]
            y_values = model['y']

            for x, y in zip(x_values, y_values):
                if isinstance(y, (float, int)) and not y.is_integer():
                    y = round(y, 2)
                rows.append({'특성': model_name, 'X': x, 'Y': y})
        df = pd.DataFrame(rows)
        # Pivot the DataFrame
        df_pivot = df.pivot(index='특성', columns='X', values='Y')

        # Reset the index
        df_pivot.reset_index(inplace=True)

        # Save the DataFrame to CSV
        output_file_path = os.path.join(output_directory, f"output_{idx}.csv")
        df_pivot.to_csv(output_file_path, index=False, header=True, quoting=csv.QUOTE_NONNUMERIC)

if __name__ == "__main__":
    json_file_path = '/root/label.json'
    output_directory = '/root/output'
    
    # Check if directory exists, if not, create it
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    convert_json_to_csv(json_file_path, output_directory)