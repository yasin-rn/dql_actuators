import torch
import pandas as pd
import json
import numpy as np


class DatasetLoader:
    def __init__(self, filename: str):
        with open(filename, "r") as f:
            data = json.load(f)

        self.data_frame = pd.DataFrame({
            "ActuatorPositions": data["ActuatorPositions"],
            "ActuatorDeviations": data["ActuatorDeviations"],
            "ActuatorActions": data["ActuatorActions"],
            "ThiknessProfiles": data["ThiknessProfiles"],
            "Averages": data["Averages"]
        })

        self.data_length = len(self.data_frame)
        print(f"DataFrame {self.data_length} satır ile başlatıldı.")

    def get_seq_data(self, seq_len, input_headers, output_headers):
        index = seq_len
        input_datas = []
        output_datas = []

        for row in range(self.data_length-seq_len):

            input_seq_data = []
            output_row = []

            for seq in range(seq_len):

                input_row = []

                for input_header in input_headers:
                    input_row = np.concatenate(
                        [input_row, self.data_frame.loc[row+index-seq, input_header]])

                input_seq_data.append(input_row)

            for outout_header in output_headers:
                output_row = np.concatenate(
                    [output_row, self.data_frame.loc[row+index-seq, outout_header]])
            input_datas.append(input_seq_data)
            output_datas.append(output_row)

        return input_datas, output_datas


