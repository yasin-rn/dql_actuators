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

    def get_nn_data(self, batch_size, input_headers, output_headers):
        batched_data = []

        if self.data_frame.empty:
            print("Uyarı: DataFrame boş. Sinir ağı verisi oluşturulamıyor.")
            return batched_data

        all_provided_headers = input_headers + output_headers
        for header in all_provided_headers:
            if header not in self.data_frame.columns:
                print(
                    f"Hata: '{header}' başlığı DataFrame sütunlarında bulunamadı: {self.data_frame.columns.tolist()}. İşlem durduruldu.")
                return batched_data

        for i in range(0, self.data_length, batch_size):
            current_batch_input_rows = []
            current_batch_output_rows = []

            end_index = min(i + batch_size, self.data_length)

            for row_idx in range(i, end_index):
                input_features_for_single_sample = []
                for header in input_headers:
                    value = self.data_frame.loc[row_idx, header]
                    if isinstance(value, list):
                        input_features_for_single_sample.extend(value)
                    elif pd.isna(value):
                        print(
                            f"Uyarı: Girdi başlığı '{header}', satır {row_idx} için NaN değeri içeriyor. 0.0 kullanılacak.")
                        input_features_for_single_sample.append(0.0)
                    else:
                        input_features_for_single_sample.append(value)
                current_batch_input_rows.append(
                    input_features_for_single_sample)

                output_targets_for_single_sample = []
                for header in output_headers:
                    value = self.data_frame.loc[row_idx, header]
                    if isinstance(value, list):
                        output_targets_for_single_sample.extend(value)
                    elif pd.isna(value):
                        print(
                            f"Uyarı: Çıktı başlığı '{header}', satır {row_idx} için NaN değeri içeriyor. 0.0 kullanılacak.")
                        output_targets_for_single_sample.append(0.0)
                    else:
                        output_targets_for_single_sample.append(value)
                current_batch_output_rows.append(
                    output_targets_for_single_sample)

            if current_batch_input_rows and current_batch_output_rows:
                try:
                    batch_X = np.array(
                        current_batch_input_rows, dtype=np.float32)
                    batch_Y = np.array(
                        current_batch_output_rows, dtype=np.float32)
                    batched_data.append((batch_X, batch_Y))
                except ValueError as e:
                    print(
                        f"Hata: İndeks {i} ile başlayan grup NumPy dizisine dönüştürülürken hata oluştu: {e}")
                    print(
                        "Bu durum, gruptaki örnekler arasında özellik/hedef sayılarının tutarsız olmasından kaynaklanabilir.")
            elif (end_index - i > 0):
                print(
                    f"Uyarı: İndeks {i} ile {end_index} arasındaki grup boş girdi/çıktı listeleriyle sonuçlandı.")

        return batched_data

