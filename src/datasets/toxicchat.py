import os
import pandas as pd
from .base import BaseDataset

class ToxicChatDataset(BaseDataset):
    """
    ToxicChat: colonnes ['user_input','toxicity'] 
    - train: toxic-chat_annotation_train.csv
    - test : toxic-chat_annotation_test.csv
    """
    def load_train(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, "toxic-chat_annotation_train.csv")
        return pd.read_csv(path)[["user_input","toxicity"]]

    def load_test(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, "toxic-chat_annotation_test.csv")
        return pd.read_csv(path)[["user_input","toxicity"]]

_default_tc = ToxicChatDataset("data/toxic-chat")
load_train = _default_tc.load_train
load       = _default_tc.load_test