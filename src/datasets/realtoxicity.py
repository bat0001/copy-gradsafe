import os
import pandas as pd
from .base import BaseDataset

class RealToxicityDataset(BaseDataset):
    """
    RealToxicityPrompts: columns ['prompt','toxicity']
    """
    def load_train(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, "realtoxicity.csv")
        df = pd.read_csv(path)[["prompt", "toxicity"]]
        return df.sample(frac=0.8, random_state=42).reset_index(drop=True)

    def load_test(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, "realtoxicity.csv")
        df = pd.read_csv(path)[["prompt", "toxicity"]]
        return df.sample(frac=0.2, random_state=42).reset_index(drop=True)

_default_rt = RealToxicityDataset("data/realtoxicity")
load_train = _default_rt.load_train
load       = _default_rt.load_test