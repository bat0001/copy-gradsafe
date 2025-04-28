import os
import pandas as pd
from .base import BaseDataset

class XSTestDataset(BaseDataset):
    """
    XSTest: colonnes ['prompt','type']
    - test only: xstest_v2_prompts.csv
    """
    def load_train(self):
        raise NotImplementedError("XSTest doesn't have split train.")

    def load_test(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, "xstest_v2_prompts.csv")
        return pd.read_csv(path)[["prompt","type"]]

# helper
_default_xs = XSTestDataset("data/xstest")
load = _default_xs.load_test