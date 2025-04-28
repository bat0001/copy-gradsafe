from abc import ABC, abstractmethod
import pandas as pd

class BaseDataset(ABC):
    """
    Abstract class.
    load_train() & load_test().
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    @abstractmethod
    def load_train(self) -> pd.DataFrame:
        """Load split train, return pd.DataFrame."""
        pass

    @abstractmethod
    def load_test(self) -> pd.DataFrame:
        """Load split test, return pd.DataFrame."""
        pass