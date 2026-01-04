from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    train_path: Path = Path("data/train.xlsx")
    test_path: Path = Path("data/test.xlsx")
    target_col: str = "Diagnosis"

    # Deney ayarları
    methods: tuple = ("none", "relief", "chi2", "infogain", "doa")
    ks: tuple = (None, 5, 10, 15, 20)  # None => All

    # Çıktı
    output_xlsx: Path = Path("feature_selection_results.xlsx")
