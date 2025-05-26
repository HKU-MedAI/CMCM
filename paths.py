from pathlib import Path

ROOT_PATH = Path(__file__).parent

MIMIC4_DATA_DIR = ROOT_PATH / "data_mimic4/"

MIMIC3_DATA_DIR = ROOT_PATH / "data_mimic3/"

CXR_DATA_DIR = ROOT_PATH / "data_mimic4/mimic-cxr/"

MIMIC4_READM_NORMALIZER_PATH = ROOT_PATH / "data_mimic4/readm_ts.normalizer"

MIMIC4_IHM_NORMALIZER_PATH = ROOT_PATH / "data_mimic4/readm_ts.normalizer"

MIMIC3_READM_NORMALIZER_PATH = ROOT_PATH / "data_mimic3/readm_ts.normalizer"

MIMIC3_IHM_NORMALIZER_PATH = ROOT_PATH / "data_mimic3/readm_ts.normalizer"