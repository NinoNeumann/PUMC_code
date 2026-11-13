"""Convert Excel cells to text for the archived R prepare_data() path."""
from pathlib import Path
import pandas as pd

root=Path(__file__).resolve().parents[2]
input_dir=root/'results/cox_es_rcs_newdata_10x5/20261104_v1/input'
for source,target in [
    (input_dir/'20261027v2.xlsx',input_dir/'20261027v2_text.csv'),
    (root/'data/raw/20261023v1-day26.xlsx',input_dir/'previous_text_for_audit.csv'),
]:
    frame=pd.read_excel(source,sheet_name='Sheet1',dtype=str,keep_default_na=False)
    if frame.shape!=(1140,31):raise ValueError((source,frame.shape))
    frame.to_csv(target,index=False,encoding='utf-8')
    print(source.name,target.name,frame.shape)
