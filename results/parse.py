from pathlib import Path
import json
import pandas as pd


model_records = {}
for path in sorted(Path(".").rglob("*.json"), key=lambda p: p.stem[-2:]):
    record = json.load(path.open())
    model, pretrained = record["model"], record["pretrained"]
    metrics = {
        f"{record['dataset']} {metric} {record['language']}": score
        for metric, score in record["metrics"].items()
    }
    if model not in model_records:
        model_records[model] = {}
    if pretrained not in model_records[model]:
        model_records[model][pretrained] = {"model": model, "version": pretrained}
    model_records[model][pretrained] |= metrics

df = pd.DataFrame.from_records(
    [record for pretrained in model_records.values() for record in pretrained.values()]
)
df.to_csv("mclip_retrieval_results.tsv", index=False, sep="\t")
