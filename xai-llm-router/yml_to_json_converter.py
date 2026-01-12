import json, yaml

with open("methods.yaml", "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

with open("methods.json", "w", encoding="utf-8") as f:
    json.dump({"toolkits": data["toolkits"]}, f, ensure_ascii=False, indent=2)

print("Wrote methods.json")