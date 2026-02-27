import pandas as pd

from sklearn.ensemble import IsolationForest

file = open("logs.txt", "r")

logs = []   # empty list to store lines

for line in file:
    logs.append(line.strip())   # remove extra spaces/newline

file.close()


data = pd.DataFrame(logs, columns=["log"])

data["is_error"] = data["log"].str.contains("error", case=False)

data["is_failed"] = data["log"].str.contains("failed", case=False)


data["is_error"] = data["is_error"].astype(int)
data["is_failed"] = data["is_failed"].astype(int)


data["suspicious_score"] = data["is_error"] + data["is_failed"]



model = IsolationForest(contamination=0.3)

model.fit(data[["suspicious_score"]])


data["ai_result"] = model.predict(data[["suspicious_score"]])


print("\nFINAL LOG ANALYSIS:\n")
print(data)