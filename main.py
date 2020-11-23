import pandas as pd

from dwmp.model import Model
from dwmp.model import VarType


base_df = pd.read_csv("./mvmt.csv")

model = Model("Test 1")
model.meta("Index", list(range(len(base_df))))
model.dependent("sales", base_df["ACTUAL"])

model.variable("price", base_df["price"], VarType.Log, "BASE", "PRICE")
model.variable("nupcs", base_df["nupcs"], VarType.Log, "BASE", "DISTRIBUTION")
model.variable("wk", base_df["wk"], VarType.Unit, "BASE", "TREND")
model.variable("display", base_df["display"], VarType.Unit, "INC", "DISPLAY")
model.variable("feature", base_df["feature"], VarType.Unit, "INC", "FEATURE")

model.run()


print(model.results["summary"])
model.decomp_df.to_csv("./testdecomp.csv")