# %%
import matplotlib.pyplot as plt
import pandas as pd

# %%
data = pd.read_csv("./data/PubMed_Timeline_Results_by_Year.csv")
data.head()

# %%
year = data["Year"].tolist()
count = data["Count"].tolist()
year[1]

# %%
plt.plot(year[2:], count[2:])
plt.xlabel("Year")
plt.ylabel("Number of publications")
plt.xticks(list(range(1781, 2022, 30)))
plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
plt.show()
