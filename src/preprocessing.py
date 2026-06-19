import pandas as pd

dfDF = pd.read_excel("~/SpectroscopyResearch/data/2023Dry-FTIR.xlsx")
dfDN = pd.read_excel("~/SpectroscopyResearch/data/2023Dry-NIR.xlsx")
dfWF = pd.read_excel("~/SpectroscopyResearch/data/2023Wet-FTIR.xlsx")
dfWN = pd.read_excel("~/SpectroscopyResearch/data/2023Wet-NIR.xlsx")

dfDF.info()
dfDN.info()
dfWF.info()
dfWN.info()