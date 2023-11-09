#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 21:13:31 2021

@author: Ghazaleh Safari

International Master- and PhD program in Mathematics
"""


"""Import modules"""
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm

"""Read the data"""
"""Read unemployment rate data"""
df_ur= pd.read_csv("LMUNRRTTDEM156S.csv")

"""Read unfilled job vacancies data"""
df_Ufjv= pd.read_csv("LMJVTTUVDEQ647S.csv")


"""clean the data"""
df_ur.columns= ["Date", "UnemploymentRate"]
df_Ufjv.columns= ["Date", "UnfilledJobVacancies"]


"""Typecast Date Columns to datetime"""
df_ur["Date"]= pd.to_datetime(df_ur["Date"])
df_Ufjv["Date"]= pd.to_datetime(df_Ufjv["Date"])


"""Print data frames"""
print("U.R. data frame\n",df_ur)
print("\n\nU.F.J.V. data frame\n", df_Ufjv)


"""Merge data frames & print it"""
df= pd.merge(df_ur, df_Ufjv, on="Date", how="inner")
#print("Merge data frame\n", df)


"""Compute growth rates"""
df["UR_Growth"]= df["UnemploymentRate"].pct_change(periods=4)
df["UFJV_Growth"]= df["UnfilledJobVacancies"].pct_change(periods=4)
#print(df["UR_Growth"])
#print(df["UnJV_Growth"])

"""Plot data""" 
fig, ax= plt.subplots(nrows=1, ncols=1, squeeze= False)
ax[0][0].scatter(df["UR_Growth"], df["UFJV_Growth"], color="b", s=5)
ax[0][0].set_xlabel("Growth of UnemploymentRate")
ax[0][0].set_ylabel("Growth of UnfilledJobVacancies")
plt.show()

"""Plot regression"""
linear_model= smf.ols(formula= "UFJV_Growth ~ UR_Growth", data=df)
linear_model_fit= linear_model.fit()
print(linear_model_fit.params)
print(linear_model_fit.summary())


""" (Additional analysis, plots, etc)"""
fig1 = sm.graphics.plot_ccpr(linear_model_fit, "UR_Growth")
fig1.tight_layout()
plt.show()

fig2 = sm.graphics.plot_fit(linear_model_fit, "UR_Growth")
fig2.tight_layout()
plt.show()

fig3 = sm.graphics.influence_plot(linear_model_fit, criterion="cooks")
fig3.tight_layout()
plt.show()

fig4 = sm.qqplot(linear_model_fit.resid, line="s")
fig4.tight_layout()
plt.show()

fig5, ax= plt.subplots(nrows=1, ncols=1, squeeze=False)
ax[0][0].hist(df["UR_Growth"], bins=100, color="b", alpha=0.6, 
              rwidth=0.85,label="UR_Growth")
ax[0][0].hist(df["UFJV_Growth"], bins=100, color="g", alpha=0.6, 
              rwidth=0.85,label="UFJV_Growth")
ax[0][0].set_xlabel("x")
ax[0][0].set_ylabel("frequency")
ax[0][0].legend()
plt.tight_layout()
plt.show()
