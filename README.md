# Optimal Trading Strategy in a Mean-Reverting Portfolio

It has been widely observed that many asset prices exhibit mean reversion, for example
foreign exchange rate as well as commodities. In industry, hedge fund managers and
investors often attempt to construct mean-reverting prices by simultaneously taking po-
sitions in two highly correlated or co-moving assets. The advent of exchange-traded funds
(ETFs) has further facilitated this pairs trading approach since some ETFs are designed
to track identical or similar indexes and assets.

Given the price dynamics of some risky assets, one important problem commonly faced
by individual and institutional investors is to determine when to open and close a posi-
tion. While observing the prevailing market prices, a speculative investor can choose to
enter the market immediately or wait for a future opportunity. After completing the rst
trade, the investor will need to decide when is best to close the position.
We would like to better understand how mean-reverting Ornstein-Uhlenbeck (OU) pro-
cesses work, and how they can be used to model trading strategies.

First of all we discretize the continuous Ornstein-Uhlenbeck process and simulate a ran-
dom path checking its mean-reverting property. Then we address the model calibration
using a simple linear regression, trying to infer the model parameters given the simulated
data. In part III we formulate a trading strategy for a certain portfolio of financial assets
which can be modeled by the OU process. Finally we will try to predict the best entry and
exit points of the trade by means of a numerical solution of the associated free boundary
problems or variational inequalities.
