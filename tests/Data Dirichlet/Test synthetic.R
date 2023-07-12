library(DirichletReg)
library(readr)

X <- read_csv("synthetic_X.csv")
Y <- read_csv("synthetic_Y.csv")
Z <- read_csv("synthetic_Z.csv")

X$Z <- Z$`1`

Y_DR <- DR_data(Y)

reg <- DirichReg(Y_DR ~ X$`1` + X$`2` | X$`1`, model="alternative")
  
predict(reg)
