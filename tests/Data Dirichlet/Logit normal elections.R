library(readr)
library(spdep)
library(spatialreg)
library(MLmetrics)

lr <- function(x) {
  log(x / rowMeans(x))
}

lr_inv <- function(x) {
  exp(x) / rowSums(exp(x))
}

normalize <- function(x) {
  (x - mean(x)) / sd(x)
}

Y_occitanie <- read.csv("occitanie/Y_occitanie.csv", sep=";")
X_occitanie <- read.csv("occitanie/X_occitanie_bis.csv", sep=";")
X_occitanie <- X_occitanie[, -1]
W_mat <- as.matrix(read.csv("occitanie/W_elections_5nn.csv", sep=" ", header=FALSE))
#W_mat <- as.matrix(read.csv("occitanie/W_elections_distance.csv", sep=",", header=FALSE))
W <- mat2listw(W_mat, style="W")
X_scaled <- as.data.frame(lapply(X_occitanie, normalize))

# we drop columns age_65 and foreign that cause aliased variables error
X_scaled <- subset(X_scaled, select = -c(`age_65`, `foreign`))

lr_Y <- lr(Y_occitanie)

lm_list <- list()
for (i in seq_along(lr_Y)) {
  # Fit a linear model for response variable i using the features in X
  lm_model <- lagsarlm(lr_Y[, i] ~ . - 1, data = X_scaled, listw=W)
  
  # Store the linear model in the list
  lm_list[[i]] <- lm_model
}

Xbeta1 <- as.matrix(X_scaled) %*% as.matrix(lm_list[[1]]$coefficients)
Xbeta2 <- as.matrix(X_scaled) %*% as.matrix(lm_list[[2]]$coefficients)
Xbeta3 <- as.matrix(X_scaled) %*% as.matrix(lm_list[[3]]$coefficients)

rho = mean(c(lm_list[[1]]$rho, lm_list[[2]]$rho, lm_list[[3]]$rho))
Minv = solve( diag(nrow(W_mat)) - rho*W_mat )


pred <- data.frame(left = Minv %*% as.matrix(Xbeta1),
                   right = Minv %*% as.matrix(Xbeta2),
                   other = Minv %*% as.matrix(Xbeta3))

pred_final <- lr_inv(pred)


cos_similarity <- function(x1, x2) {
  mean(sapply(seq_len(nrow(x1)), function(i) {
    sum(x1[i,] * x2[i,]) / (sqrt(sum(x1[i,]^2)) * sqrt(sum(x2[i,]^2)))
  }))
}

rmse_aitchison <- function(x1, x2) {
  n <- nrow(x1)
  D <- ncol(x1)
  
  log_x1 <- log(x1)
  log_x2 <- log(x2)
  
  row_rmse <- numeric(n)
  
  for (i in 1:n) {
    # Log-ratio matrices for row i
    v1 <- as.numeric(log_x1[i, ])
    v2 <- as.numeric(log_x2[i, ])
    
    diff1 <- outer(v1, v1, "-")
    diff2 <- outer(v2, v2, "-")
    
    delta <- diff1 - diff2
    squared <- delta^2
    row_rmse[i] <- sqrt(sum(squared) / (2 * D))
  }
  
  return(mean(row_rmse))
}




n = 207

mse_values <- numeric()
for (i in 1:ncol(Y_occitanie)) {
  # Compute MSE for the current column
  mse_values[i] <- MSE(pred_final[, i], Y_occitanie[, i])
}

## contiguity with 5 neighbours
# R2 = 0.4141665
mean(diag(cor(pred_final, Y_occitanie) ^ 2))
# RMSE = 0.1145348
sqrt(mean(mse_values))
# cross-entropy = -1.075672
sum(Y_occitanie * log(pred_final)) / n
# cos similarity = 0.9511475
cos_similarity(Y_occitanie, pred_final)
# RMSE_A = 0.5497698
rmse_aitchison(pred_final, Y_occitanie)
# AIC ?

## distance
# R2 = 0.4342761
mean(diag(cor(pred_final, Y_occitanie) ^ 2))
# RMSE = 0.1162009
sqrt(mean(mse_values))
# cross-entropy = -1.076869
sum(Y_occitanie * log(pred_final)) / n
# cos similarity = 0.9497338
cos_similarity(Y_occitanie, pred_final)
# RMSE_A = 0.549894
rmse_aitchison(pred_final, Y_occitanie)
# AIC ?



# Adjusted R squared for Dirichlet
n = 207
k = 76 #non-spatial
k = 77 #spatial
R = 0.602
a = (n-1)/(n-k)
1 - (1-R)*a

# Adjusted R squared for multinomial
n = 207
k = 75 #non-spatial
k = 76 #spatial
R = 0.608
a = (n-1)/(n-k)
1 - (1-R)*a

# Adjusted R squared for logit normal
n = 207
k = 19 #spatial
R = 0.434
a = (n-1)/(n-k)
1 - (1-R)*a