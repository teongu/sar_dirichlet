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

X_scaled <- read.csv("maupiti_X.csv", sep=",", header=FALSE)
Y <- read.csv("maupiti_Y.csv", sep=",", header=FALSE)
W_mat <- as.matrix(read.csv("maupiti_W_no_zeros.csv", sep=",", header=FALSE))
W <- mat2listw(W_mat, style="W")

# we drop columns because of aliased variables error
#X_scaled <- subset(X_scaled, select = -c(`age_65`, `foreign`))

lr_Y <- lr(Y)

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
Xbeta4 <- as.matrix(X_scaled) %*% as.matrix(lm_list[[4]]$coefficients)

rho = mean(c(lm_list[[1]]$rho, lm_list[[2]]$rho, lm_list[[3]]$rho, lm_list[[4]]$rho))
Minv = solve( diag(nrow(W_mat)) - rho*W_mat )

pred <- data.frame(c1 = Minv %*% as.matrix(Xbeta1),
                   c2 = Minv %*% as.matrix(Xbeta2),
                   c3 = Minv %*% as.matrix(Xbeta3),
                   c4 = Minv %*% as.matrix(Xbeta4))

pred_final <- lr_inv(pred)


cos_similarity <- function(x1, x2) {
  mean(sapply(seq_len(nrow(x1)), function(i) {
    sum(x1[i,] * x2[i,]) / (sqrt(sum(x1[i,]^2)) * sqrt(sum(x2[i,]^2)))
  }))
}

n = 2301

mse_values <- numeric()
for (i in 1:ncol(Y)) {
  # Compute MSE for the current column
  mse_values[i] <- MSE(pred_final[, i], Y[, i])
}


# R2 = 0.4147603
mean(diag(cor(pred_final, Y) ^ 2))
# RMSE = 0.3003778
sqrt(mean(mse_values))
# cross-entropy = -1.667296
sum(Y * log(pred_final)) / n
# cos similarity = 0.8031862
cos_similarity(Y, pred_final)
# AIC ?
