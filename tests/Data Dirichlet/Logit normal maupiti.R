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

for (i in 1:2301){
  unique_values <- unique(W_mat[i,])
  if (length(unique_values)==1){
    if (unique_values==1){
      W_mat[i,] <- W_mat[i,]/2301
    }
  }
}


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



n = 2301

mse_values <- numeric()
for (i in 1:ncol(Y)) {
  # Compute MSE for the current column
  mse_values[i] <- MSE(pred_final[, i], Y[, i])
}


# R2 = 0.4200099
mean(diag(cor(pred_final, Y) ^ 2))
# RMSE = 0.2984037
sqrt(mean(mse_values))
# cross-entropy = -1.125738
sum(Y * log(pred_final)) / n
# cos similarity = 0.804215
cos_similarity(Y, pred_final)
# RMSE_A = 6.590471
rmse_aitchison(pred_final, Y)
# AIC ?
pred_final[pred_final < 10e-05] <- 10e-05
rmse_aitchison(pred_final, Y) # RMSE_A = 4.328031
