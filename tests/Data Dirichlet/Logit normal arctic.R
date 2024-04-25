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


data_arctic <- read_csv("ArcticLake.csv")
X <- as.data.frame(data_arctic$depth)
Y <- as.data.frame(data_arctic[, c('sand', 'silt', 'clay')])
W_mat <- as.matrix(read_csv("W_arctic_cont.csv"))
W <- mat2listw(W_mat, style="W")
data_arctic$depth_scaled <- (data_arctic$depth-mean(data_arctic$depth))/sd(data_arctic$depth)

lr_Y <- lr(Y)

colnames(lr_Y) <- c("sand_lr", "silt_lr", "clay_lr")

data_arctic <- cbind(data_arctic, lr_Y)

list_pred <- matrix(nrow = 0, ncol = 3)
# Iterate over each row index
for (i in 1:nrow(data_arctic)) {
  # Exclude the i-th row
  training_data <- data_arctic[-i, ]
  W_mat_training <- W_mat[-i, -i]
  training_W <- mat2listw(W_mat_training, style="W")
  
  mod1 <- lagsarlm(sand_lr ~ depth_scaled - 1, data=training_data, listw=training_W)
  mod2 <- lagsarlm(silt_lr ~ depth_scaled - 1, data=training_data, listw=training_W)
  mod3 <- lagsarlm(clay_lr ~ depth_scaled - 1, data=training_data, listw=training_W)
  
  Xbeta1 <- mod1$coefficients[1]*data_arctic$depth_scaled
  Xbeta2 <- mod2$coefficients[1]*data_arctic$depth_scaled
  Xbeta3 <- mod3$coefficients[1]*data_arctic$depth_scaled
  
  rho = mean(c(mod1$rho, mod2$rho, mod3$rho))
  Minv = solve( diag(nrow(W_mat)) - rho*W_mat )
  
  pred <- data.frame(sand = Minv %*% Xbeta1,
                     silt = Minv %*% Xbeta2,
                     clay = Minv %*% Xbeta3)
  
  list_pred <- rbind(list_pred, pred[i,])
}

pred_final <- lr_inv(list_pred)

cos_similarity <- function(x1, x2) {
  mean(sapply(seq_len(nrow(x1)), function(i) {
    sum(x1[i,] * x2[i,]) / (sqrt(sum(x1[i,]^2)) * sqrt(sum(x2[i,]^2)))
  }))
}

n = 39

mse_values <- numeric()
for (i in 1:ncol(Y)) {
  # Compute MSE for the current column
  mse_values[i] <- MSE(pred_final[, i], Y[, i])
}


# R2 = 0.4305389
mean(diag(cor(pred_final, Y) ^ 2))
# RMSE = 0.2266367
sqrt(mean(mse_values))
# cross-entropy = -1.105847
sum(Y * log(pred_final)) / n
# cos similarity = 0.842516
cos_similarity(Y, pred_final)
