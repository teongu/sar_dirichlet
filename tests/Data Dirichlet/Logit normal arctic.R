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
#W_mat <- as.matrix(read_csv("W_arctic_dist.csv"))
W <- mat2listw(W_mat, style="W")
data_arctic$depth_scaled <- (data_arctic$depth-mean(data_arctic$depth))/sd(data_arctic$depth)
data_arctic$depth_scaled_square <- data_arctic$depth_scaled**2

lr_Y <- lr(Y)

colnames(lr_Y) <- c("sand_lr", "silt_lr", "clay_lr")

data_arctic <- cbind(data_arctic, lr_Y)


cos_similarity <- function(x1, x2) {
  mean(sapply(seq_len(nrow(x1)), function(i) {
    sum(x1[i,] * x2[i,]) / (sqrt(sum(x1[i,]^2)) * sqrt(sum(x2[i,]^2)))
  }))
}

list_pred <- matrix(nrow = 0, ncol = 3)
list_r2 <- cbind()
list_rmse <- cbind()
list_crossentropy <- cbind()
list_similarity <- cbind()
# Iterate over each row index
for (i in 1:nrow(data_arctic)) {
  # Exclude the i-th row
  training_data <- data_arctic[-i, ]
  W_mat_training <- W_mat[-i, -i]
  training_W <- mat2listw(W_mat_training, style="W")
  
  mod1 <- lagsarlm(sand_lr ~ depth_scaled, data=training_data, listw=training_W)
  mod2 <- lagsarlm(silt_lr ~ depth_scaled, data=training_data, listw=training_W)
  mod3 <- lagsarlm(clay_lr ~ depth_scaled, data=training_data, listw=training_W)
  
  #mod1 <- lagsarlm(sand_lr ~ depth_scaled + depth_scaled_square, data=training_data, listw=training_W)
  #mod2 <- lagsarlm(silt_lr ~ depth_scaled + depth_scaled_square, data=training_data, listw=training_W)
  #mod3 <- lagsarlm(clay_lr ~ depth_scaled + depth_scaled_square, data=training_data, listw=training_W)
  
  Xbeta1 <- mod1$coefficients[1]*data_arctic$depth_scaled
  Xbeta2 <- mod2$coefficients[1]*data_arctic$depth_scaled
  Xbeta3 <- mod3$coefficients[1]*data_arctic$depth_scaled
  
  rho = mean(c(mod1$rho, mod2$rho, mod3$rho))
  Minv = solve( diag(nrow(W_mat)) - rho*W_mat )
  
  pred <- data.frame(sand = Minv %*% Xbeta1,
                     silt = Minv %*% Xbeta2,
                     clay = Minv %*% Xbeta3)
  
  list_pred <- rbind(list_pred, pred[i,])
  pred_i <- lr_inv(pred[i,])
  list_r2 <- rbind(list_r2, cor(as.numeric(pred_i), as.numeric(Y[i,])) ^ 2)
  list_rmse <- rbind(list_rmse, mean(as.numeric((pred_i - Y[i,])^2)))
  list_crossentropy <- rbind(list_crossentropy, sum(Y[i,] * log(pred_i)))
  list_similarity <- rbind(list_similarity, cos_similarity(Y[i,], pred_i))
}


### ORDER 1 ###

## Contiguity
# R2 = 0.6745655 (0.264895)
mean(list_r2)
sqrt(var(list_r2))
# RMSE = 0.03519517 (0.02838537)
mean(list_rmse)
sqrt(var(list_rmse))
# cross-entropy = -1.076936 (0.1641631)
mean(list_crossentropy)
sqrt(var(list_crossentropy))
# cos similarity = 0.8823345 (0.09866024)
mean(list_similarity)
sqrt(var(list_similarity))

## Distance
# R2 = 0.5978643 (0.2779165)
mean(list_r2)
sqrt(var(list_r2))
# RMSE = 0.03709143 (0.02318635)
mean(list_rmse)
sqrt(var(list_rmse))
# cross-entropy = -1.088663 (0.1308945)
mean(list_crossentropy)
sqrt(var(list_crossentropy))
# cos similarity = 0.8739645 (0.08113203)
mean(list_similarity)
sqrt(var(list_similarity))


### ORDER 2 ###

## Contiguity
# R2 = 0.670666 (0.2713688)
mean(list_r2)
sqrt(var(list_r2))
# RMSE = 0.03460107 (0.02985861)
mean(list_rmse)
sqrt(var(list_rmse))
# cross-entropy = -1.081451 (0.1733934)
mean(list_crossentropy)
sqrt(var(list_crossentropy))
# cos similarity = 0.8891295 (0.09781993)
mean(list_similarity)
sqrt(var(list_similarity))

## Distance
# R2 = 0.7290594 (0.2707709)
mean(list_r2)
sqrt(var(list_r2))
# RMSE = 0.03303083 (0.03403696)
mean(list_rmse)
sqrt(var(list_rmse))
# cross-entropy = -1.059102 (0.1939663)
mean(list_crossentropy)
sqrt(var(list_crossentropy))
# cos similarity = 0.8895072 (0.116246)
mean(list_similarity)
sqrt(var(list_similarity))

