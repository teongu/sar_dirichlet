library(readr)
library(spdep)
library(spatialreg)
library(MLmetrics)

setwd('C:/Users/tnguyen001/Documents/GitHub/sar_dirichlet/tests/Data Dirichlet')

lr <- function(x) {
  log(x / rowMeans(x))
}

lr_inv <- function(x) {
  exp(x) / rowSums(exp(x))
}


data_arctic <- read_csv("ArcticLake.csv")
X <- as.data.frame(data_arctic$depth)
Y <- as.data.frame(data_arctic[, c('sand', 'silt', 'clay')])
#W_mat <- as.matrix(read_csv("W_arctic_cont.csv"))
#W_mat <- as.matrix(read_csv("W_arctic_dist.csv"))


lr_Y <- lr(Y)

colnames(lr_Y) <- c("sand_lr", "silt_lr", "clay_lr")

data_arctic <- cbind(data_arctic, lr_Y)


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



#####

aitchison_mean <- function(y) {
  # Geometric mean of each component across samples
  geometric_means <- exp(colMeans(log(y)))
  # Closure operation (normalize to sum to 1)
  return(geometric_means / sum(geometric_means))
}


aitchison_inner_product <- function(x1, x2) {
  J <- length(x1)
  log_x1 <- log(x1)
  log_x2 <- log(x2)
  
  # Compute the double sum over j and j'
  inner_sum <- 0
  for (j in 1:J) {
    for (jp in 1:J) {
      log_ratio_x1 <- log(x1[j] / x1[jp])
      log_ratio_x2 <- log(x2[j] / x2[jp])
      inner_sum <- inner_sum + log_ratio_x1 * log_ratio_x2
    }
  }
  
  # Divide by 2J as per the convention in the paper
  return(inner_sum / (2 * J))
}


aitchison_norm <- function(x) {
  return(sqrt(aitchison_inner_product(x, x)))
}


aitchison_distance <- function(x1, x2) {
  J <- length(x1)
  log_x1 <- log(x1)
  log_x2 <- log(x2)
  
  # Compute the double sum over j and j'
  inner_sum <- 0
  for (j in 1:J) {
    for (jp in 1:J) {
      log_ratio_x1 <- log(x1[j] / x1[jp])
      log_ratio_x2 <- log(x2[j] / x2[jp])
      inner_sum <- inner_sum + (log_ratio_x1 - log_ratio_x2)^2
    }
  }
  
  # Divide by 2J and take square root
  return(sqrt(inner_sum / (2 * J)))
}


r2_aitchison_adjusted <- function(y_true, y_pred, n_params) {
  if (is.vector(y_true)) y_true <- t(as.matrix(y_true))
  if (is.vector(y_pred)) y_pred <- t(as.matrix(y_pred))
  
  n <- nrow(y_true)
  J <- ncol(y_true)
  
  y_mean <- exp(colMeans(log(y_true))) / sum(exp(colMeans(log(y_true))))
  
  tss <- 0
  rss <- 0
  for (i in 1:n) {
    tss <- tss + sum((log(y_true[i,]) - log(y_mean))^2) / J
    rss <- rss + sum((log(y_true[i,]) - log(y_pred[i,]))^2) / J
  }
  
  r2 <- if (tss > 0) 1 - (rss / tss) else 0
  if (n <= n_params || n <= 1) return(r2)
  1 - (1 - r2) * (n - 1) / (n - n_params)
}

cosine_similarity_aitchison <- function(y_true, y_pred) {
  if (is.vector(y_true)) y_true <- t(as.matrix(y_true))
  if (is.vector(y_pred)) y_pred <- t(as.matrix(y_pred))
  
  n <- nrow(y_true)
  J <- ncol(y_true)
  
  total_cosine <- 0
  for (i in 1:n) {
    log_true <- log(y_true[i,])
    log_pred <- log(y_pred[i,])
    
    inner_sum <- 0
    norm_true_sum <- 0
    norm_pred_sum <- 0
    for (j in 1:J) {
      for (jp in 1:J) {
        lr_true <- log_true[j] - log_true[jp]
        lr_pred <- log_pred[j] - log_pred[jp]
        inner_sum <- inner_sum + lr_true * lr_pred
        norm_true_sum <- norm_true_sum + lr_true^2
        norm_pred_sum <- norm_pred_sum + lr_pred^2
      }
    }
    
    inner <- inner_sum / (2 * J)
    norm_true <- sqrt(norm_true_sum / (2 * J))
    norm_pred <- sqrt(norm_pred_sum / (2 * J))
    
    if (norm_true > 0 && norm_pred > 0) {
      total_cosine <- total_cosine + inner / (norm_true * norm_pred)
    }
  }
  
  total_cosine / n
}





#####


W_mat <- as.matrix(read_csv("W_arctic_cont_20.csv"))
#W_mat <- as.matrix(read_csv("W_arctic_dist_15.csv"))
W <- mat2listw(W_mat, style="W")
data_arctic$depth_scaled <- (data_arctic$depth-mean(data_arctic$depth))/sd(data_arctic$depth)
data_arctic$depth_scaled_square <- data_arctic$depth_scaled**2




list_pred <- matrix(nrow = 0, ncol = 3)
list_r2 <- cbind()
list_rmse <- cbind()
list_crossentropy <- cbind()
list_similarity <- cbind()
list_rmse_a <- cbind()
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
  #list_r2 <- rbind(list_r2, cor(as.numeric(pred_i), as.numeric(Y[i,])) ^ 2)
  list_r2 <- rbind(list_r2, r2_aitchison_adjusted(Y[i,], pred_i, n_params = 3))
  list_rmse <- rbind(list_rmse, mean(as.numeric((pred_i - Y[i,])^2)))
  list_crossentropy <- rbind(list_crossentropy, sum(Y[i,] * log(pred_i)))
  #list_similarity <- rbind(list_similarity, cos_similarity(Y[i,], pred_i))
  list_similarity <- rbind(list_similarity, 
                           cosine_similarity_aitchison(
                             matrix(as.numeric(Y[i,]), nrow=1), 
                             matrix(as.numeric(pred_i), nrow=1)
                           )
  )
  list_rmse_a <- rbind(list_rmse_a, rmse_aitchison(Y[i,], pred_i))
}



training_data <- data_arctic[-35, ]
W_mat_training <- W_mat[-35, -35]
training_W <- mat2listw(W_mat_training, style="W")

mod1 <- lagsarlm(sand_lr ~ depth_scaled, data=training_data, listw=training_W)
mod2 <- lagsarlm(silt_lr ~ depth_scaled, data=training_data, listw=training_W)
mod3 <- lagsarlm(clay_lr ~ depth_scaled, data=training_data, listw=training_W)

Xbeta1 <- mod1$coefficients[1]*data_arctic$depth_scaled
Xbeta2 <- mod2$coefficients[1]*data_arctic$depth_scaled
Xbeta3 <- mod3$coefficients[1]*data_arctic$depth_scaled

rho = mean(c(mod1$rho, mod2$rho, mod3$rho))
Minv = solve( diag(nrow(W_mat)) - rho*W_mat )

pred <- data.frame(sand = Minv %*% Xbeta1,
                   silt = Minv %*% Xbeta2,
                   clay = Minv %*% Xbeta3)

pred_i <- lr_inv(pred[35,])


r2_aitchison_adjusted(Y[35,], pred_i, n_params = 3)
cor(as.numeric(pred_i), as.numeric(Y[i,]))^2

### ORDER 1 ###

## Contiguity
# R2 = 0.6728408 (0.2677946)
mean(list_r2)
sqrt(var(list_r2))
# cross-entropy = -1.072055 (0.1587036)
mean(list_crossentropy)
sqrt(var(list_crossentropy))
# cos similarity = 0.8822487 (0.09681309)
mean(list_similarity)
sqrt(var(list_similarity))
# RMSE_A = 1.418186 (0.6872034)
mean(list_rmse_a)
sqrt(var(list_rmse_a))

# R2 aitchison
mean(list_r2)
sqrt(var(list_r2))
# Cos similarity aitchison
mean(list_similarity)
sqrt(var(list_similarity))

## Distance
# R2 = 0.5911055 (0.2779419)
mean(list_r2)
sqrt(var(list_r2))
# cross-entropy = -1.092532 (0.133295)
mean(list_crossentropy)
sqrt(var(list_crossentropy))
# cos similarity = 0.8720323 (0.08223744)
mean(list_similarity)
sqrt(var(list_similarity))
# RMSE_A = 1.51954 (0.6826123)
mean(list_rmse_a)
sqrt(var(list_rmse_a))




### ORDER 2 ###

## Contiguity
# R2 = 0.7465961 (0.2715427)
mean(list_r2)
sqrt(var(list_r2))
# cross-entropy = -1.065231 (0.214253)
mean(list_crossentropy)
sqrt(var(list_crossentropy))
# cos similarity = 0.8904019 (0.1217014)
mean(list_similarity)
sqrt(var(list_similarity))
# RMSE_A = 1.264805 (0.7310202)
mean(list_rmse_a)
sqrt(var(list_rmse_a))


## Distance
# R2 =0.7327463 (0.2709927)
mean(list_r2)
sqrt(var(list_r2))
# cross-entropy = -1.058003 (0.1948226)
mean(list_crossentropy)
sqrt(var(list_crossentropy))
# cos similarity = 0.8899549 (0.1170117)
mean(list_similarity)
sqrt(var(list_similarity))
# RMSE_A = 1.284566 (0.7297966)
mean(list_rmse_a)
sqrt(var(list_rmse_a))



##### WITH OLD SPATIAL MATRICES #####



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
# RMSE_A = 1.407812 (0.6986515))
mean(list_rmse_a)
sqrt(var(list_rmse_a))

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
# RMSE_A = 1.517462 (0.6786377)
mean(list_rmse_a)
sqrt(var(list_rmse_a))


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
# RMSE_A = 1.403279 (0.7187483))
mean(list_rmse_a)
sqrt(var(list_rmse_a))

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
# RMSE_A = 1.2877 (0.7300942))
mean(list_rmse_a)
sqrt(var(list_rmse_a))



# Adjusted R squared for Dirichlet
n = 39
k = 5 #order 1, non-spatial
k = 6 #order 1, spatial
k = 7 #order 2, non-spatial
k = 8 #order 2, spatial
R = 0.678
a = (n-1)/(n-k)
1 - (1-R)*a
stdR = 0.015
stdR*a

# Adjusted R squared for multinomial
n = 39
k = 4 #order 1, non-spatial
k = 5 #order 1, spatial
k = 6 #order 2, non-spatial
k = 7 #order 2, spatial
R = 0.705
a = (n-1)/(n-k)
1 - (1-R)*a
stdR = 0.016
stdR*a

# Adjusted R squared for logit normal
n = 39
k = 3 #order 1, spatial
k = 4 #order 2, spatial
R = 0.733
a = (n-1)/(n-k)
1 - (1-R)*a
stdR = 0.271
stdR*a



# Adjusted Rsquared for the synthetic datasets
n = 1000
k = 6 #non-spatial
k = 7 #spatial
R = 0.9991
a = (n-1)/(n-k)
1 - (1-R)*a
stdR = 0.0027
stdR*a




##########


