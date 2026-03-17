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

pred_final_save <- pred_final

pred_final[pred_final < 0.0001086484] <- 0.0001086484

n = 2301
k = 4 * 16 + 1

mse_values <- numeric()
for (i in 1:ncol(Y)) {
  # Compute MSE for the current column
  mse_values[i] <- MSE(pred_final[, i], Y[, i])
}

### REPLACING ZEROS

# R2 = 0.4199996
mean(diag(cor(pred_final, Y) ^ 2))
# Adjusted R2 = 0.4033985
1 - (1-0.4199996) * (n-1)/(n-k)
# RMSE = 0.2984022
sqrt(mean(mse_values))
# cross-entropy = -0.9997579
sum(Y * log(pred_final)) / n
# cos similarity = 0.8042171
cos_similarity(Y, pred_final)
# RMSE_A = 4.300422
rmse_aitchison(pred_final, Y)

# R2 aitchison = 0.6085225
r2_aitchison_adjusted(Y, pred_final, n_params = k)
# cosine aitchison = 0.7405137
cosine_similarity_aitchison(Y, pred_final)


####

# Without replacing 0

# R2 = 0.4200099
mean(diag(cor(pred_final, Y) ^ 2))
# Adjusted R2 = 0.4034091
1 - (1-0.4200099) * (n-1)/(n-k)
# RMSE = 0.2984037
sqrt(mean(mse_values))
# cross-entropy = -1.125738
sum(Y * log(pred_final)) / n
# cos similarity = 0.804215
cos_similarity(Y, pred_final)
# RMSE_A = 6.590471
rmse_aitchison(pred_final, Y)
# AIC ?
#pred_final[pred_final < 10e-05] <- 10e-05
rmse_aitchison(pred_final, Y) # RMSE_A = 4.328031

r2_aitchison_adjusted(Y, pred_final_save, n_params = k)
