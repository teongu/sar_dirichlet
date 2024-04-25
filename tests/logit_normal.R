library(readr)
library(spdep)
library(spatialreg)

lr <- function(x) {
  log(x / rowMeans(x))
}

lr_inv <- function(x) {
  exp(x) / rowSums(exp(x))
}

logit_normal_model <- function(inputData, n_features, W_mat) {
  
  data <- inputData[,1:n_features]
  data <- cbind(data, lr(inputData[,(n_features+1):ncol(inputData)]))
  W <- mat2listw(W_mat, style="W")
  
  list_rhos <- c()
  list_coefficients <- c()
  
  for (i in (n_features+1):ncol(inputData)){
    formula <- as.formula(paste0(colnames(data)[i], " ~ ", paste0(names(data)[1:n_features], collapse = " + "), " - 1"))
    
    mod <- lagsarlm(formula, data=data,listw=W)
    list_rhos <- append(list_rhos,mod$rho)
    list_coefficients <- append(list_coefficients,mod$coefficients)
  }
  
  return(list(list_rhos, list_coefficients))
}
