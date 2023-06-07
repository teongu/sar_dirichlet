library(nnet)

df = data.frame(Y=Y_maupiti,X_maupiti)
colnames(df)  <- c('Y','x1','x2','x3','x4','x5')

reg <- multinom(Y ~ x1+x2+x3+x4+x5, data=df)

summary(reg)
coefficients(reg)

write.csv(x = t(coefficients(reg)), row.names = FALSE, file = 'coefficients_softmax.csv')
