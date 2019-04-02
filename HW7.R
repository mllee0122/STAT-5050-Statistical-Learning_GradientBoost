#Statistical Learning_Homework7
#107064522

library(MASS)
library(ISLR)
# Hitters dataset
?Hitters
summary(Hitters)
head(Hitters)
dim(Hitters)
# Remove unknown
unknown = which(is.na(Hitters$Salary))
Hitters = Hitters[-unknown, ]
par(mfrow=c(1,2))
plot(Hitters$Salary)
# take log-transform the salaries
Hitters$Salary = log(Hitters$Salary)
# Digitizing dataset
Hitters.mod = Hitters
Hitters.mod$League = ifelse(Hitters.mod$League=='N', 1, 0) # N <- 1, A <- 0
Hitters.mod$Division = ifelse(Hitters.mod$Division=='W', 1, 0) # W <- 1, E <- 0
Hitters.mod$NewLeague = ifelse(Hitters.mod$NewLeague=='N', 1, 0) # N <- 1, A <- 0
plot(Hitters.mod$Salary)
summary(Hitters.mod$Salary)
cor(Hitters.mod, Hitters.mod$Salary)
# create training and test data
tr = Hitters[1:200, ]
te = Hitters[-(1:200), ]

##### 1. #####
# install.packages("gbm")
library(gbm)
B <- 200
lambda.seq <- seq(1e-3, 1, len = B)
tr_MSE = rep(NA, length(lambda.seq))
te_MSE = rep(NA, length(lambda.seq))

for (i in 1:B){
  md_gbm = gbm(Salary~., data = tr, distribution = "gaussian",
               n.trees = 1000,
               bag.fraction = 0.5,
               shrinkage = lambda.seq[i],
               interaction.depth = 1)
  tr.pred = predict(md_gbm, tr, n.trees=1000)
  te.pred = predict(md_gbm, te, n.trees=1000)
  tr_MSE[i] = mean((tr$Salary - tr.pred)^2)
  te_MSE[i] = mean((te$Salary - te.pred)^2)
}
par(mfrow=c(1,3))
plot(lambda.seq, tr_MSE, type = "l", ylab = "MSE", ylim = c(0, 1))
lines(lambda.seq, te_MSE, col = "red")
legend("topright", legend = c("train MSE","test MSE"), col = c("black","red"), lwd=c(1,1))

##### 2. #####
# install.packages("xgboost")
library(xgboost)
B <- 200
lambda.seq <- seq(1e-3, 1, len = B)
tr.xgb_MSE = rep(NA, length(lambda.seq))
te.xgb_MSE = rep(NA, length(lambda.seq))
tr.mod = Hitters.mod[1:200, ]
te.mod = Hitters.mod[-(1:200), ]
tr.xgb = xgb.DMatrix(data = as.matrix(tr.mod[,-19]), label = tr.mod[,19])
te.xgb = xgb.DMatrix(data = as.matrix(te.mod[,-19]), label = te.mod[,19])

for (i in 1:B){
  param <- list(max_depth = 1,
                eta = lambda.seq[i],
                subsample = 0.5,
                colsample_bytree = 1,
                objective = "reg:linear",
                lambda = 0, alpha = 0)
  obj.xgb <- xgb.train(param, data = tr.xgb, nrounds = 1000, maximize = F)
  tr.xgb.pred = predict(obj.xgb, newdata = as.matrix(tr.mod[,-19]))
  te.xgb.pred = predict(obj.xgb, newdata = as.matrix(te.mod[,-19]))
  tr.xgb_MSE[i] = mean((tr.mod[,19] - tr.xgb.pred)^2)
  te.xgb_MSE[i] = mean((te.mod[,19] - te.xgb.pred)^2)
  
}

plot(lambda.seq, tr.xgb_MSE, type = "l", ylab = "MSE", ylim = c(0, 1))
lines(lambda.seq, te.xgb_MSE, col = "red")
legend("topright", legend = c("train MSE","test MSE"), col = c("black","red"), lwd=c(1,1))

##### 3. #####
for (i in 1:B){
  param <- list(max_depth = 1,
                eta = lambda.seq[i],
                subsample = 0.5,
                colsample_bytree = 0.5,
                objective = "reg:linear",
                lambda = 1, alpha = 1)
  obj.xgb <- xgb.train(param, data = tr.xgb, nrounds = 1000, maximize = F)
  tr.xgb.pred = predict(obj.xgb, newdata = as.matrix(tr.mod[,-19]))
  te.xgb.pred = predict(obj.xgb, newdata = as.matrix(te.mod[,-19]))
  tr.xgb_MSE[i] = mean((tr.mod[,19] - tr.xgb.pred)^2)
  te.xgb_MSE[i] = mean((te.mod[,19] - te.xgb.pred)^2)
  
}

plot(lambda.seq, tr.xgb_MSE, type = "l", ylab = "MSE", ylim = c(0, 1))
lines(lambda.seq, te.xgb_MSE, col = "red")
legend("topright", legend = c("train MSE","test MSE"), col = c("black","red"), lwd=c(1,1))

