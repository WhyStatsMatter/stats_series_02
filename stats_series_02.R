# Load necessary libraries
library(MASS)
library(lmtest)
library(caret)
library(boot)
library(glmnet)

# Day 1 and Day 2
set.seed(42)
X1 <- runif(100)
epsilon <- rnorm(100, mean = 0, sd = 1) # Noise term
y <- 3 * X1 + 2 + epsilon # Adding noise to the linear relationship

# Day 3
X <- cbind(1, X1)
beta <- solve(t(X) %*% X) %*% t(X) %*% y
print(paste("Beta values: ", beta))

# Day 4
model <- lm(y ~ X)
summary(model)

# Day 5
X2 <- runif(100)
y <- 3 * X1 + 2 * X2 + 2 + epsilon # Adding another independent variable X2
model <- lm(y ~ X + X2)

# Day 6
plot(model$fitted.values, rstandard(model))
title('Residual Plot')

# Day 7
y_pred <- predict(model, newdata=data.frame(X, X2))
print(paste('RMSE:', sqrt(mean((y - y_pred)^2))))
print(paste('MAE:', mean(abs(y - y_pred))))
print(paste('R-squared:', summary(model)$r.squared))

# Day 8-9 skipped for brevity, implement transformations as needed

# Day 10
dwtest(model)

# Day 11
X <- as.matrix(cbind(X, X2))
y <- as.matrix(y)
ridge <- glmnet(X, y, alpha = 0)
lasso <- glmnet(X, y, alpha = 1)

# Day 12-13
control <- trainControl(method="cv", number=5)
model_cv <- train(X, y, method="glmnet", trControl=control)
print(paste('Cross-validation score:', max(model_cv$results$Rsquared)))
