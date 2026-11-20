
################################################
## RF Baseball Salary Prediction
################################################

library(randomForest)

# 固定亂數種子
set.seed(123)

# 建立隨機森林迴歸模型
rf.fit <- randomForest(
  Logsalary ~ . - Id,
  data = train_data,
  ntree = 500,
  mtry = 4,
  nodesize = 5,
  importance = TRUE
)

# 查看模型結果
print(rf.fit)

# 查看模型摘要
summary(rf.fit)

################################################
## RF Prediction
################################################

# 使用 RF 預測測試集
pred.rf <- predict(
  rf.fit,
  newdata = test_data
)

# 查看前 6 筆預測值
head(pred.rf)

# 確認預測筆數
length(pred.rf)

################################################
## RF Model Evaluation
################################################

# 計算 MSE
mse.rf <- mean(
  (test_data$Logsalary - pred.rf)^2
)

# 計算 RMSE
rmse.rf <- sqrt(mse.rf)

# 計算 MAE
mae.rf <- mean(
  abs(test_data$Logsalary - pred.rf)
)

# 顯示結果
cat("RF MSE =", mse.rf, "\n")
cat("RF RMSE =", rmse.rf, "\n")
cat("RF MAE =", mae.rf, "\n")

################################################
## CART vs RF Comparison
################################################

# 建立模型比較表
model.comparison <- data.frame(
  Model = c("CART", "RF"),
  MSE = c(mse.cart, mse.rf),
  RMSE = c(rmse.cart, rmse.rf)
)

# 顯示比較結果
print(model.comparison)

# 找出 RMSE 較低的模型
best.model <- model.comparison$Model[
  which.min(model.comparison$RMSE)
]

cat("目前 RMSE 較低的模型：", best.model, "\n")

################################################
## RF Actual vs Predicted Logsalary
################################################

# 繪製 RF 預測散佈圖
plot(
  test_data$Logsalary,
  pred.rf,
  main = "RF: Actual vs Predicted Logsalary",
  xlab = "Actual Logsalary",
  ylab = "Predicted Logsalary",
  col = "darkgreen",
  pch = 19
)

# 加入理想預測線
abline(
  a = 0,
  b = 1,
  col = "red",
  lty = 2,
  lwd = 2
)

################################################
## RF Variable Importance
################################################

# 查看變數重要性數值
importance(rf.fit)

# 繪製變數重要性圖
varImpPlot(
  rf.fit,
  main = "RF Variable Importance"
)

################################################
## RF Error vs Number of Trees
################################################

plot(
  rf.fit,
  main = "RF: Error vs Number of Trees",
  xlab = "Number of Trees",
  ylab = "OOB Mean Squared Error",
  col = "blue",
  lwd = 2
)

################################################
## CART vs RF RMSE Comparison
################################################

# 建立模型比較表
model.comparison <- data.frame(
  Model = c("CART", "RF"),
  MSE = c(0.1904637, 0.1141903),
  RMSE = c(0.4364215, 0.3379205)
)

# 繪製 RMSE 比較長條圖
barplot(
  model.comparison$RMSE,
  names.arg = model.comparison$Model,
  col = c("steelblue", "darkgreen"),
  main = "CART vs RF: RMSE Comparison",
  xlab = "Machine Learning Model",
  ylab = "RMSE",
  ylim = c(0, 0.55)
)

# 顯示比較表
print(model.comparison)

################################################
## Save RF Prediction Plot
################################################

png(
  "RF_Actual_vs_Predicted.png",
  width = 1000,
  height = 700,
  res = 120
)

plot(
  test_data$Logsalary,
  pred.rf,
  main = "RF: Actual vs Predicted Logsalary",
  xlab = "Actual Logsalary",
  ylab = "Predicted Logsalary",
  col = "darkgreen",
  pch = 19
)

abline(0, 1, col = "red", lty = 2, lwd = 2)

dev.off()

################################################
## Save Model Comparison Plot
################################################

png(
  "CART_RF_RMSE_Comparison.png",
  width = 1000,
  height = 700,
  res = 120
)

barplot(
  model.comparison$RMSE,
  names.arg = model.comparison$Model,
  col = c("steelblue", "darkgreen"),
  main = "CART vs RF: RMSE Comparison",
  xlab = "Model",
  ylab = "RMSE",
  ylim = c(0, 0.55)
)

dev.off()
