
# 確認目前工作目錄
getwd()

# 查看資料夾內的檔案
list.files()

################################################
## Baseball Salary Prediction
## Step 1: Load Data
################################################

# 讀取棒球員資料
Baseball <- read.table(
  "ball.txt",
  header = TRUE,
  sep = "",
  quote = "\"",
  stringsAsFactors = TRUE
)

# 查看資料維度
dim(Baseball)

# 查看所有欄位名稱
names(Baseball)

# 查看前六筆資料
head(Baseball)

# 查看資料結構
str(Baseball)

# 查看敘述統計
summary(Baseball)

################################################
## Step 2: Data Cleaning
################################################

# 刪除 Name 與 Salary
ball_clean <- Baseball[
  , !(names(Baseball) %in% c("Name", "Salary"))
]

# 查看刪除後的資料維度
dim(ball_clean)

# 查看剩餘欄位
names(ball_clean)

# 確認 Logsalary 仍然存在
head(ball_clean$Logsalary)

# 檢查欄位是否正確
stopifnot(
  !("Name" %in% names(ball_clean)),
  !("Salary" %in% names(ball_clean)),
  "Logsalary" %in% names(ball_clean)
)

# 備份原始檔案
file.copy(
  "ball.txt",
  "ball_original.txt",
  overwrite = FALSE
)

# 將清理後的資料寫回 ball.txt
write.table(
  ball_clean,
  file = "ball.txt",
  row.names = FALSE,
  quote = TRUE,
  sep = "\t"
)

# 重新讀取，確認結果
ball_check <- read.table(
  "ball.txt",
  header = TRUE,
  sep = "",
  stringsAsFactors = TRUE
)

dim(ball_check)
names(ball_check)

################################################
## CART Baseball Salary Prediction
## Step 1: Load Packages
################################################

# 安裝套件（只需要安裝一次）
install.packages(c("rpart", "rpart.plot"))

# 載入套件
library(rpart)
library(rpart.plot)




################################################
## Step 2: Load Baseball Data
################################################

# 讀取資料
Baseball <- read.table(
  "ball.txt",
  header = TRUE,
  sep = "",
  stringsAsFactors = TRUE
)

# 確認資料筆數及欄位
dim(Baseball)

# 查看前六筆資料
head(Baseball)

# 查看欄位名稱
names(Baseball)

# 查看資料結構
str(Baseball)

# 查看薪資對數的統計
summary(Baseball$Logsalary)

################################################
## Step 3: Training and Testing Data
################################################

# 固定亂數種子
set.seed(123)

# 隨機抽取 80% 作為訓練資料
train_index <- sample(
  1:nrow(Baseball),
  size = floor(0.8 * nrow(Baseball))
)

# 建立訓練集
train_data <- Baseball[train_index, ]

# 建立測試集
test_data <- Baseball[-train_index, ]

# 查看資料數量
dim(train_data)
dim(test_data)

################################################
## Step 4: CART Regression Model
################################################

# 建立 CART 迴歸決策樹
set.seed(123)

cart.fit <- rpart(
  Logsalary ~ . - Id,
  data = train_data,
  method = "anova",
  control = rpart.control(
    cp = 0.01,
    minsplit = 10
  )
)

# 查看模型摘要
print(cart.fit)

# 查看決策樹詳細資訊
summary(cart.fit)

# 查看模型複雜度表
rpart::printcp(cart.fit)

# 繪製 CART 決策樹
rpart.plot::rpart.plot(
  cart.fit,
  type = 2,
  extra = 101,
  fallen.leaves = TRUE,
  main = "CART Baseball Salary Prediction"
)

################################################
## Step 6: Variable Importance
################################################

# 查看重要變數
cart.fit$variable.importance

# 繪製變數重要性長條圖
importance.cart <- sort(
  cart.fit$variable.importance,
  decreasing = TRUE
)

barplot(
  importance.cart,
  las = 2,
  col = "steelblue",
  main = "CART Variable Importance",
  ylab = "Importance"
)


################################################
## Step 7: CART Prediction
################################################

# 預測測試資料的 Logsalary
pred.cart <- predict(
  cart.fit,
  newdata = test_data
)

# 比較實際值與預測值
cart.results <- data.frame(
  Actual_Logsalary = test_data$Logsalary,
  Predicted_Logsalary = pred.cart
)

# 查看前 10 筆
head(cart.results, 10)

################################################
## Step 8: Convert Logsalary to Salary
################################################

# 換算回原始薪資
cart.results$Actual_Salary <- exp(
  cart.results$Actual_Logsalary
)

cart.results$Predicted_Salary <- exp(
  cart.results$Predicted_Logsalary
)

# 查看預測結果
head(cart.results, 10)

################################################
## Step 9: Model Evaluation
################################################

# MSE：平均平方誤差
mse.cart <- mean(
  (pred.cart - test_data$Logsalary)^2
)

# RMSE：均方根誤差
rmse.cart <- sqrt(mse.cart)

# MAE：平均絕對誤差
mae.cart <- mean(
  abs(pred.cart - test_data$Logsalary)
)

# 顯示模型評估結果
cart.metrics <- data.frame(
  Model = "CART",
  MSE = mse.cart,
  RMSE = rmse.cart,
  MAE = mae.cart
)

print(cart.metrics)

################################################
## Step 10: Prediction Plot
################################################


################################################
## Step 12: CART Actual vs Predicted Plot
################################################

# 繪製實際值與預測值散佈圖
plot(
  test_data$Logsalary,
  pred.cart,
  main = "CART: Actual vs Predicted Logsalary",
  xlab = "Actual Logsalary",
  ylab = "Predicted Logsalary",
  col = "steelblue",
  pch = 19
)

# 加入理想預測線
abline(
  a = 0,
  b = 1,
  col = "red",
  lwd = 2,
  lty = 2
)

# 加入圖例
legend(
  "topleft",
  legend = c("CART Prediction", "Perfect Prediction"),
  col = c("steelblue", "red"),
  pch = c(19, NA),
  lty = c(NA, 2),
  bty = "n"
)


# 加上理想預測線
abline(
  a = 0,
  b = 1,
  col = "red",
  lty = 2,
  lwd = 2
)

