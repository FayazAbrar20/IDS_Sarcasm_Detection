library(tm)
library(e1071)
library(caret)
library(glmnet)

gen <- read.csv("GEN-sarc-notsarc.csv", stringsAsFactors = FALSE)
hyp <- read.csv("HYP-sarc-notsarc.csv", stringsAsFactors = FALSE)
rq <- read.csv("RQ-sarc-notsarc.csv", stringsAsFactors = FALSE)
raw <- rbind(gen, hyp, rq)

data <- data.frame(
  tweet = raw$text,
  sarcastic = ifelse(raw$class == "sarc", 1, 0),
  stringsAsFactors = FALSE
)
data$sarcastic <- as.factor(data$sarcastic)
data <- data[!is.na(data$tweet) & data$tweet != "", ]

data$tweet <- gsub("http\\S+|www\\S+", " ", data$tweet)
data$tweet <- gsub("@\\w+", " ", data$tweet)
data$tweet <- iconv(data$tweet, "UTF-8", "ASCII", sub = " ")

corpus <- VCorpus(VectorSource(data$tweet))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, stripWhitespace)

dtm_raw <- DocumentTermMatrix(corpus, control = list(bounds = list(global = c(3, Inf))))
keep_idx <- which(slam::row_sums(dtm_raw) > 0)
corpus <- corpus[keep_idx]
labels <- data$sarcastic[keep_idx]

dtm_bow <- DocumentTermMatrix(corpus, control = list(bounds = list(global = c(3, Inf))))
dtm_tfidf <- DocumentTermMatrix(corpus, control = list(weighting = weightTfIdf, bounds = list(global = c(3, Inf))))

bow_df <- as.data.frame(as.matrix(dtm_bow))
tfidf_df <- as.data.frame(as.matrix(dtm_tfidf))
bow_df$sarcastic <- labels
tfidf_df$sarcastic <- labels

set.seed(42)
train_idx <- createDataPartition(labels, p = 0.8, list = FALSE)

bow_train <- bow_df[train_idx, ]
bow_test <- bow_df[-train_idx, ]
tfidf_train <- tfidf_df[train_idx, ]
tfidf_test <- tfidf_df[-train_idx, ]

convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
  factor(x, levels = c("No", "Yes"))
}

train_and_eval <- function(train_data, test_data) {
  feature_cols <- setdiff(names(train_data), "sarcastic")
  train_bin <- as.data.frame(lapply(train_data[, feature_cols], convert_counts))
  test_bin <- as.data.frame(lapply(test_data[, feature_cols], convert_counts))
  train_bin$sarcastic <- train_data$sarcastic
  test_bin$sarcastic <- test_data$sarcastic
  
  nb_model <- naiveBayes(sarcastic ~ ., data = train_bin)
  nb_pred <- predict(nb_model, test_bin)
  
  x_train <- as.matrix(train_data[, feature_cols])
  x_test <- as.matrix(test_data[, feature_cols])
  y_train <- train_data$sarcastic
  glm_model <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 1, nfolds = 5)
  glm_prob <- predict(glm_model, x_test, s = "lambda.min", type = "response")
  glm_pred <- factor(ifelse(glm_prob > 0.5, 1, 0), levels = levels(test_data$sarcastic))
  
  svm_model <- svm(sarcastic ~ ., data = train_data, kernel = "linear")
  svm_pred <- predict(svm_model, test_data)
  
  list(NaiveBayes = nb_pred, LogisticRegression = glm_pred, SVM = svm_pred)
}

bow_preds <- train_and_eval(bow_train, bow_test)
tfidf_preds <- train_and_eval(tfidf_train, tfidf_test)

print_confusion <- function(true_labels, pred_labels, rep_name, clf_name) {
  cm <- confusionMatrix(pred_labels, true_labels, positive = "1")
  cat("\nConfusion Matrix -", rep_name, "-", clf_name, "\n")
  print(cm$table)
}

print_confusion(bow_test$sarcastic, bow_preds$NaiveBayes, "BoW", "Naive Bayes")
print_confusion(bow_test$sarcastic, bow_preds$LogisticRegression, "BoW", "Logistic Regression")
print_confusion(bow_test$sarcastic, bow_preds$SVM, "BoW", "SVM")
print_confusion(tfidf_test$sarcastic, tfidf_preds$NaiveBayes, "TF-IDF", "Naive Bayes")
print_confusion(tfidf_test$sarcastic, tfidf_preds$LogisticRegression, "TF-IDF", "Logistic Regression")
print_confusion(tfidf_test$sarcastic, tfidf_preds$SVM, "TF-IDF", "SVM")

evaluate_model <- function(true_labels, pred_labels, rep_name, clf_name) {
  cm <- confusionMatrix(pred_labels, true_labels, positive = "1")
  data.frame(
    Representation = rep_name,
    Classifier = clf_name,
    Accuracy = round(cm$overall["Accuracy"], 4),
    Precision = round(cm$byClass["Precision"], 4),
    Recall = round(cm$byClass["Recall"], 4),
    F1 = round(cm$byClass["F1"], 4)
  )
}

results <- rbind(
  evaluate_model(bow_test$sarcastic, bow_preds$NaiveBayes, "BoW", "Naive Bayes"),
  evaluate_model(bow_test$sarcastic, bow_preds$LogisticRegression, "BoW", "Logistic Regression"),
  evaluate_model(bow_test$sarcastic, bow_preds$SVM, "BoW", "SVM"),
  evaluate_model(tfidf_test$sarcastic, tfidf_preds$NaiveBayes, "TF-IDF", "Naive Bayes"),
  evaluate_model(tfidf_test$sarcastic, tfidf_preds$LogisticRegression, "TF-IDF", "Logistic Regression"),
  evaluate_model(tfidf_test$sarcastic, tfidf_preds$SVM, "TF-IDF", "SVM")
)

print(results)

best_row <- results[which.max(results$F1), ]
cat("\nSarcasm Detection Findings\n")
cat("Best representation:", as.character(best_row$Representation), "\n")
cat("Best classifier:", as.character(best_row$Classifier), "\n")
cat("Best F1-score:", best_row$F1, "\n")
cat("Accuracy at best F1:", best_row$Accuracy, "\n")
cat("Precision at best F1:", best_row$Precision, "\n")
cat("Recall at best F1:", best_row$Recall, "\n")

bow_avg_f1 <- mean(results$F1[results$Representation == "BoW"])
tfidf_avg_f1 <- mean(results$F1[results$Representation == "TF-IDF"])
cat("Average F1 for BoW:", round(bow_avg_f1, 4), "\n")
cat("Average F1 for TF-IDF:", round(tfidf_avg_f1, 4), "\n")
if (bow_avg_f1 > tfidf_avg_f1) {
  cat("Overall, BoW representation performed better on average across classifiers.\n")
} else if (tfidf_avg_f1 > bow_avg_f1) {
  cat("Overall, TF-IDF representation performed better on average across classifiers.\n")
} else {
  cat("Overall, BoW and TF-IDF performed equally well on average across classifiers.\n")
}


library(ggplot2)

cm_data <- as.data.frame(confusionMatrix(bow_preds$LogisticRegression, bow_test$sarcastic, positive = "1")$table)

ggplot(cm_data, aes(x = Prediction, y = Reference, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 5) +
  scale_fill_gradient(low = "white", high = "darkgreen") +
  labs(title = "Confusion Matrix - BoW + Logistic Regression", x = "Predicted", y = "Actual") +
  theme_minimal()

ggsave("fig1_confusion_matrix.png", width = 6, height = 5, dpi = 150)

results$Label <- paste(results$Representation, results$Classifier, sep = "\n")

ggplot(results, aes(x = Label, y = F1, fill = Representation)) +
  geom_col() +
  geom_text(aes(label = round(F1, 4)), vjust = -0.3, size = 3) +
  geom_hline(yintercept = mean(results$F1), linetype = "dashed", color = "gray40") +
  labs(title = "F1-Score Comparison Across Representations and Classifiers", x = "", y = "F1-Score") +
  theme_minimal()

ggsave("fig2_f1_comparison.png", width = 7, height = 5, dpi = 150)