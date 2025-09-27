# Complete Bag-of-Words Implementation for Spam Detection
library(tidytext)
library(readr)
library(dplyr)
library(ggplot2)

# Function to read data
read_data <- function(path_to_train, path_to_test) {
  return(list(
    train = read.csv(path_to_train),
    test  = read.csv(path_to_test)
  ))
}

# Set PATH to be relative
setwd(dirname(rstudioapi::getSourceEditorContext()$path))
datasets <- read_data("data/train.csv", "data/test.csv")

# Inspect the structure
str(datasets$train)

# PREPROCESSING FUNCTION
preprocess_messages <- function(df) {
  # Create document IDs first
  df$doc_id <- seq_len(nrow(df))
  
  # Basic text cleaning
  df$Message <- tolower(df$Message)
  df$Message <- gsub('[[:punct:] ]+',' ', df$Message)  # Replace punct/spaces with single space
  df$Message <- trimws(df$Message)  # Trim whitespace
  
  return(df)
}

# TOKENIZATION AND BAG-OF-WORDS CREATION
create_bow_matrix <- function(df) {
  # Load stop words
  stop_wrds <- readr::read_lines("./stop_words.txt")
  stop_wrds <- trimws(stop_wrds)
  stop_wrds <- stop_wrds[nzchar(stop_wrds)] # drop empty lines
  stop_wrds <- data.frame(word = stop_wrds, stringsAsFactors = FALSE)
  
  # Tokenize messages
  df_tokens <- tidytext::unnest_tokens(df, output = "word", input = "Message", 
                                       token = "words", to_lower = TRUE)
  
  # Remove stop words
  df_tokens <- dplyr::anti_join(df_tokens, stop_wrds, by = "word")
  
  # Count word frequencies per document
  word_counts <- dplyr::count(df_tokens, doc_id, word, name = "frequency")
  
  # Create document-term matrix (DTM)
  dtm <- xtabs(frequency ~ doc_id + word, data = word_counts)
  dtm <- as.matrix(dtm)
  
  return(list(
    dtm = dtm,
    tokens = df_tokens,
    word_counts = word_counts,
    labels = df[, c("doc_id", "Category")]
  ))
}

# MAIN PROCESSING
# Process training data
df_train <- preprocess_messages(datasets$train)
bow_train <- create_bow_matrix(df_train)

# Process test data (if available)
if (!is.null(datasets$test) && nrow(datasets$test) > 0) {
  df_test <- preprocess_messages(datasets$test)
  bow_test <- create_bow_matrix(df_test)
}

# FINAL DATA STRUCTURE - Each message as bag-of-words
print("=== BAG-OF-WORDS DATA STRUCTURE ===")
print(paste("Training documents:", nrow(bow_train$dtm)))
print(paste("Vocabulary size:", ncol(bow_train$dtm)))
print(paste("Total non-zero entries:", sum(bow_train$dtm > 0)))

# Display structure
cat("\nDocument-Term Matrix (first 5 docs, first 10 words):\n")
print(bow_train$dtm[1:5, 1:10])

cat("\nLabel distribution:\n")
print(table(bow_train$labels$Category))

# ALTERNATIVE: Convert to data frame format (each row = document, each column = word)
bow_dataframe <- as.data.frame(bow_train$dtm)
bow_dataframe$doc_id <- as.numeric(rownames(bow_dataframe))
bow_dataframe$Category <- bow_train$labels$Category[match(bow_dataframe$doc_id, bow_train$labels$doc_id)]

# Reorder columns: Category first, then word features
bow_dataframe <- bow_dataframe[, c("Category", "doc_id", setdiff(names(bow_dataframe), c("Category", "doc_id")))]

cat("\nFinal Bag-of-Words DataFrame structure:\n")
str(bow_dataframe[, 1:10])  # Show first few columns

# UTILITY FUNCTIONS FOR ANALYSIS
analyze_vocabulary <- function(bow_result) {
  # Most frequent words across all documents
  word_freq <- colSums(bow_result$dtm)
  top_words <- sort(word_freq, decreasing = TRUE)[1:20]
  
  cat("\nTop 20 most frequent words:\n")
  print(top_words)
  
  # Vocabulary statistics
  cat(paste("\nVocabulary statistics:"))
  cat(paste("\n- Total unique words:", length(word_freq)))
  cat(paste("\n- Words appearing only once:", sum(word_freq == 1)))
  cat(paste("\n- Words appearing >10 times:", sum(word_freq > 10)))
  
  return(top_words)
}

# Analyze the vocabulary
top_words <- analyze_vocabulary(bow_train)

# VISUALIZATION (optional)
create_word_frequency_plot <- function(top_words) {
  word_df <- data.frame(
    word = names(top_words),
    frequency = as.numeric(top_words),
    stringsAsFactors = FALSE
  )
  
  ggplot(word_df[1:15, ], aes(x = reorder(word, frequency), y = frequency)) +
    geom_col(fill = "steelblue") +
    coord_flip() +
    labs(title = "Top 15 Most Frequent Words",
         x = "Words", y = "Frequency") +
    theme_minimal()
}

# Create visualization
word_plot <- create_word_frequency_plot(top_words)
print(word_plot)

# EXPORT RESULTS
# Save the bag-of-words matrix
write.csv(bow_dataframe, "bow_train_data.csv", row.names = FALSE)
cat("\nBag-of-words data saved to 'bow_train_data.csv'\n")

# Summary of final data structure
cat("\n=== FINAL DATA STRUCTURE SUMMARY ===\n")
cat("Each row represents one message/document\n")
cat("Each column (except Category, doc_id) represents a unique word\n")
cat("Values represent word frequency in that document\n")
cat("This is your bag-of-words representation ready for machine learning!\n")

