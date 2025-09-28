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
