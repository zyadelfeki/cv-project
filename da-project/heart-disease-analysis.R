# Heart Disease Analysis in R
# Using Cleveland Heart Disease Dataset

# Install and load required libraries
if (!require("ggplot2")) install.packages("ggplot2")
if (!require("dplyr")) install.packages("dplyr")
if (!require("corrplot")) install.packages("corrplot")

library(ggplot2)
library(dplyr)
library(corrplot)

# Load the data
df <- read.csv("heart-disease.csv")

# Basic exploration
cat("\n=== Data Structure ===\n")
str(df)

cat("\n=== Data Summary ===\n")
summary(df)

cat("\n=== Missing Values ===\n")
# Replace "?" with NA for proper handling
df[df == "?"] <- NA
sum(is.na(df))

# Convert target to factor for plotting
df$target <- as.factor(ifelse(as.numeric(as.character(df$target)) > 0, 1, 0))

# Convert sex to factor for plotting
df$sex <- as.factor(df$sex)
levels(df$sex) <- c("Female", "Male")

# Convert target labels
levels(df$target) <- c("No Disease", "Heart Disease")

# =============================================================================
# Plot 1: Correlation Matrix
# =============================================================================
# This plot shows the correlation between all numerical features
# Red = positive correlation, Blue = negative correlation
cat("\n=== Plot 1: Correlation Matrix ===\n")
cor_matrix <- cor(df %>% select_if(is.numeric), use = "complete.obs")
corrplot(cor_matrix, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 45,
         title = "Correlation Matrix of Heart Disease Features")

# =============================================================================
# Plot 2: Count of Heart Disease vs No Heart Disease
# =============================================================================
# Bar chart showing the distribution of the target variable
cat("\n=== Plot 2: Heart Disease Distribution ===\n")
ggplot(df, aes(x = target, fill = target)) +
  geom_bar() +
  labs(title = "Heart Disease Distribution",
       x = "Heart Disease Status",
       y = "Count") +
  theme_minimal() +
  scale_fill_manual(values = c("No Disease" = "#2E86AB", "Heart Disease" = "#E94F37"))

# =============================================================================
# Plot 3: Age Distribution by Heart Disease Status
# =============================================================================
# Histogram showing age distribution colored by disease status
cat("\n=== Plot 3: Age Distribution by Disease Status ===\n")
ggplot(df, aes(x = age, fill = target, alpha = 0.7)) +
  geom_histogram(bins = 20, position = "identity") +
  labs(title = "Age Distribution by Heart Disease Status",
       x = "Age",
       y = "Count",
       fill = "Status") +
  theme_minimal() +
  scale_fill_manual(values = c("No Disease" = "#2E86AB", "Heart Disease" = "#E94F37"))

# =============================================================================
# Plot 4: Max Heart Rate (thalach) by Target Group
# =============================================================================
# Boxplot showing maximum heart rate distribution for each disease group
cat("\n=== Plot 4: Max Heart Rate by Disease Status ===\n")
ggplot(df, aes(x = target, y = thalach, fill = target)) +
  geom_boxplot() +
  labs(title = "Maximum Heart Rate by Heart Disease Status",
       x = "Heart Disease Status",
       y = "Maximum Heart Rate (thalach)") +
  theme_minimal() +
  scale_fill_manual(values = c("No Disease" = "#2E86AB", "Heart Disease" = "#E94F37"))

# =============================================================================
# Plot 5: Chest Pain Type by Target
# =============================================================================
# Grouped bar chart showing chest pain types and their relation to heart disease
cat("\n=== Plot 5: Chest Pain Type by Disease Status ===\n")
# Convert cp to factor with labels
df$cp <- as.factor(df$cp)
levels(df$cp) <- c("Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic")

ggplot(df, aes(x = cp, fill = target)) +
  geom_bar(position = "dodge") +
  labs(title = "Chest Pain Type by Heart Disease Status",
       x = "Chest Pain Type",
       y = "Count",
       fill = "Status") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_manual(values = c("No Disease" = "#2E86AB", "Heart Disease" = "#E94F37"))

# =============================================================================
# Summary of Key Findings
# =============================================================================
cat("\n=== KEY FINDINGS ===\n")

# Calculate summary statistics
disease_count <- sum(df$target == "Heart Disease")
no_disease_count <- sum(df$target == "No Disease")

cat("1. Dataset contains", nrow(df), "patients\n")
cat("2. Heart Disease: ", disease_count, " patients (", round(disease_count/nrow(df)*100,1), "%)\n", sep="")
cat("3. No Heart Disease: ", no_disease_count, " patients (", round(no_disease_count/nrow(df)*100,1), "%)\n", sep="")

# Average age by disease status
age_by_disease <- df %>% group_by(target) %>% summarise(mean_age = mean(age, na.rm = TRUE))
cat("4. Average age - Heart Disease:", round(age_by_disease$mean_age[age_by_disease$target == "Heart Disease"],1), "years\n")
cat("5. Average age - No Disease:", round(age_by_disease$mean_age[age_by_disease$target == "No Disease"],1), "years\n")

# Average max heart rate by disease status
thalach_by_disease <- df %>% group_by(target) %>% summarise(mean_thalach = mean(thalach, na.rm = TRUE))
cat("6. Avg Max Heart Rate - Heart Disease:", round(thalach_by_disease$mean_thalach[thalach_by_disease$target == "Heart Disease"],1), "bpm\n")
cat("7. Avg Max Heart Rate - No Disease:", round(thalach_by_disease$mean_thalach[thalach_by_disease$target == "No Disease"],1), "bpm\n")

cat("\nAnalysis complete!\n")