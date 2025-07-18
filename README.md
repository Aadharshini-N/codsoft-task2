# codsoft-task2
This project aims to predict the rating of a movie based on various features such as genre, director, and actors, using machine learning regression techniques. By analyzing historical movie data, this model learns the underlying patterns that influence how movies are rated by users or critics.
# Install required packages (if not already installed)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import joblib
import ast
# Load datasets (make sure you have both CSV files)
df= pd.read_csv('IMDb Movies India.csv', encoding='latin1')
print("\n📄 Dataset Loaded:")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df)
import pandas as pd
df = pd.read_csv("C:/Users/Aadha/internship/task2/IMDb Movies India.csv", encoding='ISO-8859-1')
# Step 3: Clean & Preprocess Data
print("\n🧼 Cleaning & Preprocessing...")
# Drop rows with missing important data
df.dropna(subset=['Rating', 'Genre', 'Director', 'Actor 1'], inplace=True)
# Convert Genre from string to list (assume comma-separated)
df['Genre'] = df['Genre'].apply(lambda x: [g.strip() for g in str(x).split(',')])
# One-hot encode genres
mlb = MultiLabelBinarizer()
genre_dummies = pd.DataFrame(mlb.fit_transform(df['Genre']), columns=mlb.classes_)
df = pd.concat([df, genre_dummies], axis=1)
# Encode categorical features
for col in ['Director', 'Actor 1', 'Actor 2', 'Actor 3']:
    df[col] = df[col].astype(str)
    df[col] = LabelEncoder().fit_transform(df[col])
# Convert to numeric and fill missing values
df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')
df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')
df['Duration'] = df['Duration'].fillna(df['Duration'].median())
df['Votes'] = df['Votes'].fillna(df['Votes'].median())
print("✅ Data cleaning complete.")
print("📊 Plotting: Distribution of Movie Ratings")
# showing distribution of movie rating
plt.figure(figsize=(8, 5))
sns.histplot(df['Rating'], bins=15, kde=True, color='skyblue')
plt.title('Distribution of Movie Ratings')
plt.xlabel('Rating')
plt.ylabel('Number of Movies')
plt.tight_layout()
plt.show()
print("📊 Plotting: Distribution of Votes")
# showing distribution of votes
plt.figure(figsize=(8, 5))
sns.histplot(df['Votes'], bins=30, color='orange')
plt.title('Distribution of Votes')
plt.xlabel('Votes')
plt.ylabel('Number of Movies')
plt.tight_layout()
plt.show()
print("📊 Plotting: Top 10 Directors by Number of Movies")

top_directors = df['Director'].value_counts().head(10)
plt.figure(figsize=(10, 5))
sns.barplot(x=top_directors.values, y=top_directors.index, hue=top_directors.index, palette='pastel', dodge=False, legend=False)
# showing number of movies
plt.title("Top 10 Most Frequent Directors")
plt.xlabel("Number of Movies")
plt.ylabel("Director")
plt.tight_layout()
plt.show()
print("📊 Plotting: Distribution of Movie Durations")
# distribution of movie duration
plt.figure(figsize=(8, 5))
sns.boxplot(x=df['Duration'], color='lightgreen')
plt.title('Movie Duration Distribution')
plt.xlabel('Duration (minutes)')
plt.tight_layout()
plt.show()
print("\n📊 Plotting: Votes vs Rating")
# distribution of votes vs rating
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='Votes', y='Rating', hue='Rating', palette='coolwarm', alpha=0.6)
plt.title("Votes vs Rating")
plt.xlabel("Number of Votes")
plt.ylabel("Rating")
plt.xscale("log")
plt.grid(True)
plt.tight_layout()
plt.show()
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Heatmap of Numeric Features")
plt.tight_layout()
plt.show()
