# codsoft-task2
This project aims to predict the rating of a movie based on various features such as genre, director, and actors, using machine learning regression techniques. By analyzing historical movie data, this model learns the underlying patterns that influence how movies are rated by users or critics.
# Evaluate Random Forest
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt="d")
plt.title("Confusion Matrix - Random Forest")
plt.show()
# Gender count
gender_counts = df['Sex'].value_counts()
# Map back if encoded
if df["Sex"].dtype in ['int64', 'int32']:
    gender_counts.index = gender_counts.index.map({0: 'Female', 1: 'Male'})
# Plot pie chart
plt.figure(figsize=(6, 6))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', colors=['lightcoral', 'skyblue'])
plt.title("Gender Distribution of Titanic Passengers")  # no emoji
plt.axis('equal')
plt.show()
