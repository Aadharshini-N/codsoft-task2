# codsoft-task2
This project aims to predict the rating of a movie based on various features such as genre, director, and actors, using machine learning regression techniques. By analyzing historical movie data, this model learns the underlying patterns that influence how movies are rated by users or critics.
# Evaluate Random Forest
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt="d")
plt.title("Confusion Matrix - Random Forest")
plt.show()
