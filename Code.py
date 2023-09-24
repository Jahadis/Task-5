# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Step 1: Load the dataset
complaints_df = pd.read_csv("complaints.csv")  # Replace with your file path

# Step 2: Text Pre-processing

X = complaints_df['complaint_text']
y = complaints_df['category']  # Replace 'category' with your actual target column name

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data to TF-IDF features
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust the number of features
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Step 3: Model Selection (Using Multinomial Naive Bayes as an example)
model = MultinomialNB()

# Step 4: Model Training
model.fit(X_train_tfidf, y_train)

# Step 5: Model Evaluation
y_pred = model.predict(X_test_tfidf)
report = classification_report(y_test, y_pred)

# Print the classification report
print(report)

# Step 6: Make Predictions
new_complaints = ["This company charged me incorrectly.",
                  "I can't access my account. Please help!"]
new_complaints_tfidf = tfidf_vectorizer.transform(new_complaints)
predicted_categories = model.predict(new_complaints_tfidf)

for complaint, category in zip(new_complaints, predicted_categories):
    print(f"Complaint: {complaint}")
    print(f"Predicted Category: {category}")
