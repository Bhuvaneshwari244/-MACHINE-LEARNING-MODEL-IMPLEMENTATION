import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Create a small sample dataset
data = {
    'label': ['ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'ham', 'spam', 'ham'],
    'message': [
        'Hey, are we still on for dinner tonight?',
        'Congratulations! You have won a free lottery. Claim now!',
        'Let’s meet at 5 pm for the project discussion.',
        'You have been selected for a free iPhone! Click here.',
        'Can you send me the report by tomorrow?',
        'Urgent! Your bank account is compromised. Contact us now!',
        'Happy Birthday! Hope you have a great day.',
        'Meeting postponed to next Monday.',
        'Win a free vacation to Dubai! Limited time offer.',
        'Don’t forget to submit the assignment by Friday.'
    ]
}

df = pd.DataFrame(data)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})  # Convert labels to binary

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])

# Create a pipeline for text processing and model training
pipeline = Pipeline([
    ('vect', CountVectorizer(stop_words='english')),  # Remove common stop words
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'\n✅ Model Accuracy: {accuracy:.2f}')
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Prevent window from closing immediately
input("\nPress Enter to exit...")
