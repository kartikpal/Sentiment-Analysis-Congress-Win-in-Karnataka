import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

# Download required NLTK resources (only required for the first run)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Load comments from CSV file
comments_df = pd.read_csv('comments.csv')

# Convert any non-string values to strings
comments_df['comment'] = comments_df['comment'].astype(str)

comments = comments_df['comment'].tolist()

# Preprocess the comments
stop_words = set(stopwords.words('english'))

processed_comments = []
for comment in comments:
    # Tokenize the comment
    tokens = word_tokenize(comment)

    # Remove stopwords
    filtered_tokens = [token.lower() for token in tokens if token.lower() not in stop_words]

    # Add the processed comment to the list
    processed_comments.append(filtered_tokens)

# Perform sentiment analysis on each comment
sia = SentimentIntensityAnalyzer()

sentiment_scores = []
for comment in processed_comments:
    # Join the preprocessed tokens back into a sentence
    comment_text = ' '.join(comment)

    # Perform sentiment analysis using Vader
    sentiment = sia.polarity_scores(comment_text)
    sentiment_scores.append(sentiment)

# Add sentiment scores to the comments dataframe
comments_df['Sentiment Score'] = sentiment_scores

# Save the updated dataframe to a new CSV file
comments_df.to_csv('comments_with_sentiment.csv', index=False)


# Count the number of positive, negative, and neutral sentiments
positive_count = 0
negative_count = 0
neutral_count = 0

for sentiment in comments_df['Sentiment Score']:
    compound_score = sentiment['compound']
    if compound_score > 0.1:
        positive_count += 1
    elif compound_score < -0.1:
        negative_count += 1
    else:
        neutral_count += 1

# Calculate the overall sentiment
if positive_count > negative_count:
    overall_sentiment = 'Positive'
elif negative_count > positive_count:
    overall_sentiment = 'Negative'
else:
    overall_sentiment = 'Neutral'

print('Overall Sentiment:', overall_sentiment)

import pandas as pd
import matplotlib.pyplot as plt
import ast

# Load the comments dataframe with sentiment scores
comments_df = pd.read_csv('comments_with_sentiment.csv')

# Convert the sentiment scores from string to dictionary
comments_df['Sentiment Score'] = comments_df['Sentiment Score'].apply(ast.literal_eval)

# Calculate the average sentiment score for each comment
average_sentiment = comments_df['Sentiment Score'].apply(lambda x: x['compound']).mean()

# Plot the sentiment scores
plt.figure(figsize=(10, 6))
plt.hist(comments_df['Sentiment Score'].apply(lambda x: x['compound']), bins=10, edgecolor='black')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.title('Distribution of Sentiment Scores')
plt.axvline(x=average_sentiment, color='red', linestyle='--', label='Average Sentiment')
plt.legend()
plt.show()
plt.savefig('Distribution of Sentiment Scores.png')

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Convert comments to strings and concatenate them into a single string
all_comments = ' '.join(comments_df['comment'].astype(str))

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_comments)

# Plot the word cloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
plt.savefig('Wordcloud.png')



