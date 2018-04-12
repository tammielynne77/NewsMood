

```python
import tweepy
import numpy as np
import pandas as pd
import datetime as datetime
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
```


```python
from config import (consumer_key, consumer_secret, access_token, access_token_secret)
```


```python
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser = tweepy.parsers.JSONParser())
```


```python
news_orgs = ['@BBC', '@CBSNews', '@CNN', '@FoxNews', '@nytimes']
    
counter = 1

oldest_tweet = None

sentiment = []
        
for org in news_orgs:
    for x in range(5):
        news_tweets = api.user_timeline(org)
        for tweet in news_tweets:
            results = analyzer.polarity_scores(tweet['text'])
            compound = results['compound']
            pos = results['pos']
            neu = results['neu']
            neg = results['neg']
            tweets_ago = counter
            
            oldest_tweet = tweet['id'] - 1
            
            sentiment.append({'News_Organization': org, 'Date': tweet['created_at'], 'Tweet': tweet['text'], 
                              'Compound': compound, 'Positive': pos, 'Neutral': neu,'Negative': neg, 
                              'Tweets_Ago': counter})
            
            counter += 1
```


```python
sentiment_final = pd.DataFrame(sentiment)
sentiment_final.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Compound</th>
      <th>Date</th>
      <th>Negative</th>
      <th>Neutral</th>
      <th>News_Organization</th>
      <th>Positive</th>
      <th>Tweet</th>
      <th>Tweets_Ago</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.2732</td>
      <td>Wed Apr 11 19:03:05 +0000 2018</td>
      <td>0.000</td>
      <td>0.890</td>
      <td>@BBC</td>
      <td>0.110</td>
      <td>Tonight, @bettanyhughes investigates the story...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.1531</td>
      <td>Wed Apr 11 18:00:22 +0000 2018</td>
      <td>0.000</td>
      <td>0.814</td>
      <td>@BBC</td>
      <td>0.186</td>
      <td>💪🥊🇲🇱🇫🇷\nAya Cissoko has always been a fighter....</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0000</td>
      <td>Wed Apr 11 16:24:06 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>@BBC</td>
      <td>0.000</td>
      <td>RT @bbcwritersroom: Just announced -  the 10 #...</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.4939</td>
      <td>Wed Apr 11 16:00:22 +0000 2018</td>
      <td>0.146</td>
      <td>0.538</td>
      <td>@BBC</td>
      <td>0.315</td>
      <td>🎨 Rujazzle is a drag queen, artist and Scottis...</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0000</td>
      <td>Wed Apr 11 14:01:05 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>@BBC</td>
      <td>0.000</td>
      <td>😎☀️ Summer's coming to #SLFN!\nWho can't wait ...</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.set('talk', style = 'dark', palette = 'Dark2', font_scale = 7)
sns.lmplot('Tweets_Ago', 'Compound', hue = 'News_Organization', data = sentiment_final, fit_reg = False, aspect = 1.5,
          size = 50, scatter_kws = {'s' : 3000}) 
plt.title('Tweet Sentiment by News Outlet')
plt.xlabel('Tweet Number')
plt.ylabel('Compound Sentiment')
plt.show()
```


![png](output_6_0.png)



```python
tweet_pivot = pd.pivot_table(sentiment_final, values = 'Compound', index = 'News_Organization')
tweet_pivot
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Compound</th>
    </tr>
    <tr>
      <th>News_Organization</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>@BBC</th>
      <td>0.025190</td>
    </tr>
    <tr>
      <th>@CBSNews</th>
      <td>-0.094830</td>
    </tr>
    <tr>
      <th>@CNN</th>
      <td>-0.066430</td>
    </tr>
    <tr>
      <th>@FoxNews</th>
      <td>-0.135185</td>
    </tr>
    <tr>
      <th>@nytimes</th>
      <td>0.039185</td>
    </tr>
  </tbody>
</table>
</div>




```python
x_axis = ['BBC', 'CBS News', 'CNN', 'Fox', 'New York Times']
```


```python
sns.set('talk', palette = 'Dark2', font_scale = 4)
sns.set_style('ticks', {'xtick.major.size' : 20, 'ytick.major.size' : 20})
sns.factorplot(x_axis, 'Compound', data = tweet_pivot, kind = 'bar', size = 30, aspect = 1.5)

plt.title('Tweet Sentiment by News Outlet')
plt.xlabel('News Outlet')
plt.ylabel('Compound Sentiment')


plt.show()
```


![png](output_9_0.png)



```python
news_final = sentiment_final.drop(['Tweets_Ago'], axis = 1)
```


```python
news_final.to_csv
```
