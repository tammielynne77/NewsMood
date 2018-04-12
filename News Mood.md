

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




    <bound method DataFrame.to_csv of      Compound                            Date  Negative  Neutral  \
    0      0.2732  Wed Apr 11 19:03:05 +0000 2018     0.000    0.890   
    1      0.1531  Wed Apr 11 18:00:22 +0000 2018     0.000    0.814   
    2      0.0000  Wed Apr 11 16:24:06 +0000 2018     0.000    1.000   
    3      0.4939  Wed Apr 11 16:00:22 +0000 2018     0.146    0.538   
    4      0.0000  Wed Apr 11 14:01:05 +0000 2018     0.000    1.000   
    5      0.0000  Wed Apr 11 13:04:48 +0000 2018     0.000    1.000   
    6      0.0000  Wed Apr 11 12:42:05 +0000 2018     0.000    1.000   
    7     -0.4803  Wed Apr 11 12:21:33 +0000 2018     0.221    0.779   
    8      0.0000  Wed Apr 11 12:03:02 +0000 2018     0.000    1.000   
    9     -0.0752  Wed Apr 11 11:28:06 +0000 2018     0.132    0.681   
    10     0.3612  Wed Apr 11 11:01:05 +0000 2018     0.000    0.848   
    11     0.0000  Wed Apr 11 10:00:57 +0000 2018     0.000    1.000   
    12    -0.3595  Wed Apr 11 09:37:34 +0000 2018     0.151    0.849   
    13     0.5106  Wed Apr 11 09:07:39 +0000 2018     0.000    0.798   
    14     0.1779  Wed Apr 11 08:49:25 +0000 2018     0.238    0.476   
    15     0.4522  Wed Apr 11 08:01:02 +0000 2018     0.000    0.852   
    16     0.0000  Wed Apr 11 07:28:07 +0000 2018     0.000    1.000   
    17     0.0000  Wed Apr 11 07:00:19 +0000 2018     0.000    1.000   
    18    -0.4927  Tue Apr 10 18:42:00 +0000 2018     0.158    0.842   
    19    -0.5106  Tue Apr 10 18:00:27 +0000 2018     0.216    0.784   
    20     0.2732  Wed Apr 11 19:03:05 +0000 2018     0.000    0.890   
    21     0.1531  Wed Apr 11 18:00:22 +0000 2018     0.000    0.814   
    22     0.0000  Wed Apr 11 16:24:06 +0000 2018     0.000    1.000   
    23     0.4939  Wed Apr 11 16:00:22 +0000 2018     0.146    0.538   
    24     0.0000  Wed Apr 11 14:01:05 +0000 2018     0.000    1.000   
    25     0.0000  Wed Apr 11 13:04:48 +0000 2018     0.000    1.000   
    26     0.0000  Wed Apr 11 12:42:05 +0000 2018     0.000    1.000   
    27    -0.4803  Wed Apr 11 12:21:33 +0000 2018     0.221    0.779   
    28     0.0000  Wed Apr 11 12:03:02 +0000 2018     0.000    1.000   
    29    -0.0752  Wed Apr 11 11:28:06 +0000 2018     0.132    0.681   
    ..        ...                             ...       ...      ...   
    470   -0.6808  Wed Apr 11 23:02:03 +0000 2018     0.255    0.677   
    471    0.0000  Wed Apr 11 22:47:01 +0000 2018     0.000    1.000   
    472   -0.3400  Wed Apr 11 22:32:04 +0000 2018     0.195    0.699   
    473    0.0000  Wed Apr 11 22:17:01 +0000 2018     0.000    1.000   
    474    0.0000  Wed Apr 11 22:01:03 +0000 2018     0.000    1.000   
    475    0.5859  Wed Apr 11 21:46:05 +0000 2018     0.000    0.714   
    476    0.0000  Wed Apr 11 21:31:04 +0000 2018     0.000    1.000   
    477   -0.2960  Wed Apr 11 21:16:07 +0000 2018     0.155    0.845   
    478    0.3885  Wed Apr 11 21:03:04 +0000 2018     0.000    0.832   
    479    0.6249  Wed Apr 11 21:01:06 +0000 2018     0.000    0.788   
    480    0.0772  Thu Apr 12 01:32:03 +0000 2018     0.000    0.939   
    481   -0.1027  Thu Apr 12 01:17:01 +0000 2018     0.105    0.806   
    482    0.0000  Thu Apr 12 01:02:05 +0000 2018     0.000    1.000   
    483    0.0000  Thu Apr 12 00:47:04 +0000 2018     0.000    1.000   
    484    0.0000  Thu Apr 12 00:32:06 +0000 2018     0.000    1.000   
    485    0.0000  Thu Apr 12 00:17:05 +0000 2018     0.000    1.000   
    486    0.0000  Thu Apr 12 00:02:08 +0000 2018     0.000    1.000   
    487    0.5267  Wed Apr 11 23:47:06 +0000 2018     0.000    0.794   
    488    0.0000  Wed Apr 11 23:32:06 +0000 2018     0.000    1.000   
    489    0.0000  Wed Apr 11 23:17:03 +0000 2018     0.000    1.000   
    490   -0.6808  Wed Apr 11 23:02:03 +0000 2018     0.255    0.677   
    491    0.0000  Wed Apr 11 22:47:01 +0000 2018     0.000    1.000   
    492   -0.3400  Wed Apr 11 22:32:04 +0000 2018     0.195    0.699   
    493    0.0000  Wed Apr 11 22:17:01 +0000 2018     0.000    1.000   
    494    0.0000  Wed Apr 11 22:01:03 +0000 2018     0.000    1.000   
    495    0.5859  Wed Apr 11 21:46:05 +0000 2018     0.000    0.714   
    496    0.0000  Wed Apr 11 21:31:04 +0000 2018     0.000    1.000   
    497   -0.2960  Wed Apr 11 21:16:07 +0000 2018     0.155    0.845   
    498    0.3885  Wed Apr 11 21:03:04 +0000 2018     0.000    0.832   
    499    0.6249  Wed Apr 11 21:01:06 +0000 2018     0.000    0.788   
    
        News_Organization  Positive  \
    0                @BBC     0.110   
    1                @BBC     0.186   
    2                @BBC     0.000   
    3                @BBC     0.315   
    4                @BBC     0.000   
    5                @BBC     0.000   
    6                @BBC     0.000   
    7                @BBC     0.000   
    8                @BBC     0.000   
    9                @BBC     0.187   
    10               @BBC     0.152   
    11               @BBC     0.000   
    12               @BBC     0.000   
    13               @BBC     0.202   
    14               @BBC     0.286   
    15               @BBC     0.148   
    16               @BBC     0.000   
    17               @BBC     0.000   
    18               @BBC     0.000   
    19               @BBC     0.000   
    20               @BBC     0.110   
    21               @BBC     0.186   
    22               @BBC     0.000   
    23               @BBC     0.315   
    24               @BBC     0.000   
    25               @BBC     0.000   
    26               @BBC     0.000   
    27               @BBC     0.000   
    28               @BBC     0.000   
    29               @BBC     0.187   
    ..                ...       ...   
    470          @nytimes     0.068   
    471          @nytimes     0.000   
    472          @nytimes     0.107   
    473          @nytimes     0.000   
    474          @nytimes     0.000   
    475          @nytimes     0.286   
    476          @nytimes     0.000   
    477          @nytimes     0.000   
    478          @nytimes     0.168   
    479          @nytimes     0.212   
    480          @nytimes     0.061   
    481          @nytimes     0.089   
    482          @nytimes     0.000   
    483          @nytimes     0.000   
    484          @nytimes     0.000   
    485          @nytimes     0.000   
    486          @nytimes     0.000   
    487          @nytimes     0.206   
    488          @nytimes     0.000   
    489          @nytimes     0.000   
    490          @nytimes     0.068   
    491          @nytimes     0.000   
    492          @nytimes     0.107   
    493          @nytimes     0.000   
    494          @nytimes     0.000   
    495          @nytimes     0.286   
    496          @nytimes     0.000   
    497          @nytimes     0.000   
    498          @nytimes     0.168   
    499          @nytimes     0.212   
    
                                                     Tweet  
    0    Tonight, @bettanyhughes investigates the story...  
    1    💪🥊🇲🇱🇫🇷\nAya Cissoko has always been a fighter....  
    2    RT @bbcwritersroom: Just announced -  the 10 #...  
    3    🎨 Rujazzle is a drag queen, artist and Scottis...  
    4    😎☀️ Summer's coming to #SLFN!\nWho can't wait ...  
    5    🎧🎶 From @CraigDavid to @Camila_Cabello, here a...  
    6    'I'm going to be very careful'. \n🤦‍♂️🌊😂@MikeB...  
    7    RT @BBCLookNorth: Remember Benji the goat..? H...  
    8    🤖🚀 Scientists are using this giant sand pit to...  
    9    This 12-year-old Manchester attack survivor ha...  
    10   What it’s like when your son is diagnosed with...  
    11   😻 Meet the tree-climbing cat-saving superheroe...  
    12   RT @BBCBreakfast: 🤣 In case you missed it... 😱...  
    13   RT @bbcthree: Meet the vet who treats homeless...  
    14   RT @BBCFOUR: Bad news if you love wasabi 😱 htt...  
    15   ❤️ Eight-year-old Cara has learned sign langua...  
    16   🐧❄️ Antarctica: A journey to the edge of a fro...  
    17   ❤️ This scheme has already funded 20,000 meals...  
    18   It wasn't until Henry VIII's reign that the #T...  
    19   Meet Rebekah - a former professional footballe...  
    20   Tonight, @bettanyhughes investigates the story...  
    21   💪🥊🇲🇱🇫🇷\nAya Cissoko has always been a fighter....  
    22   RT @bbcwritersroom: Just announced -  the 10 #...  
    23   🎨 Rujazzle is a drag queen, artist and Scottis...  
    24   😎☀️ Summer's coming to #SLFN!\nWho can't wait ...  
    25   🎧🎶 From @CraigDavid to @Camila_Cabello, here a...  
    26   'I'm going to be very careful'. \n🤦‍♂️🌊😂@MikeB...  
    27   RT @BBCLookNorth: Remember Benji the goat..? H...  
    28   🤖🚀 Scientists are using this giant sand pit to...  
    29   This 12-year-old Manchester attack survivor ha...  
    ..                                                 ...  
    470  “I want to see a serial rapist convicted,” a w...  
    471  Evening Briefing: Here's what you need to know...  
    472  The world knew them as missing teenagers, in d...  
    473  Many apply, but only 60 older musicians are ch...  
    474  RT @TheSteinLine: Here’s a link to the latest ...  
    475  Mark Zuckerberg "answered with white-noise-mac...  
    476  $1,499,000 https://t.co/CEhtgQQjG6 https://t.c...  
    477  All the major firings and resignations in the ...  
    478  RT @kenvogel: SCOOP: @EPAScottPruitt wanted to...  
    479  RT @NYTNational: Trump has signed legislation ...  
    480  Home number. Address book. The names of his ex...  
    481  RT @tminsberg: NFL cheerleaders are strictly r...  
    482  Evening Briefing: Here's what you need to know...  
    483  “They became a rallying point.” Face-to-face w...  
    484  When Scott Pruitt decided to refashion the EPA...  
    485  RT @NYTMetro: Cynthia Nixon has made legalizin...  
    486  How Facebook let politicians, companies and ot...  
    487  A former U.S. intelligence official who came u...  
    488  “This is the nightmare scenario,” said former ...  
    489  RT @bxchen: Just published: I downloaded the d...  
    490  “I want to see a serial rapist convicted,” a w...  
    491  Evening Briefing: Here's what you need to know...  
    492  The world knew them as missing teenagers, in d...  
    493  Many apply, but only 60 older musicians are ch...  
    494  RT @TheSteinLine: Here’s a link to the latest ...  
    495  Mark Zuckerberg "answered with white-noise-mac...  
    496  $1,499,000 https://t.co/CEhtgQQjG6 https://t.c...  
    497  All the major firings and resignations in the ...  
    498  RT @kenvogel: SCOOP: @EPAScottPruitt wanted to...  
    499  RT @NYTNational: Trump has signed legislation ...  
    
    [500 rows x 7 columns]>


