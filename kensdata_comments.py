#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime
from textblob import TextBlob
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS,ImageColorGenerator
from PIL import Image
import seaborn as sns
from pathlib import Path


# In[12]:


Agg_met_country_suscrib = Path(__file__).parents[0] /'Aggregated_Metrics_By_Country_And_Subscriber_Status.csv'


# In[3]:


def style_negative(v,props=''):
    """Style negative values"""
    try:
        return props if v < 0 else None
    except:
        pass
    
def style_positive(v,props=''):
    """Style positive values"""
    try:
        return props if v > 0 else None
    except:
        pass

def audience_simple(country):
    '''SHOW TOP COUNTRIES'''
    if country == "US":
        return 'USA'
    elif country == 'IN':
        return 'INDIA'
    else:
        return 'Others'
def find_names(m):
    try:
        return Aggremetvid2.loc[m,'Video title']
    
    except:
        pass
    
def polarity(reviews):
    return TextBlob(reviews).sentiment.polarity


# In[13]:


@st.cache
def load_data():
    Agg_met_country_suscriber = pd.read_csv(Agg_met_country_suscrib)
    Agg_met_video =pd.read_csv('C:/Users/Bumblebee/Downloads/nlp-getting-started/kaggle files/youtube folder/Aggregated_Metrics_By_Video.csv')
    All_comments =pd.read_csv('C:/Users/Bumblebee/Downloads/nlp-getting-started/kaggle files/youtube folder/All_Comments_Final.csv')
    video_performace =pd.read_csv('C:/Users/Bumblebee/Downloads/nlp-getting-started/kaggle files/youtube folder/Video_Performance_Over_Time.csv')
    Agg_met_video.columns =["Video",'Video title','Video publish time','comments added',
                               'Shares','Dislikes','Likes','Suscribers lost','Suscribers gained',
                               'RPM(USD)','CMP(USD)','Average percent viewed (%)','Average view duration',                                                                 
                                'Views', 'Watch time (hours)','Subscribers' ,'Your estimated revenue (USD)',
                                'Impressions','Impressions click throughrate(%)']

    Agg_met_video['Video publish time'] = pd.to_datetime(Agg_met_video['Video publish time'])
    Agg_met_video['Average view duration'] = Agg_met_video['Average view duration'].apply(lambda x:datetime.strptime(x,"%H:%M:%S"))
    Agg_met_video['Average duration sec'] = Agg_met_video['Average view duration'].apply(lambda x:x.second + x.minute*60 +x.hour*3600)
    Agg_met_video['Engagement_ratio'] = (Agg_met_video['comments added'] +Agg_met_video['Shares']+ Agg_met_video['Dislikes'] + Agg_met_video['Dislikes'] +Agg_met_video['Likes'])/Agg_met_video.Views
    Agg_met_video['Views/sub gained'] = Agg_met_video['Views']/Agg_met_video['Suscribers gained']
    Agg_met_video.sort_values('Video publish time',ascending = False,inplace = True)
    video_performace['Date'] = pd.to_datetime(video_performace['Date'])
    Aggremetvid2 = pd.read_csv('C:/Users/Bumblebee/Downloads/nlp-getting-started/kaggle files/youtube folder/Aggregated_Metrics_By_Video - Copy.csv',index_col ='Video')
    comments = All_comments[['VidId','Comments']]
    return  Agg_met_country_suscriber,Agg_met_video,All_comments,video_performace,Aggremetvid2


# In[14]:


#loading data into streamlit
Agg_met_country_suscriber,Agg_met_video,All_comments,video_performace,Aggremetvid2 = load_data()


# In[15]:


Agg_met_video2 = Agg_met_video.copy()
metric_data_12mo = Agg_met_video2['Video publish time'].max( ) - pd.DateOffset(months = 12)
median_agg =Agg_met_video2 [Agg_met_video2 ['Video publish time']>=metric_data_12mo].median()
numeric_cols = np.array((Agg_met_video2.dtypes =="float64")|(Agg_met_video2.dtypes == 'int64'))
Agg_met_video2.iloc[:,numeric_cols] = (Agg_met_video2.iloc[:,numeric_cols] - median_agg).div(median_agg)


# In[ ]:





# In[7]:


add_sidebar = st.sidebar.selectbox('Aggregate or individual video or Comments',('Aggregate Metrics','Individual video Analysis','Comments Analysis'))


# In[ ]:





# In[8]:


##Total Picture
if add_sidebar == 'Aggregate Metrics':
    df_agg_metrics = Agg_met_video[['Views','Video publish time','Likes','Subscribers',
                                 'Shares','comments added','Average duration sec','Average percent viewed (%)','RPM(USD)','Engagement_ratio',
                                'Views/sub gained']]
    metric_date_6mo = df_agg_metrics['Video publish time'].max() -pd.DateOffset(months = 6)
    metric_date_12mo = df_agg_metrics['Video publish time'].max() - pd.DateOffset(months =12)
    metric_medians6mo = df_agg_metrics[df_agg_metrics['Video publish time']>=metric_date_6mo].median()
    metric_medians12mo = df_agg_metrics[df_agg_metrics['Video publish time']>=metric_date_12mo].median()

    col1,col2,col3,col4,col5 = st.columns(5)
    columns = [col1,col2,col3,col4,col5]
    count=0
    for i in metric_medians6mo.index:
        with columns[count]:
            delta = (metric_medians6mo[i] -metric_medians12mo[i])/metric_medians12mo[i]
            st.metric(label =i,value = round(metric_medians6mo[i],1),delta = '{:.2%}'.format(delta))
            count +=1
            if count >=5:
                count =0
    Agg_met_video2['publish date'] = Agg_met_video2['Video publish time'].apply(lambda x: x.date())
    Agg_met_video2_final =Agg_met_video2.loc[:,['Video title','Views','publish date','Likes','Subscribers',
                                 'Shares','comments added','Average duration sec','Average percent viewed (%)','RPM(USD)','Engagement_ratio',
                                'Views/sub gained']]
    agg_numeric_lst = Agg_met_video2_final.median().index.tolist()
    add_to_pct = {}
    for i in agg_numeric_lst:
        add_to_pct[i] = '{:.1%}' .format 
    st.dataframe(Agg_met_video2_final.style.applymap(style_negative,props = 'color:red;').applymap(style_positive,props = 'color:green;').format(add_to_pct))
   
        


# In[9]:


pct_20=lambda x:np.percentile(x,20)
pct_20.__name__='pct_20'


# In[10]:


#merge daily data with published data to get delta
df_time_diff = pd.merge(video_performace,Agg_met_video.loc[:,['Video','Video publish time']],
                     left_on = "External Video ID",right_on = 'Video')
df_time_diff['days_published'] = (df_time_diff['Date'] - df_time_diff['Video publish time']).dt.days
#collecting daily published data for 12 months
date_12mo = Agg_met_video['Video publish time'].max() - pd.DateOffset(months = 12)
df_time_diff_yr = df_time_diff[df_time_diff['Video publish time']>= date_12mo]

#get daily view data (first 30) median & percentiles
views_days = pd.pivot_table(df_time_diff_yr,index='days_published',values='Views',aggfunc=[np.mean,np.median,lambda x:np.percentile(x,80),pct_20]).reset_index()
views_days.columns=['days_published','mean_views','median_views','80pct_views','20pct_views']
views_days = views_days[views_days['days_published'].between(0,30)]
views_cumulative = views_days.loc[:,['days_published','median_views','80pct_views','20pct_views']]
views_cumulative.loc[:,['median_views','80pct_views','20pct_views']] = views_cumulative.loc[:,['median_views','80pct_views','20pct_views']].cumsum() 


# In[ ]:





# In[19]:


if add_sidebar == 'Individual video Analysis':
    #selecting videos for selectbox
    Videos = tuple(Agg_met_video['Video title'])
    Videos_select = st.selectbox('pick A Video',Videos)
    gg_filtered = Agg_met_video[Agg_met_video['Video title']== Videos_select]
    agg_sub_filtered = Agg_met_country_suscriber[Agg_met_country_suscriber['Video Title']==   Videos_select]
    agg_sub_filtered['Country'] = agg_sub_filtered['Country Code'].apply(audience_simple)
    #ploting of number suscribers by location 
    agg_sub_filtered.sort_values('Is Subscribed',inplace= True)
    fig = px.bar(agg_sub_filtered,x = 'Views',y='Is Subscribed', color ='Country',orientation = 'h')
    st.plotly_chart(fig)
    #ploting of percentile plot
    agg_time_filtered =  df_time_diff[df_time_diff['Video Title'] ==  Videos_select]
    first_30 = agg_time_filtered[agg_time_filtered['days_published'].between(0,30)]
    first_30 = first_30.sort_values('days_published')
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=views_cumulative['days_published'],y=views_cumulative['20pct_views'],
                              mode='lines',
                              name= '20th percentile',line=dict(color='purple',dash = 'dash')))
    fig2.add_trace(go.Scatter(x=views_cumulative['days_published'],y=views_cumulative['median_views'],
                              mode='lines',
                              name= '50th percentile',line=dict(color='black',dash = 'dash')))
    fig2.add_trace(go.Scatter(x=views_cumulative['days_published'],y=views_cumulative['80pct_views'],
                              mode='lines',
                              name= '80th percentile',line=dict(color='royalblue',dash = 'dash')))
    fig2.add_trace(go.Scatter(x=first_30['days_published'],y=first_30['Views'].cumsum(),
                              mode='lines',
                              name= 'current video',line=dict(color='firebrick',width=8)))    
    fig2.update_layout(title= 'view comparison first 30 days',xaxis_title = 'days since published',yaxis_title='views commulative')
    st.plotly_chart(fig2)
    


# In[ ]:


#collecting comments data
comments = All_comments[['VidId','Comments']]
a=comments['Comments'].apply(lambda x:re.findall(r'[^.].*'+'[?$]',str(x)))#extracting questions
b=comments["VidId"].apply(lambda x:find_names(x))#matching video_id to corresponding Video names
#turning data to dataframes(a and b)
a=pd.DataFrame(a)
b=pd.DataFrame(b)
comments=pd.concat([comments,a,b], axis=1, ignore_index=True)#joining a&b to comments dataframe
comments=comments.rename(columns={0:"VidId",1:'Comments',2:'Questions',3:'Video Title'})#renaming title
comments=comments.reindex(columns=["VidId",'Video Title','Comments','Questions'])#renaming the columns


# In[ ]:


if add_sidebar=='Comments Analysis':
    #selecting video names for select bar
    Video_slect=comments['Video Title'].value_counts()
    Videos = tuple(Video_slect.index)
    Videos_select = st.selectbox('pick A Video',Videos)
    comm_video_names= comments[comments['Video Title']==   Videos_select]
    #select questions asked for each video 
    for i in comm_video_names['Video Title']:
        filt= (comments['Video Title'] == i)
        a=comments.loc[filt,['Comments','Questions']]
    st.dataframe(a)
    #running a sentiment analysis for comment section
    comm_video_names['Comments']=comm_video_names['Comments'].astype('str')
    comm_video_names['polarity'] = comm_video_names['Comments'].apply(lambda x :polarity(x))
    comm_video_names['Expression']=np.where(comm_video_names['polarity'] > 0,"Positive",'Negative')
    comm_video_names.loc[comm_video_names.polarity ==0,'Expression']= 'Neutral'
    #ploting a countplot to show sentiment for each video
    siz =(3,2)
    fig,ax = plt.subplots(figsize= siz,dpi=40)
    ax=sns.countplot(x = 'Expression',data=comm_video_names,palette = 'Set1')

    st.pyplot(fig)

    #plotting wordcloud for each selected video
    for i in comm_video_names['Video Title']:
        filt= (comments['Video Title'] == i)
        a=comments.loc[filt,['Comments']].values
    stopwords = set(STOPWORDS)
    wc = WordCloud(stopwords = stopwords).generate(str(a))
    fig, ax = plt.subplots()
    ax.scatter([1, 2, 3], [1, 2, 3])
    plt.imshow(wc,interpolation = 'bilinear' )
    plt.axis('off')
    plt.show()
    st.pyplot(fig,figsize= siz,dpi=40)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[155]:





# In[ ]:





# In[ ]:





# In[ ]:




