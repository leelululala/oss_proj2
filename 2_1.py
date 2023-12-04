import pandas as pd
import numpy as np

kbo = pd.read_csv("2019_kbo_for_kaggle_v2.csv")

y2015 = kbo[kbo['year'] == 2015]
y2016 = kbo[kbo['year'] == 2016]
y2017 = kbo[kbo['year'] == 2017]
y2018 = kbo[kbo['year'] == 2018]

ranking = pd.Index(np.arange(1, 11))


# In[42]:


def top2015():
    th = y2015.sort_values(by=['H'], ascending=False).head(n=10)
    th = th[['batter_name', 'H']]
    th.index = ranking
    th.columns = ['name', 'hits']

    ta = y2015.sort_values(by=['avg'], ascending=False).head(n=10)
    ta = ta[['batter_name', 'avg']]
    ta.index = ranking
    ta.columns = ['name', 'batting avg']

    thr = y2015.sort_values(by=['HR'], ascending=False).head(n=10)
    thr = thr[['batter_name', 'HR']]
    thr.index = ranking
    thr.columns = ['name', 'homerun']

    to = y2015.sort_values(by=['OBP'], ascending=False).head(n=10)
    to = to[['batter_name', 'OBP']]
    to.index = ranking
    to.columns = ['name', 'obp']

    print('< top 10 players in 2015 >')
    print('\n 1)hits')

    print(th)
    print('\n 2)battting average')
    print(ta)
    print('\n 3)homerun')
    print(thr)
    print('\n 4)on-base percentage')
    print(to)


def top2016():
    th = y2016.sort_values(by=['H'], ascending=False).head(n=10)
    th = th[['batter_name', 'H']]
    th.index = ranking
    th.columns = ['name', 'hits']

    ta = y2016.sort_values(by=['avg'], ascending=False).head(n=10)
    ta = ta[['batter_name', 'avg']]
    ta.index = ranking
    ta.columns = ['name', 'batting avg']

    thr = y2016.sort_values(by=['HR'], ascending=False).head(n=10)
    thr = thr[['batter_name', 'HR']]
    thr.index = ranking
    thr.columns = ['name', 'homerun']

    to = y2016.sort_values(by=['OBP'], ascending=False).head(n=10)
    to = to[['batter_name', 'OBP']]
    to.index = ranking
    to.columns = ['name', 'obp']

    print('< top 10 players in 2016 >')
    print('\n 1)hits')
    print(th)
    print('\n 2)battting average')
    print(ta)
    print('\n 3)homerun')
    print(thr)
    print('\n 4)on-base percentage')
    print(to)


def top2017():
    th = y2017.sort_values(by=['H'], ascending=False).head(n=10)
    th = th[['batter_name', 'H']]
    th.index = ranking
    th.columns = ['name', 'hits']

    ta = y2017.sort_values(by=['avg'], ascending=False).head(n=10)
    ta = ta[['batter_name', 'avg']]
    ta.index = ranking
    ta.columns = ['name', 'batting avg']

    thr = y2017.sort_values(by=['HR'], ascending=False).head(n=10)
    thr = thr[['batter_name', 'HR']]
    thr.index = ranking
    thr.columns = ['name', 'homerun']

    to = y2017.sort_values(by=['OBP'], ascending=False).head(n=10)
    to = to[['batter_name', 'OBP']]
    to.index = ranking
    to.columns = ['name', 'obp']

    print('< top 10 players in 2015 >')
    print('\n 1)hits')
    print(th)
    print('\n 2)battting average')
    print(ta)
    print('\n 3)homerun')
    print(thr)
    print('\n 4)on-base percentage')
    print(to)


def top2018():
    th = y2018.sort_values(by=['H'], ascending=False).head(n=10)
    th = th[['batter_name', 'H']]
    th.index = ranking
    th.columns = ['name', 'hits']

    ta = y2018.sort_values(by=['avg'], ascending=False).head(n=10)
    ta = ta[['batter_name', 'avg']]
    ta.index = ranking
    ta.columns = ['name', 'batting avg']

    thr = y2018.sort_values(by=['HR'], ascending=False).head(n=10)
    thr = thr[['batter_name', 'HR']]
    thr.index = ranking
    thr.columns = ['name', 'homerun']

    to = y2018.sort_values(by=['OBP'], ascending=False).head(n=10)
    to = to[['batter_name', 'OBP']]
    to.index = ranking
    to.columns = ['name', 'obp']

    print('< top 10 players in 2015 >')
    print('\n 1)hits')
    print(th)
    print('\n 2)battting average')
    print(ta)
    print('\n 3)homerun')
    print(thr)
    print('\n 4)on-base percentage')
    print(to)


def top10():
    top2015()
    top2016()
    top2017()
    top2018()


def high_war():
    catcher = y2018[y2018['cp'] == '포수'].sort_values(by=['war']).tail(n=1)
    catcher = catcher[['war', 'cp', 'batter_name']]

    fbm = y2018[y2018['cp'] == '1루수'].sort_values(by=['war']).tail(n=1)
    fbm = fbm[['war', 'cp', 'batter_name']]

    sbm = y2018[y2018['cp'] == '2루수'].sort_values(by=['war']).tail(n=1)
    sbm = sbm[['war', 'cp', 'batter_name']]

    tbm = y2018[y2018['cp'] == '3루수'].sort_values(by=['war']).tail(n=1)
    tbm = tbm[['war', 'cp', 'batter_name']]

    sstop = y2018[y2018['cp'] == '유격수'].sort_values(by=['war']).tail(n=1)
    sstop = sstop[['war', 'cp', 'batter_name']]

    lfd = y2018[y2018['cp'] == '좌익수'].sort_values(by=['war']).tail(n=1)
    lfd = lfd[['war', 'cp', 'batter_name']]

    cfd = y2018[y2018['cp'] == '중견수'].sort_values(by=['war']).tail(n=1)
    cfd = cfd[['war', 'cp', 'batter_name']]

    rfd = y2018[y2018['cp'] == '우익수'].sort_values(by=['war']).tail(n=1)
    rfd = rfd[['war', 'cp', 'batter_name']]

    result = pd.concat([catcher, fbm, sbm, tbm, sstop, lfd, cfd, rfd])
    print('< player with highest war by position in 2018 >')
    print(result)


def high_corr():
    all_d = kbo[['R', 'H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG', 'salary']]
    corr_with_s = pd.DataFrame(all_d.corr().loc['salary'])
    corr_with_s.drop('salary', inplace=True)
    print(corr_with_s)
    md = -1
    maxidx = 'H'
    for idx in corr_with_s.index:
        if md < corr_with_s.at[idx, 'salary']:
            md = corr_with_s.at[idx, 'salary']
            maxidx = idx

    high = corr_with_s.at[maxidx, 'salary']
    print('< highest correlation with salary among R,H,HR,RBI,SB,war,avg,OBP,SLG >')
    print(maxidx + ": ", high)


top10()
high_war()
high_corr()


