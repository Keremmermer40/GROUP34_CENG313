"""Yalnızca geliştirme verisi için keşif grafikleri."""
import matplotlib.pyplot as plt
import pandas as pd


def plot_target(frame):
    ax = frame['rule_violation'].value_counts().sort_index().plot.bar(color=['#5b8db8','#e19b54'])
    ax.set(title='Hedef dağılımı (geliştirme)', xlabel='İhlal etiketi', ylabel='Satır sayısı')
    ax.tick_params(axis='x', rotation=0)
    plt.tight_layout()
    return ax.figure


def rule_summary(frame):
    return frame.groupby('rule')['rule_violation'].agg(satir='size', ihlal_orani='mean')


def plot_rules(frame):
    table = rule_summary(frame).copy()
    table.index = [s.split(':')[0] for s in table.index]
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    table['satir'].plot.bar(ax=axes[0],color='#5b8db8',rot=15,title='Kural dağılımı')
    table['ihlal_orani'].plot.bar(ax=axes[1],color='#e19b54',rot=15,title='Kural başına ihlal oranı')
    axes[0].set_ylabel('Satır'); axes[1].set_ylim(0,1)
    fig.tight_layout()
    return fig


def plot_subreddits(frame):
    table=frame.groupby('subreddit')['rule_violation'].agg(satir='size', ihlal_orani='mean')
    table=table.nlargest(10,'satir').sort_values('satir')
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    table['satir'].plot.barh(ax=axes[0],title='En sık 10 subreddit',color='#5b8db8')
    table['ihlal_orani'].plot.barh(ax=axes[1],title='Aynı topluluklarda ihlal oranı',color='#e19b54')
    axes[1].set_xlim(0,1); fig.tight_layout()
    return fig


def plot_lengths(frame):
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for label in (0,1):
        values=frame.loc[frame['rule_violation']==label,'body'].fillna('').str.len()
        axes[0].hist(values,bins=20,alpha=.55,label=f'Etiket {label}')
    axes[0].legend(); axes[0].set(title='Yorum uzunluğu',xlabel='Karakter',ylabel='Satır')
    values=[frame.loc[frame['rule_violation']==label,'body'].fillna('').str.len() for label in (0,1)]
    axes[1].boxplot(values,tick_labels=['İhlal yok','İhlal var'])
    axes[1].set(title='Etikete göre uzunluk',ylabel='Karakter')
    fig.tight_layout(); return fig
