import pandas as pd
from pathlib import Path
import numpy as np  


import plotext as plt

if __name__ == "__main__":
    df = pd.read_csv("https://gist.githubusercontent.com/simplymathematics/8c6c04bd151950d5ea9e62825db97fdd/raw/34e546e4813f154d11d4f13869b9e3481fc3e829/kdd_nsl.csv")
    del df['difficulty_level']
    X = df.drop('label', axis=1)
    y = df['label']
    y = pd.DataFrame(y, columns=['label'])
    df = pd.concat([X, y], axis=1)
    df.to_csv("raw_data/kdd_nsl.csv", index=False)        
    # Find the number of entries for each label
    counts = pd.DataFrame(df['label']).value_counts().values
    labels = range(len(counts))
    # Plot the counts
    plt.simple_bar(labels, counts, title="KDD NSL Label Counts", width=50,)
    plt.show()
    df = pd.read_csv("https://gist.githubusercontent.com/simplymathematics/8c6c04bd151950d5ea9e62825db97fdd/raw/34e546e4813f154d11d4f13869b9e3481fc3e829/truthseeker.csv")
    X = df['tweet']
    label = 'BotScoreBinary'
    y = df[label]
    df = pd.concat([X, y], axis=1)
    df.to_csv("raw_data/truthseeker.csv", index=False)
    # Find the number of entries for each label
    counts = pd.DataFrame(df[label]).value_counts().values
    labels = range(len(counts))
    # Plot the counts
    plt.simple_bar(labels, counts, title="Truthseeker Label Counts", width=50,)
    plt.show()
    df = pd.read_csv("https://gist.githubusercontent.com/simplymathematics/8c6c04bd151950d5ea9e62825db97fdd/raw/c91944733b8f2b9a6ac0b8c8fab01ddcdf0898eb/sms-spam.csv")
    X = df['message']
    y = df['label']
    y = y.str.replace('ham', '0').replace('spam', '1')
    df = pd.concat([X, y], axis=1)
    df.to_csv("raw_data/sms-spam.csv", index=False)
    # Find the number of entries for each label
    counts = pd.DataFrame(df['label']).value_counts().values
    labels = range(len(counts))
    # Plot the counts
    plt.simple_bar(labels, counts, title="SMS Spam Label Counts", width=50,)
    plt.show()
    df = pd.read_csv("https://gist.githubusercontent.com/simplymathematics/8c6c04bd151950d5ea9e62825db97fdd/raw/712b528dcd212d5a6d1767332f50161fc1cfe55c/ddos.csv")
    # Find the number of entries for each label
    X = df.drop('Label', axis=1)
    y = df['Label']
    y = y.str.replace('Benign', '0').replace('ddos', '1')
    df = pd.concat([X, y], axis=1)
    df.to_csv("raw_data/ddos.csv", index=False)
    counts = pd.DataFrame(y).value_counts().values
    labels = range(len(counts))
    # Plot the counts
    plt.simple_bar(labels, counts, title="DDoS Label Counts", width=50,)
    plt.show()
    
    
    
        
        