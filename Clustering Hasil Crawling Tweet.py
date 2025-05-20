import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Load data
df = pd.read_csv('C:/Kelompok1_22230020.csv')
tweets = df['Isi Tweet'].dropna().tolist()

# Preprocessing ringan + hapus stopword Indonesia
factory = StopWordRemoverFactory()
stop_words = set(factory.get_stop_words())

def preprocess(text):
    text = re.sub(r"http\S+", "", text)        # hapus link
    text = re.sub(r"[^a-zA-Z\s]", "", text)     # hapus simbol
    text = text.lower()
    tokens = text.split()
    return ' '.join([w for w in tokens if w not in stop_words])

tweets_cleaned = [preprocess(t) for t in tweets]

# Vektorisasi
vectorizer = TfidfVectorizer(max_df=0.8, min_df=2)
X = vectorizer.fit_transform(tweets_cleaned)

# Clustering
k = 30
model = KMeans(n_clusters=k, random_state=42)
model.fit(X)

# Simpan hasil clustering ke dalam DataFrame
clustered_data = []
for i, label in enumerate(model.labels_):
    clustered_data.append({
        'Tweet': tweets[i],
        'Cluster': label
    })

# Buat DataFrame dari hasil clustering
clustered_df = pd.DataFrame(clustered_data)

# Simpan ke file CSV
output_file = 'hasil_clustering.csv'
clustered_df.to_csv(output_file, index=False)
print(f"Hasil clustering disimpan di {output_file}")
