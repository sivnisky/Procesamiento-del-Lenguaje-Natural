# Procesamiento del Lenguaje Natural - ITBA
# Trabajo práctico

# 1° Importar las librerías a utilizar
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as ss
import re
import time
import string
import nltk
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
import spacy
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from matplotlib import pyplot as plt
color = sns.color_palette()

# 2° Cargar los datos
train = pd.read_csv("/Users/sofiaivnisky/Desktop/train.csv")
test = pd.read_csv("/Users/sofiaivnisky/Desktop/test.csv")

# 3° Ver los datos
print(train.head())
print(test.head())

# 4° Explorar los datos (EDA)
print("Información de train.csv: ",train.info())
print("Información de test.csv: ",test.info())
print("Descripción de train.csv: ",train.describe())
print("Descripción de test.csv: ",test.describe())
print("Largo de las bd: ")
print("Train: ",len(train))
print("Test: ",len(test))

# Media, desviación estándar y valor máximo
length = train.comment_text.str.len()
print("Media: ", length.mean())
print("Desviación estándar: ", length.std())
print("Máximo: ", length.max())

# Largo de los comentarios
# Histograma sobre el largo de los comentarios
np.histogram(length,bins = [0,1000,2000,3000,4000,5000])
hist, bins = np.histogram(length,bins = [0,1000,2000,3000,4000,5000])
print(hist)
print(bins)
plt.hist(length, bins = [0,1000,2000,3000,4000,5000])
plt.title("Histograma sobre el largo de los comentarios")
plt.show()

# Marcar a los comentarios sin etiqueta como "clean"
x = train.iloc[:,2:].sum() #En la columna 2 están los comentarios
rowsums = train.iloc[:,2:].sum(axis=1)
train['clean']=(rowsums==0)
#Contar la cantidad de "cleans"
print("Total de comentarios = ", len(train))
print("Total de comentarios 'clean' = ", train['clean'].sum())
print("Total de etiquetas =", x.sum())
# Valores NULL
print("Buscar missing values en train.csv")
null_check=train.isnull().sum()
print(null_check)
print("Buscar missing values en test.csv")
null_check=test.isnull().sum()
print(null_check)

# N° de ocurrencias por clase
x = train.iloc[:,2:].sum()
# Histograma
plt.figure(figsize=(8,4))
ax = sns.barplot(x = x.index, y = x.values, alpha = 0.8)
plt.title("N° de ocurrencias por clase")
plt.ylabel('N°', fontsize=12)
plt.xlabel('Clase ', fontsize=12)
# Etiquetas de texto
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
plt.show()

# Comentarios con múltiples etiquetas
x = rowsums.value_counts()
# Gráfico de barras
plt.figure(figsize=(8,4))
ax = sns.barplot(x = x.index, y = x.values, alpha = 0.8)
plt.title("Múltiples etiquetas por comentario")
plt.ylabel('Ocurrencias', fontsize=12)
plt.xlabel('Etiquetas ', fontsize=12)
# Etiquetas
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
plt.show()

# Queremos saber de que forma se suelen agrupar las etiquetas
# Diagrama de correlación
temp_df = train.iloc[:,2:-1]
# Remover los comentarios 'clean'
corr = temp_df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values, annot=True)
plt.show()
#Como las variables son categóricas, no se si este es el gráfico más apropiado
#Podría utilizar en cambio una Confusion matrix/Crosstab (no me salió) o matriz de Cramers
#Confusion matrix/Crosstab #Ver que hago con esto, no me salió (error: Jinja2)
#Cramer's
def cramers_corrected_stat(confusion_matrix):
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))
#Toxic y Severe toxic
col1 = "toxic"
col2 = "severe_toxic"
confusion_matrix = pd.crosstab(temp_df[col1], temp_df[col2])
print("Matriz entre Toxic y Severe toxic :")
print(confusion_matrix)
new_corr=cramers_corrected_stat(confusion_matrix)
print("La correlación entre Toxic y Severe toxic usando la regla de Cramer es =",new_corr)

#Threat e Identity hate
col4 = "threat"
col6 = "identity_hate"
confusion_matrix = pd.crosstab(temp_df[col4], temp_df[col6])
print("Matriz entre Threat e Identity hate:")
print(confusion_matrix)
new_corr=cramers_corrected_stat(confusion_matrix)
print("La correlación entre Threat e Identity hate usando la regla de Cramer es =",new_corr)

#Threat y Toxic
col4 = "threat"
col1 = "toxic"
confusion_matrix = pd.crosstab(temp_df[col4], temp_df[col1])
print("Matriz entre Threat y Toxic:")
print(confusion_matrix)
new_corr=cramers_corrected_stat(confusion_matrix)
print("La correlación entre Threat y Toxic usando la regla de Cramer es =",new_corr)

#Etc
#Después de estas matrices creo que la matriz de correlación inical es útil en este caso

# Para los siguientes análisis se debe tener una base de datos más "limpia" en cuanto al contenido de los comentarios
#Vamos a crear algunas funciones basadas en la distribución de frecuencia de las palabras. Inicialmente, se toman palabras una a la vez (Unigrams)
#SKlearn de Python: crea un diccionario de palabras y luego una matriz dispersa de conteo de palabras.
merge = pd.concat([train.iloc[:,0:2],test.iloc[:,0:2]])
corpus = merge.comment_text
#Usar el marco de datos concatenado "merge" que contiene tanto el contenido de train como el de test para garantizar que el vocabulario que creamos no pierda las palabras que son exclusivas del conjunto test.
APPO = {
"aren't" : "are not",
"can't" : "cannot",
"couldn't" : "could not",
"didn't" : "did not",
"doesn't" : "does not",
"don't" : "do not",
"hadn't" : "had not",
"hasn't" : "has not",
"haven't" : "have not",
"he'd" : "he would",
"he'll" : "he will",
"he's" : "he is",
"i'd" : "I would",
"i'd" : "I had",
"i'll" : "I will",
"i'm" : "I am",
"isn't" : "is not",
"it's" : "it is",
"it'll":"it will",
"i've" : "I have",
"let's" : "let us",
"mightn't" : "might not",
"mustn't" : "must not",
"shan't" : "shall not",
"she'd" : "she would",
"she'll" : "she will",
"she's" : "she is",
"shouldn't" : "should not",
"that's" : "that is",
"there's" : "there is",
"they'd" : "they would",
"they'll" : "they will",
"they're" : "they are",
"they've" : "they have",
"we'd" : "we would",
"we're" : "we are",
"weren't" : "were not",
"we've" : "we have",
"what'll" : "what will",
"what're" : "what are",
"what's" : "what is",
"what've" : "what have",
"where's" : "where is",
"who'd" : "who would",
"who'll" : "who will",
"who're" : "who are",
"who's" : "who is",
"who've" : "who have",
"won't" : "will not",
"wouldn't" : "would not",
"you'd" : "you would",
"you'll" : "you will",
"you're" : "you are",
"you've" : "you have",
"'re": " are",
"wasn't": "was not",
"we'll":" will",
"didn't": "did not",
"tryin'":"trying"
}
#Crear la función clean que recibirá comentarios y devolverá una lista de palabras "limpias"
def clean(comment):
    # Convertimos todo a minúsculas para evitar problemas
    comment = comment.lower()
    # Sacamos los "\n"
    comment = re.sub("\\n", "", comment)
    # Sacamos elementos como ip,user
    comment = re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", "", comment)
    # Sacamos nombres de usuario
    comment = re.sub("\[\[.*\]", "", comment)
    # Dividir las oraciones en palabras
    tokenizer = TweetTokenizer() #Tweet Tokenizer no divide en apóstrofes
    lem = WordNetLemmatizer()
    eng_stopwords = set(stopwords.words("english"))
    words = tokenizer.tokenize(comment)
    words = [APPO[word] if word in APPO else word for word in words]
    words = [lem.lemmatize(word, "v") for word in words]
    words = [w for w in words if not w in eng_stopwords]
    #Limpiar
    clean_sent = " ".join(words)
    return(clean_sent)
# Ejemplo de clean
print(corpus.iloc[12235])
print(clean(corpus.iloc[12235]))

# Vectorizer
#Vectorizador TF-IDF
#TF - Frecuencia de términos -- Recuento de las palabras (Términos) en el corpus de texto (igual que Count Vect)
#IDF - Frecuencia de documento inversa: penaliza las palabras que son demasiado frecuentes (regularización).
clean_corpus=corpus.apply(lambda x :clean(x))
start_unigrams = time.time()
tfv = TfidfVectorizer(min_df=200,  max_features=10000,
            strip_accents='unicode', analyzer='word',ngram_range=(1,1),
            use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')
tfv.fit(clean_corpus)
features = np.array(tfv.get_feature_names())

train_unigrams =  tfv.transform(clean_corpus.iloc[:train.shape[0]]) #De una palabra a la vez
test_unigrams = tfv.transform(clean_corpus.iloc[train.shape[0]:])


def top_tfidf_feats(row, features, top_n=25):
    #Trae los valores tfidf top de una fila con sus correcpondientes feature names
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df


def top_feats_in_doc(Xtr, features, row_id, top_n=25):
    #Devuelve las características top de tfidf
    row = np.squeeze(Xtr[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)


def top_mean_feats(Xtr, features, grp_ids, min_tfidf=0.1, top_n=25):
    #Devuelve las top n características que en promedio son mas importantes
    D = Xtr[grp_ids].toarray()
    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)

def top_feats_by_class(Xtr, features, min_tfidf=0.1, top_n=20):
    train_tags = train.iloc[:, 2:]
    #Devuelve una lista de dfs, donde cada df tiene características y su valor tfidf medio.
    dfs = []
    cols = train_tags.columns
    for col in cols:
        ids = train_tags.index[train_tags[col]==1]
        feats_df = top_mean_feats(Xtr, features, ids, min_tfidf=min_tfidf, top_n=top_n)
        feats_df.label = label
        dfs.append(feats_df)
    return dfs
# Obtener un top n por unigrams
tfidf_top_n_por_clase = top_feats_by_class(train_unigrams,features)
start_time = time.time()
end_unigrams = time.time()

print("Tiempo total en unigrams",end_unigrams-start_unigrams)
print("Tiempo total en unigrams",end_unigrams-start_time)

#Gráficos
plt.figure(figsize=(16,22))
plt.suptitle("TF_IDF Top palabras por clase(unigrams)",fontsize=20)
gridspec.GridSpec(4,2)
plt.subplot2grid((4,2),(0,0))
sns.barplot(x = tfidf_top_n_por_clase[0].feature.iloc[0:9], y = tfidf_top_n_por_clase[0].tfidf.iloc[0:9],color=color[0])
plt.title("Clase : Toxic",fontsize=15)
plt.xlabel('Palabra', fontsize=12)
plt.ylabel('TF-IDF score', fontsize=12)

plt.subplot2grid((4,2),(0,1))
sns.barplot(x = tfidf_top_n_por_clase[1].feature.iloc[0:9],y = tfidf_top_n_por_clase[1].tfidf.iloc[0:9],color=color[1])
plt.title("Clase : Severe toxic",fontsize=15)
plt.xlabel('Palabra', fontsize=12)
plt.ylabel('TF-IDF score', fontsize=12)
plt.subplot2grid((4,2),(1,0))
sns.barplot(x = tfidf_top_n_por_clase[2].feature.iloc[0:9],y = tfidf_top_n_por_clase[2].tfidf.iloc[0:9],color=color[2])
plt.title("Clase : Obscene",fontsize=15)
plt.xlabel('Palabra', fontsize=12)
plt.ylabel('TF-IDF score', fontsize=12)


plt.subplot2grid((4,2),(1,1))
sns.barplot(x = tfidf_top_n_por_clase[3].feature.iloc[0:9], y = tfidf_top_n_por_clase[3].tfidf.iloc[0:9],color=color[3])
plt.title("Clase : Threat",fontsize=15)
plt.xlabel('Palabra', fontsize=12)
plt.ylabel('TF-IDF score', fontsize=12)


plt.subplot2grid((4,2),(2,0))
sns.barplot(x = tfidf_top_n_por_clase[4].feature.iloc[0:9],y = tfidf_top_n_por_clase[4].tfidf.iloc[0:9],color=color[4])
plt.title("Clase : Insult",fontsize=15)
plt.xlabel('Palabra', fontsize=12)
plt.ylabel('TF-IDF score', fontsize=12)
plt.subplot2grid((4,2),(2,1))
sns.barplot(x = tfidf_top_n_por_clase[5].feature.iloc[0:9],y = tfidf_top_n_por_clase[5].tfidf.iloc[0:9],color=color[5])
plt.title("Clase : Identity hate",fontsize=15)
plt.xlabel('Palabra', fontsize=12)
plt.ylabel('TF-IDF score', fontsize=12)


plt.subplot2grid((4,2),(3,0),colspan=2)
sns.barplot(x = tfidf_top_n_por_clase[6].feature.iloc[0:19],y = tfidf_top_n_por_clase[6].tfidf.iloc[0:19])
plt.title("Clase : Clean",fontsize=15)
plt.xlabel('Palabra', fontsize=12)
plt.ylabel('TF-IDF score', fontsize=12)

plt.show()