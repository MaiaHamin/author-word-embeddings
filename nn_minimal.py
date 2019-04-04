def clean(s):
    return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()

def w_tokenize(s):
    return nltk.word_tokenize(s)

def s_tokenize(p):
    return nltk.sent_tokenize(p)

def lemmatize(word_tokens):
    return [lemmatizer.lemmatize(t) for t in word_tokens]

def remove_stopwords(word_tokens):
    return [w for w in word_tokens if not w in stop_words]

def w_super_clean(s):
    return remove_stopwords(lemmatize(w_tokenize(clean(s))))

def s_super_clean(p):
    sentences = s_tokenize(p)
    clean_sentences = []
    for s in sentences:
        clean_sentences.append(" ".join(remove_stopwords(lemmatize(w_tokenize(clean(s))))))
    return clean_sentences
