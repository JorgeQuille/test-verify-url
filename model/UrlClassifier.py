import pickle
import re

class UrlClassifier:

    models = {
        "Random Forest":'modelo_random_forest1',
        "Regresión Logística": 'modelo_regresion_logistica1',
        "SVM": 'modelo_SVM1'
    }
    
    def __init__(self):
        with open('vectorizer_param.pkl', 'rb') as f:
            self.vectorizer = pickle.load(f)
    
    def get_models(self):
        return list(self.models.keys())
    
    def normalize_url(self, url):
        url = re.sub(r'http[s]?://', '', url)
        url = re.sub(r'^www\.', '', url)
        url = url.lower()
        url = re.sub(r'[^a-z0-9/_-]', '', url)
        return url

    def predic(self, url, model_name):
        
        with open(f'{self.models.get(model_name)}.pkl', 'rb') as f:
            self.model = pickle.load(f)
        url = self.normalize_url(url)
        url_vectorizada = self.vectorizer.transform([url])
        prediction = self.model.predict(url_vectorizada)
        probability = self.model.predict_proba(url_vectorizada)[0][1]
        return prediction[0], probability