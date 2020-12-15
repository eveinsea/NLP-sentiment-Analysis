import string
import nltk
#nltk.download('stopwords')         #download once
#nltk.download('wordnet')           #download once
from nltk.corpus import stopwords
import pickle


class BaseModel:
    def __init__(self):
        self.lemmatizer = nltk.WordNetLemmatizer()
        self.stop = stopwords.words('english')
        self.translation = str.maketrans(string.punctuation,' '*len(string.punctuation))

        self.model = None
        self.vec = None

    # Load Vec
    def load_vec(self, vec_path, mode='rb'):
        with open(vec_path, mode) as pkl_file:
            self.vec = pickle.load(pkl_file)

    # Load Model
    def load_model(self, model_path, mode='rb'):
        with open(model_path, mode) as pkl_file:
            self.model = pickle.load(pkl_file)

    # Preprocessing
    def preprocessing(self, line: str) -> str:
        line = str(line).translate(self.translation)
        line = nltk.word_tokenize(line.lower())

        line = [self.lemmatizer.lemmatize(t) for t in line if t not in self.stop]
        return ' '.join(line)

    # Predict
    def predict(self, line):
        if self.model is None or self.vec is None:
            print("Model / Vec no loaded")
            return ""

        line = self.preprocessing(line)
        features = self.vec.transform([line])

        return self.model.predict(features)[0]
