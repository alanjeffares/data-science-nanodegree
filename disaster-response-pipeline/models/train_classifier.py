import sys
import pandas as pd 
import nltk
import pickle
import os
from sqlalchemy import create_engine
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.base import BaseEstimator, TransformerMixin


def tokenize(text):

    tokens = nltk.tokenize.word_tokenize(text)
    lemmatizer = nltk.stem.WordNetLemmatizer()

    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]

    return clean_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
    
def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql('tweets', con=engine)
    X = df['message']
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])
    category_names = Y.columns
    return X, Y, category_names

def build_model():
    pipeline = Pipeline([
        ('features', FeatureUnion([

        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
        ])),
            ('starting_verb', StartingVerbExtractor())
        ])),
        ('clf', DecisionTreeClassifier())
    ])

    parameters = [
        {
            'features__text_pipeline__vect__max_df': (0.5, 1.0),
            'features__text_pipeline__vect__min_df': (1, 0.01),
            'features__text_pipeline__vect__max_features': (None, 5000),
            'features__text_pipeline__tfidf__use_idf': (True, False),
            'clf': (DecisionTreeClassifier(min_samples_split=3),),
            'clf__max_depth': (None, 4)
        }, {
            'features__text_pipeline__vect__max_df': (0.5, 1.0),
            'features__text_pipeline__vect__min_df': (1, 0.01),
            'features__text_pipeline__vect__max_features': (None, 5000),
            'features__text_pipeline__tfidf__use_idf': (True, False),
            'clf': (MultiOutputClassifier(LinearSVC(multi_class='ovr')),)
        }, {
            'features__text_pipeline__vect__max_df': (0.5, 1.0),
            'features__text_pipeline__vect__min_df': (1, 0.01),
            'features__text_pipeline__vect__max_features': (None, 5000),
            'features__text_pipeline__tfidf__use_idf': (True, False),
            'clf': (MLPClassifier(),),
            'clf__hidden_layer_sizes': ((100, 10), (50,), (50, 10))
        }
    ]

    cv = GridSearchCV(pipeline, parameters, cv=3, n_jobs=4, verbose=10)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    Y_pred = pd.DataFrame(Y_pred, columns=category_names)
    
    # calculate summary stats on test data
    results = pd.DataFrame()
    for column_name in Y_pred.columns:
        col_report = classification_report(y_true=Y_test[[column_name]], y_pred=Y_pred[[column_name]], output_dict=True)
        accuracy = col_report['accuracy']
        precision = col_report['macro avg']['precision']
        recall = col_report['macro avg']['recall']
        results[column_name] = [accuracy, precision, recall]
    results.index = ['accuracy', 'precision', 'recall']
    results.mean(axis=1) 
    
    # save results to local csv file
    model_name = type(model.best_params_['clf']).__name__
    avg_accuracy = results.mean(axis=1)['accuracy']
    avg_precision = results.mean(axis=1)['precision']
    avg_recall = results.mean(axis=1)['recall']
    params = model.best_params_
    stored_results = pd.DataFrame({'Model': [model_name], 'Accuracy': [avg_accuracy], 'Precision': [avg_precision], 
                               'Recall': [avg_recall], 'Parameters': [params]})

    add_header = not os.path.isfile('model_results.csv')
    with open('Model_results.csv', 'a') as f:
        stored_results.to_csv(f, header=add_header, index=False)


def save_model(model, model_filepath):
    with open(model_filepath,'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()
    