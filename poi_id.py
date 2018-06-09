#!/usr/bin/python

import sys
import pickle

import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append("tools/")

# ================================================== #
#               Classifiers                          #
# ================================================== #
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
# ================================================== #
#               Scaler and feature selection         #
# ================================================== #
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from feature_format import featureFormat, targetFeatureSplit
# ================================================== #
#               Validation and Test                  #
# ================================================== #
from sklearn import cross_validation
from tester import dump_classifier_and_data
from sklearn.cross_validation import StratifiedShuffleSplit, train_test_split
from tester import dump_classifier_and_data, test_classifier
from sklearn.metrics import recall_score, precision_score, f1_score
# ===================================================#
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from time import time
import numpy as numpy

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'deferral_payments', 'total_payments',
                 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value',
                 'expenses', 'exercised_stock_options', 'long_term_incentive', 'restricted_stock',
                 'director_fees', 'to_messages', 'from_poi_to_this_person','from_messages',
                 'from_this_person_to_poi', 'shared_receipt_with_poi'] 

att_financial =  ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 
                    'bonus',  'restricted_stock_deferred', 'deferred_income', 
                    'total_stock_value', 'expenses', 'exercised_stock_options', 
                    'other', 'long_term_incentive', 'restricted_stock', 'director_fees']

att_email =  ['to_messages', 'email_address', 'from_poi_to_this_person', 
              'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

def funcionario_sem_dados_financeiros():
    df_financial = df[att_financial]
    index = 0
    print "funcionarios sem dados financeiros"
    for total in pd.isnull(df_financial).sum(axis = 1):
        if total == len(att_financial):
            print df.iloc[index]['nome']
        index += 1

def funcionario_sem_dados_email():
    df_mail = df[att_email]

    index = 0
    print "funcionario sem dados de email"
    for total in pd.isnull(df_mail).sum(axis = 1):
        if total == len(att_email):
            print df.iloc[index]['nome']
        index += 1

def explore_data():
    
    #Tamanho do dataset 
    print 'Existem %d pessoas no dataset' % len(df)

    #nome das colunas
    print df.columns

    #tipos dos dados
    print df.info()

    #dados ausentes
    print pd.isnull(df).sum()

    #Visao estatistica do dataset completo
    print df.describe()

    #Visao estatistica dos POI
    print df[df.poi.isin([True])].describe()

    #Visao estatistica dos nao POI
    print df[df.poi.isin([False])].describe()

    #Numero de atributos
    print len(df.columns) 

    #Numero de POIs
    print len(df[df.poi.isin([True])])
    #Numero de nao POIs
    print len(df[df.poi.isin([False])])

    funcionario_sem_dados_financeiros()

    funcionario_sem_dados_email()
    
def remove_outliers(dictionary):
    """Remove os outliers identificados na EDA.
    
    """
    try:
        del dictionary['TOTAL']
        del dictionary['THE TRAVEL AGENCY IN THE PARK']
        del dictionary['LOCKHART EUGENE E']
    except KeyError:
        print "Outliers ja foram removidos"

def computeFraction(poi_messages, all_messages):
    """ Calcula a fracao de mensagens enviadas e recebidas dos POIs.
    
    Args:
        poi_messages: total de mensagens relacionadas com os POIs
        all_messages: total das mensagens
        
    Returns:
        fracao das mensagens.
    """

    fraction = 0.
    if poi_messages != "NaN":
        fraction = float(poi_messages)/float(all_messages)

    return fraction

def compute_income(data_point):
    """Calcula o rendimento do funcionario.
    
    Args:
        data_point: dados de determinado funcionario.
    
    Returns:
        Rendimento de determinado funcionario.
        income = salary + bonus
    """
    salary = 0.
    bonus = 0.
    if data_point['salary'] != "NaN":
        salary = data_point['salary']
    if data_point['bonus'] != "NaN":
        bonus = data_point['bonus']
    return salary + bonus

def create_new_features(dictionary):
    """Cria novas features no dicionario.
    """
    for name in dictionary.keys():
        data_point = dictionary[name]
        #fracao de mensagens from_poi
        from_poi_to_this_person = data_point["from_poi_to_this_person"]
        to_messages = data_point["to_messages"]
        fraction_from_poi = computeFraction(from_poi_to_this_person, to_messages)
        dictionary[name]["fraction_from_poi"] = fraction_from_poi
        #fracao de mensagens to_poi
        from_this_person_to_poi = data_point["from_this_person_to_poi"]
        from_messages = data_point["from_messages"]
        fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
        dictionary[name]["fraction_to_poi"] = fraction_to_poi
        #income
        dictionary[name]['income'] = compute_income(data_point)
    features_list.append('fraction_to_poi')
    features_list.append('fraction_from_poi')
    features_list.append('income')
    

def select_best_features(n_features):
    """Seleciona as n melhores features de acordo com a variancia.
    
    Responsavel por selecionar as n melhores features com base 
    na analise de variancia.
    
    Args:
        n_features: numero de features que deseja
        features: features de treinamento.
        labels: labels de treinamento.
    
    Returns:
        
    """
    data = featureFormat(data_dict, features_list, remove_all_zeroes = False, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=42)
    selector = SelectKBest(k = n_features)  
    selector.fit_transform(features, labels)
    selected_indices = selector.get_support(indices=True)
    final_features = []
    for indice in selected_indices:
        #print 'feature -> {} with score -> {}'.format(features_list[indice + 1], selector.scores_[indice])
        final_features.append(features_list[indice + 1])
    return final_features

def selectKBest_f1_scores(clf, dataset, n_kbest_features, folds = 1000):
    """ Verifica os scores do numero de features selecionadas.
    
    Responsavel por selecionar o score F1 de 2 ate n_kbest_features.
    
    Args: 
        clf: classificador utilizado para a analise
        dataset: dados utilizados
        n_kbest_features: numero de maximo de features permitido.
        
    Returns:
        retorno1: Lista de valores K
        retorno2: Lista de Scores F1
    """
    graficoX = []
    graficoY = []
    for k in range(2, n_kbest_features):
        features_selected = select_best_features(k)
        features_selected.insert(0, "poi")
        data = featureFormat(dataset, features_selected, sort_keys = True)
        labels, features = targetFeatureSplit(data)
        cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
        true_negatives = 0
        false_negatives = 0
        true_positives = 0
        false_positives = 0
        for train_idx, test_idx in cv: 
            features_train = []
            features_test  = []
            labels_train   = []
            labels_test    = []
            for ii in train_idx:
                features_train.append( features[ii] )
                labels_train.append( labels[ii] )
            for jj in test_idx:
                features_test.append( features[jj] )
                labels_test.append( labels[jj] )

            clf.fit(features_train, labels_train)
            predictions = clf.predict(features_test)
            for prediction, truth in zip(predictions, labels_test):
                if prediction == 0 and truth == 0:
                    true_negatives += 1
                elif prediction == 0 and truth == 1:
                    false_negatives += 1
                elif prediction == 1 and truth == 0:
                    false_positives += 1
                elif prediction == 1 and truth == 1:
                    true_positives += 1
                else:
                    print "Warning: Found a predicted label not == 0 or 1."
                    print "All predictions should take value 0 or 1."
                    print "Evaluating performance for processed predictions:"
                    break
        try:
            f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
            graficoY.append(f1)
            graficoX.append(k)
        except:
            print "Got a divide by zero when trying out:", clf
            print "Precision or recall may be undefined due to a lack of true positive predicitons."
    return  graficoX, graficoY
        

#O modelo mais promissor foi o AdaBoost! Para facilitar o processo de tuning, 
# irei automatizar o processo com a construcao de uma pipeline, 
# afim de realizar os ajustes necessarios para melhorar a precisao e o recall.
def get_classifier_and_param_grid(classifier):
    """Devolve o classificador informado e seu param_grid definido.
    
    Responsavel por retornar uma instancia do classificador
    especificado e seus parametros definidos.
    
    Classificadores definidos: adaboost, random_forest e decision_tree.
    
    Args: 
        classifier: nome do classificador desejado.
        
    Returns:
        return 1: GaussianNB()
        return 2: param_grid
    
    """
    return {
        'adaboost': get_adaboost_classifier(),
        'random_forest': get_random_forest_classifier(),
        'decision_tree': get_decision_tree_classifier(),
    }.get(classifier)

def get_adaboost_classifier():
    """Devolve uma instancia do Adaboost e o param_grid definido.
    
    Returns:
        return 1: AdaBoostClassifier()
        return 2: param_grid
    
    """
    decision_tree = []
    for index in range(1, 5):
        decision_tree.append(DecisionTreeClassifier(max_depth=(index), 
                                                    class_weight='balanced',
                                                    min_samples_leaf=2))
    param_grid = {"adaboost__base_estimator": decision_tree,
                  "adaboost__n_estimators": [50, 55, 70],
                  "adaboost__learning_rate": [0.1, 1]
                  }    
    
    return AdaBoostClassifier(), param_grid

def get_random_forest_classifier():
    """Devolve uma instancia da RandomForestClassifier e o param_grid definido.
    
    Returns:
        return 1: RandomForestClassifier()
        return 2: param_grid
    
    """
    param_grid = {"random_forest__max_depth": [3, None],
                  "random_forest__max_features": range(1, 5),
                  "random_forest__min_samples_split": [2, 4]
                  }
    
    return RandomForestClassifier(), param_grid

def get_decision_tree_classifier():
    """Devolve uma instancia da DecisionTreeClassifier e o param_grid definido.
    
    Returns:
        return 1: DecisionTreeClassifier()
        return 2: param_grid
    
    """
    param_grid = {"decision_tree__max_depth": range(1, 5),
                  "decision_tree__min_samples_leaf": range(1, 5),
                  "decision_tree__min_samples_split": range(2, 5)
                  }
        
    return DecisionTreeClassifier(), param_grid

def build_pipeline(labels, features, classifier_name):
    """ Constroi uma pipeline com GridSearchCV.
    
    Constroi uma pipeline a fim de obter o melhor ajuste 
    para o classificador informado.
    
    Args: 
        labels: labels de treinamento
        features: features de treinamento
        classifier_name: nome do classificador.
        
    Returns:
        GridSearchCV com as informacoes da melhor configuracao.
    
    """
    # Necessario porque o dataset eh pequeno.
    sss = StratifiedShuffleSplit(labels, 100, random_state = 42)
    
    classifier, param_grid = get_classifier_and_param_grid(classifier_name)
    
    pipeline  = Pipeline([('scaler',  MinMaxScaler()),
                         ('feature_selection', SelectKBest(k = 10)),
                         (classifier_name, classifier),])

    grid_search = GridSearchCV(pipeline, param_grid, scoring = 'f1', cv = sss)

    grid_search.fit(features, labels)

    return grid_search

def best_estimator_by_classifiers(classifiers_names):
    """ Constroi um dicionario com o melhor ajuste para cada classificador.
    
    Responsavel por construir um dicionario com todos os classificadores
    informados. Onde a chave eh o nome do classificador e o valor eh o 
    melhor ajuste encontrado no GridSearch.
    
    Args:
        classifiers_names: lista com os nomes dos classificadores.
        
    Returns:
        Dicionario com o nome do classificador como key e o melhor
        ajuste como valor
    
    """
    best_estimators = {}
    for name in classifiers_names:
        best_estimators[name] = build_pipeline(labels_train, features_train, name)
    return best_estimators


### Store to my_dataset for easy export below.
my_dataset = data_dict

#########################################################################################
#                               Inicio da execucao                                      #
#########################################################################################
#Convertendo o dicionario em um dataframe
df = pd.DataFrame.from_records(my_dataset).T.reset_index().rename({'index': 'nome'}, axis=1)
df.replace(to_replace='NaN', value=numpy.nan, inplace=True)
#########################################################################################
#                               Explora os dados                                        #
#########################################################################################
explore_data()
#########################################################################################
#                               Remove Outliers                                         #
#########################################################################################
remove_outliers(my_dataset)
#########################################################################################
#                               Cria novas features                                     #
#########################################################################################
create_new_features(my_dataset)
#########################################################################################
#                               Seleciona as melhores features                          #
#########################################################################################
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

x, y = selectKBest_f1_scores(GaussianNB(), data_dict, 20)
plt.figure()
plt.xlabel("Numero de features selecionadas")
plt.ylabel("Valor de F1-Score")
plt.plot(x, y)
plt.savefig('featureSelection.png', transparent=True)
plt.show()


features_list =  select_best_features(17)
features_list.insert(0, "poi")
print features_list
#########################################################################################
#                               Escolha e Afinamento de um Modelo                       #
#########################################################################################
classifiers = ['adaboost', 'random_forest', 'decision_tree']
#########################################################################################
#                               Testa DecisionTree                                      #
#########################################################################################
clf_dc = DecisionTreeClassifier()
t0 = time()
test_classifier(clf_dc, data_dict, features_list, folds = 100)
print("Tempo de ajuste da DecisionTree:", round(time()-t0, 3), "s")
#########################################################################################
#                               Testa RandonForest                                      #
#########################################################################################
clf_rf = RandomForestClassifier()
t0 = time()
test_classifier(clf_rf, data_dict, features_list, folds = 100)
print("Tempo de ajuste da RandomForest:", round(time()-t0, 3), "s")
#########################################################################################
#                               Testa Adaboost                                          #
#########################################################################################
clf_ab = AdaBoostClassifier()
t0 = time()
test_classifier(clf_ab, data_dict, features_list, folds = 100)
print("Tempo de ajuste do Adaboost:", round(time()-t0, 3), "s")
#########################################################################################
#                  ******************************                                       #
#                  * Modelo Selecionado Adaboost*                                       #
#                  ******************************                                       #
#########################################################################################
#########################################################################################
#                   Pesquisa no gridSearch o melhor ajuste dos classificadores          #
#########################################################################################
#dict_cv_by_classifier =  best_estimator_by_classifiers(classifiers)
print "Inicio adaboost Ajustada"
dict_cv_by_classifier =  best_estimator_by_classifiers(['adaboost'])
#########################################################################################
#                   Teste Adaboost ajustada                                             #
#########################################################################################
#clf = dict_cv_by_classifier['adaboost']
#test_classifier(clf.best_estimator_, data_dict, features_list)
#Logo abaixo esta o melhor ajuste encontrado para o Adaboost
clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1, min_samples_leaf=2, class_weight='balanced'),
                         n_estimators=50, learning_rate=.1)
test_classifier(clf, my_dataset, features_list)
#########################################################################################
#                   Salva o classificador                                               #
#########################################################################################
dump_classifier_and_data(clf, my_dataset, features_list)