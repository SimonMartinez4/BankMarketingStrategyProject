# -*- coding: utf-8 -*-
"""

@author: Simon Martinez
"""

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

st.title("Prédictions d'une campagne marketing d'une banque")
df = pd.read_csv("bank.csv")

st.sidebar.title("Sommaire")
pages = ["Visualisation des données"]
page = st.sidebar.radio("Aller à la page", pages)

if page == pages[0] : 
    st.write("## II. Visualisation des données")
    st.write("### A. Visualisation générale")
    st.write("#### 1. Analyse de la variable cible")
    
    # Création du graphique de la variable cible avec Plotly et personnalisation
    pie_y = px.pie(df, names="deposit", title="Distribution de la variable cible deposit", color='deposit',
    color_discrete_map={'yes': 'salmon', 'no': 'skyblue'},
    labels={'yes': 'Oui', 'no': 'Non'},
    hole=0.4)

    # Affichage du graphique
    st.plotly_chart(pie_y)
    
    # Ajout d'un encadré déroulant "Ce qu'on peut noter :"
    with st.expander("Ce qu'on peut noter :"):
        st.write("La répartition de la variable cible est équilibrée.")

    st.write("#### 2. Analyse des variables explicatives")
    # Exclure la dernière colonne de la liste des variables explicatives
    excluded_columns = [df.columns[-1]]
    available_columns = [col for col in df.columns if col not in excluded_columns]

    # Sélection de la variable explicative
    selected_variable = st.selectbox("Sélectionnez une variable explicative", available_columns)
    
    # Vérifier le type de variable
    variable_type = df[selected_variable].dtype

    # Sélection du type de graphique
    if variable_type == 'int64' or variable_type == 'float64':
        chart_type = st.selectbox("Sélectionnez le type de graphique", ["Histogramme", "Boxplot"])
    else : chart_type = st.selectbox("Sélectionnez le type de graphique", ["Camembert"])

    # Création du graphique en fonction des sélections de l'utilisateur
    fig = None
    if variable_type == 'int64' or variable_type == 'float64':
        # C'est une variable continue
        if chart_type == "Histogramme":
            fig = px.histogram(df, x=selected_variable, title=f'Histogramme de {selected_variable}')
            fig.update_xaxes(title_text=f'{selected_variable}', showgrid=True, gridcolor='lightgray')
            fig.update_yaxes(title_text='Fréquence', showgrid=True, gridcolor='lightgray')
        
        elif chart_type == "Boxplot":
            fig = px.box(df, x=selected_variable, title=f'Boxplot de {selected_variable}')
            fig.update_xaxes(title_text=f'{selected_variable}', showgrid=True, gridcolor='lightgray')
    else:
    # C'est une variable catégorielle
        fig = px.pie(df, names=selected_variable, title=f'Graphique en camembert de {selected_variable}')
        fig.update_traces(textinfo='label+percent', pull=[0.2, 0], textposition='inside')

    # Affichage du graphique
    st.plotly_chart(fig)
    
    # Ajout d'un encadré déroulant "Ce qu'on peut noter :"
    with st.expander("Ce qu'on peut noter :"):
        st.markdown("- duration : les valeurs supérieures à 3000 secondes considérées comme valeurs aberrantes")
        st.markdown("- campaign : les valeurs supérieures à 10 sont aberrantes")
        st.markdown("- job : modalité unknown pour 0.6% des valeurs ; elle sera remplacée par le mode")
        st.markdown("- education : modalité unknown pour 4% des valeurs ; elle sera remplacée par le mode")
        st.markdown("- contact : cette variable ne comporte pas d'information utile ; elle sera supprimée")
        st.markdown("- poutcome : modalité unknown pour 75% des valeurs ; signifie l'absence de sollicitation lors d'une campagne antérieure ; sera remplacé par other")
        st.markdown("- pdays et previous : les modalités respectives -1 et 0 signifient l'absence de sollicitation lors d'une campagne antérieure")
        st.markdown("- pdays et previous : les modalités respectives -1 et 0 signifient l'absence de sollicitation lors d'une campagne antérieure")
        st.markdown("- previous : valeurs supérieures à 10 considérées comme valeurs aberrantes")
    
    # Analyse multivariée
    st.write("#### 3. Analyse en fonction de la variable cible")

    # Sélection de la variable explicative
    widget_key = "selectbox_variable_explicative"  # Clé unique pour le widget
    variable_explicative = st.selectbox("Sélectionnez une variable explicative", df.columns[:-1], key=widget_key)  # Exclure la variable cible 'deposit'

    # Personnalisation des couleurs
    couleur_saumon = '#FF6347'  # Saumon
    couleur_bleu_ciel = '#87CEEB'  # Bleu ciel

    # Création du graphique en fonction de la nature de la variable explicative
    if df[variable_explicative].dtype == 'O':  # Si la variable explicative est catégorielle
        fig = px.histogram(df, x=variable_explicative, color='deposit', barmode='stack',
                       color_discrete_map={'yes': couleur_saumon, 'no': couleur_bleu_ciel},
                       title=f"Count Plot pour {variable_explicative}")

    # Personnalisation supplémentaire pour le count plot catégoriel
        fig.update_layout(
            xaxis=dict(title=variable_explicative),
            yaxis=dict(title="Fréquence"),
            legend_title="deposit",
            barmode='stack',
            bargap=0.1,  # Espacement entre les barres
            bargroupgap=0.2  # Espacement entre les groupes de barres
    )

    elif df[variable_explicative].dtype in ['int64', 'float64']:  # Si la variable explicative est continue
        fig = px.violin(df, x='deposit', y=variable_explicative, box=True, points="all",
                    violinmode='overlay', color='deposit', color_discrete_map={'yes': couleur_saumon},
                    title=f"Violon Plot pour {variable_explicative}")

        # Personnalisation supplémentaire pour le violin plot continu
        fig.update_layout(
        xaxis=dict(title="deposit"),
        yaxis=dict(title=variable_explicative),
        legend_title="deposit",
        violinmode='overlay'
    )

    # Affichage du graphique
    st.plotly_chart(fig)
    
    # Ajout d'un encadré déroulant "Ce qu'on peut noter :"
    with st.expander("Ce qu'on peut noter :"):
        st.markdown("- age : différence significative de la représentation des personnes âgées de plus de 60 ans entre les clients souscripteurs et les autres")
        st.markdown("- housing : clients souscripteurs surreprésentés pour la modalité no par rapport à yes")
        st.markdown("- duration : durées d'appel plus longue chez les clients souscripteurs")
        st.markdown("- marital : les célibataires sont plus souvent souscripteurs")
        st.markdown("- job : étudiants, retraités, managers et chômeurs souscrivent davantage")
        st.markdown("- education : les clients ayant fait des études supérieures souscrivent davantage")
        st.markdown("- poutcome : les clients ayant souscrit lors d'une campagne précédente souscrivent davantage")

    # Matrice de correlation
    st.write("#### 4. Matrice de correlation")
    
    df_num=df.select_dtypes(include=['number'])
    correlation_matrix = df_num.corr()

    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale=['skyblue', 'salmon']
        ))

    st.plotly_chart(fig)

    # Tests statistiques
    st.write("#### 5. Tests d'indépendance entre variables")

    # Sélection des deux variables pour le test
    variable1 = st.selectbox("Sélectionnez la première variable", df.columns)
    variable2 = st.selectbox("Sélectionnez la deuxième variable", df.columns)

    # Effectuer le test d'indépendance approprié en fonction du type de variables
    if df[variable1].dtype in ['int64', 'float64'] and df[variable2].dtype in ['int64', 'float64']:
        # Test de corrélation de Pearson pour les variables continues
        test_stat, p_value = stats.pearsonr(df[variable1], df[variable2])
        test_type = "Test de corrélation de Pearson"
    elif (df[variable1].dtype in ['int64', 'float64'] and df[variable2].dtype == 'O') or (df[variable1].dtype == 'O' and df[variable2].dtype in ['int64', 'float64']):
        # Test ANOVA pour une variable quantitative et une variable qualitative
        groups = [df[variable1][df[variable2] == group] for group in df[variable2].unique()]
        test_stat, p_value = stats.f_oneway(*groups)
        test_type = "Test ANOVA"
    else:
        # Test du chi2 pour les variables catégorielles
        contingency_table = pd.crosstab(df[variable1], df[variable2])
        test_stat, p_value, _, _ = stats.chi2_contingency(contingency_table)
        test_type = "Test du chi2"

    # Afficher les résultats du test
    st.write(f"Type de test : {test_type}")
    st.write(f"P-value : {p_value}")

    # Interpréter les résultats en fonction de la p-value
    alpha = 0.05
    if p_value < alpha:
        st.write("La p-value est inférieure au seuil de signification (alpha), nous rejetons donc l'hypothèse nulle.")
        st.write("Il existe une dépendance significative entre les deux variables.")
    else:
        st.write("La p-value est supérieure au seuil de signification (alpha), nous ne pouvons pas rejeter l'hypothèse nulle.")
        st.write("Il n'y a pas suffisamment de preuves pour affirmer une dépendance entre les deux variables.")
    
    # Ajout d'un encadré déroulant "Ce qu'on peut noter :"
    with st.expander("Ce qu'on peut noter :"):
        st.write(
            """
            
            Toutes les variables explicatives sont statistiquement liées à la variable cible.
            
            Les variables pdays et previous semblent très fortement correllées. Cela s'explique par le fait que la modalité -1 pour l'une et 0 pour l'autre correspondent à un même fait : l'inexistence d'un contact lors d'une campagne précédente. Cela correspond également à la modalité "unknown" pour la variable poutcome.
            
            """
            )
    
    st.write("### B. Quelques axes d'analyse")
    
    # Analyse de distances
    st.write("#### 1. Clustering : affichage des concommitances entre les modalités des variables catégorielles relevant des informations personnelles des clients")
    
    # Sélection des variables catégorielles
    df_cat=df.iloc[:,[1,2,3,4,6,7]]
    
    # On remplace les valeurs manquantes par le mode pour les variables job et education
    df_cat.loc[df_cat.job == "unknown", 'job'] = df_cat.job.mode()[0]
    df_cat.loc[df_cat.education == "unknown", 'education'] = df_cat.education.mode()[0]
    
    ## Encodage One Hot à l'aide de la méthode get_dummies
    M=pd.get_dummies(data=df_cat,drop_first=False)
    ## On donne la valeur "False" au paramètre "drop_first" pour avoir toutes les modalités dans notre arborescence finale
    
    ## Formule de Dice pour mesurer la distance entre 2 variables
    def Dice(col1,col2):
        # Ensure col1 and col2 are numeric arrays
        col1 = np.asarray(col1, dtype=np.float64)
        col2 = np.asarray(col2, dtype=np.float64)
        return (0.5*np.sum((col1 - col2)**2))
    
    ## On transforme notre tableau en array pour la suite
    MN=M.values
    
    ## Création d'un array ayant pour dimension x et y le nombre de colonnes du dataframe
    D=np.zeros(shape=(M.shape[1],M.shape[1]))
    
    ## On remplit cet array en appliquant la fonction préalablement créer pour obtenir une matrice contenant les distance entre une colonne x et une colonne y (c1 et c2 ici)
    for c1 in range(M.shape[1]):
        for c2 in range(M.shape[1]):
            D[c1,c2]=Dice(MN[:,c1],MN[:,c2])
    D=np.sqrt(D)
    
    ## l'array est symétrique en diagonale puisqu'on a calculé la distance Dice entre 2 colonnes 2 fois et avec elle-même. On utilise squareform puis corriger ce problème
    from scipy.spatial.distance import squareform
    VD=squareform(D)
    
    ## on applique une classification ascendante hiérarchique sur les colonnes (qui correspondent à des modalités auxquelles on a appliqué un get.dummies)
    from scipy.cluster.hierarchy import ward
    cah=ward(VD)
    
    from scipy.cluster.hierarchy import dendrogram

    plt.figure(figsize=(12,6))
    plt.title("CAH en fonction des modalités des variables catégorielles")
    dendrogram(Z=cah,labels=M.columns, orientation="left",color_threshold=70)
    st.pyplot(plt)
    
    # Ajout d'un encadré déroulant "Pourquoi ce graphique ?"
    with st.expander("Pourquoi ce graphique ?"):
        st.markdown("Trouver des concommitances entre les variables catégorielles pour établir des profils, des stéréotypes.")
        st.markdown("- Type 1, 'célibataire diplômé' : études supérieures, managers, célibataires et sans emprunts immobiliers")
        st.markdown("- Type 2, 'stable et marié : niveau d’éducation du secondaire, mariés, avec un emprunt immobilier, sans autre emprunt et n’ayant jamais fait défaut")
        st.markdown("- Type 3a, 'situation plus fragile ? a déjà fait défaut, agent de ménage, étudiant, auto-entrepreneur, sans emploi'")
        st.markdown("- Type 3b, 'classe moyenne ? a déjà contracté un crédit autre, ouvriers, techniciens, postes administratifs ou dans les services, retraités, niveaux d’éducation plutôt faibles, personnes ayant pu connaître un divorce")
    
    # Définir les tranches d'âge
    age_bins = [0, 30, 40, 50, 60, float('inf')]
    age_labels = ['0-30', '31-40', '41-50', '51-60', '60+']

    # Ajouter une colonne 'age_group' au DataFrame
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)

    # Filtrer selon les données personnelles
    st.write("#### 2. Répartition de 'deposit' : quel est le client idéal ? ")

    # Sélection des variables pour filtrer (avec st.multiselect pour permettre plusieurs sélections)
    # Création de trois colonnes
    col1, col2, col3 = st.columns(3)

    # Ajout de SelectBox à chaque colonne
    with col1:
        selected_job = st.multiselect("Sélectionnez le job", df['job'].unique(), default=df['job'].unique())
        selected_education = st.multiselect("Sélectionnez l'éducation", df['education'].unique(), default=df['education'].unique())

    with col2:
        selected_housing = st.multiselect("Sélectionnez le housing", df['housing'].unique(), default=df['housing'].unique())
        selected_marital = st.multiselect("Sélectionnez le marital", df['marital'].unique(), default=df['marital'].unique())
        selected_loan = st.multiselect("Sélectionnez le loan", df['loan'].unique(), default=df['loan'].unique())

    with col3:
        selected_poutcome = st.multiselect("Sélectionnez le poutcome", df['poutcome'].unique(), default=df['poutcome'].unique())
        selected_age_group = st.multiselect("Sélectionnez la tranche d'âge", age_labels, default=age_labels)

    # Filtrer les données en fonction des sélections
    filtered_data = df[
    (df['job'].isin(selected_job)) &
    (df['marital'].isin(selected_marital)) &
    (df['education'].isin(selected_education)) &
    (df['housing'].isin(selected_housing)) &
    (df['loan'].isin(selected_loan)) &
    (df['poutcome'].isin(selected_poutcome)) &
    (df['age_group'].isin(selected_age_group))
    ]

    # Créer un diagramme camembert avec Plotly Express
    fig = px.pie(
        filtered_data,
        names='deposit',
        title="Répartition de 'deposit' pour les sélections spécifiées",
        color='deposit',
        color_discrete_map={'yes': 'salmon', 'no': 'skyblue'},
        labels={'yes': 'Oui', 'no': 'Non'},
        hole=0.4  # Contrôle de la taille du trou au centre du camembert
        )

    # Afficher le pourcentage à l'intérieur du camembert
    fig.update_traces(textposition='inside', textinfo='percent+label')

    # Afficher le graphique
    st.plotly_chart(fig)
    
    # Afficher le nombre d'enregistrements sélectionnés
    st.write(f"##### Proportion d'enregistrements sélectionnés : {round((len(filtered_data)/len(df)*100),2)} %")
    
    # Filtrer selon la relation client
    st.write("#### 3. Répartition de 'deposit' : quelle est la relation client idéale ? ")
    
    # Filtrer les enregistrements indésirables
    df = df[(df['duration'] <= 3000) & (df['campaign'] <= 10)]
    
    # Sidebar avec des curseurs pour les intervalles
    duration_range = st.slider('Sélectionner la plage de duration', min(df['duration']), max(df['duration']), (min(df['duration']), max(df['duration'])))
    campaign_range = st.slider('Sélectionner la plage de campaign', min(df['campaign']), max(df['campaign']), (min(df['campaign']), max(df['campaign'])))

    # Filtrer le DataFrame en fonction des intervalles sélectionnés
    filtered_df = df[(df['duration'] >= duration_range[0]) & (df['duration'] <= duration_range[1]) & (df['campaign'] >= campaign_range[0]) & (df['campaign'] <= campaign_range[1])]

    # Créer un graphique pie interactif avec Plotly Express
    fig = px.pie(filtered_df, names='deposit', title='Répartition des dépôts', color='deposit', color_discrete_map={'yes': 'salmon', 'no': 'skyblue'}, labels={'yes': 'Oui', 'no': 'Non'}, hole=0.4)
    st.plotly_chart(fig)
    
    # Afficher le nombre d'enregistrements sélectionnés
    st.write(f"##### Proportion d'enregistrements sélectionnés : {round((len(filtered_df)/len(df)*100),2)} %")