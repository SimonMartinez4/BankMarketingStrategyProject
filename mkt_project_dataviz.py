# -*- coding: utf-8 -*-
"""

@author: Simon Martinez
"""

import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import plotly.graph_objects as go

st.title("Prédictions d'une campagne marketing d'une banque")
df = pd.read_csv("bank.csv")

st.sidebar.title("Sommaire")
pages = ["Visualisation des données"]
page = st.sidebar.radio("Aller à la page", pages)

if page == pages[0] : 
    st.write("## Visualisation des données")
    
    st.write("#### Analyse de la variable cible")
    
    # Création du graphique de la variable cible avec Plotly et personnalisation
    pie_y = px.pie(df, names="deposit", title="Distribution de la variable cible deposit")

    # Personnalisation des couleurs
    pie_y.update_traces(textinfo='label+percent', pull=[0.2, 0], textposition='inside')

    # Affichage du graphique
    st.plotly_chart(pie_y)

    st.write("#### Analyse des variables explicatives")
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
    
    # Matrice de correlation
    df_num=df.select_dtypes(include=['number'])
    correlation_matrix = df_num.corr()

    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale=['skyblue', 'salmon']
        ))
    fig.update_layout(
        title='Matrice de corrélation',
        title_x=0.32,  # Centre le titre horizontalement
        title_y=0.9,  # Ajuste la position verticale du titre
        title_font=dict(size=24, color='salmon')  # Taille et couleur du titre
        )

    st.plotly_chart(fig)