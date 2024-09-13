import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from tabulate import tabulate
import tkinter as tk
from tkinter import ttk, messagebox
import random

# Criar um DataFrame fictício com dados variados
def generate_data():
    data = {
        'local_saida': ['São Paulo', 'Rio de Janeiro', 'Belo Horizonte', 'São Paulo', 'Brasília', 'São Paulo', 'Rio de Janeiro'],
        'local_chegada': ['Rio de Janeiro', 'São Paulo', 'São Paulo', 'Belo Horizonte', 'São Paulo', 'Brasília', 'Belo Horizonte'],
        'horario_saida': ['08:00', '09:30', '11:00', '13:00', '15:30', '17:00', '19:00'],
        'numero_voo': ['SP123', 'RJ456', 'BH789', 'SP234', 'BS345', 'SP567', 'RJ678'],
        'empresa': ['LATAM', 'Gol', 'Azul', 'LATAM', 'Gol', 'Azul', 'LATAM'],
        'cidade_destino': ['Rio de Janeiro', 'São Paulo', 'São Paulo', 'Belo Horizonte', 'São Paulo', 'Brasília', 'Belo Horizonte'],
        'tempo_duracao': [1.5, 1.0, 1.5, 1.0, 1.5, 2.0, 1.5],
        'atraso': [0, 1, 0, 0, 1, 0, 1]  # 0: Não atrasado, 1: Atrasado
    }
    df = pd.DataFrame(data)
    return df

# Preprocessar dados
def preprocess_data(df):
    le_local = LabelEncoder()
    le_empresa = LabelEncoder()
    le_cidade = LabelEncoder()

    df['local_saida'] = le_local.fit_transform(df['local_saida'])
    df['local_chegada'] = le_local.transform(df['local_chegada'])
    df['empresa'] = le_empresa.fit_transform(df['empresa'])
    df['cidade_destino'] = le_cidade.fit_transform(df['cidade_destino'])

    df['horario_saida'] = pd.to_datetime(df['horario_saida'], format='%H:%M').dt.hour

    X = df[['local_saida', 'local_chegada', 'horario_saida', 'empresa', 'cidade_destino', 'tempo_duracao']]
    y = df['atraso']

    return X, y, le_local, le_empresa, le_cidade

# Criar e treinar o modelo
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model, X_test, y_test

# Avaliar o modelo
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    return accuracy, report

# Gerar novos voos
def generate_new_flights(le_local, le_empresa, le_cidade, model):
    locais = ['São Paulo', 'Rio de Janeiro', 'Belo Horizonte', 'Brasília']
    empresas = ['LATAM', 'Gol', 'Azul']
    cidades = ['Rio de Janeiro', 'São Paulo', 'Belo Horizonte', 'Brasília']

    new_flights = []
    for _ in range(5):  # Gerar 5 voos aleatórios
        local_saida = random.choice(locais)
        local_chegada = random.choice(locais)
        empresa = random.choice(empresas)
        cidade_destino = random.choice(cidades)
        horario_saida = pd.to_datetime(f'{random.randint(0, 23)}:00', format='%H:%M').hour
        tempo_duracao = round(random.uniform(1.0, 3.0), 1)

        novo_voo = {
            'local_saida': le_local.transform([local_saida])[0],
            'local_chegada': le_local.transform([local_chegada])[0],
            'horario_saida': horario_saida,
            'empresa': le_empresa.transform([empresa])[0],
            'cidade_destino': le_cidade.transform([cidade_destino])[0],
            'tempo_duracao': tempo_duracao
        }

        new_flights.append(novo_voo)

    df_novos_voos = pd.DataFrame(new_flights)
    df_novos_voos['atraso_previsto'] = model.predict(df_novos_voos)

    df_novos_voos['local_saida'] = le_local.inverse_transform(df_novos_voos['local_saida'])
    df_novos_voos['local_chegada'] = le_local.inverse_transform(df_novos_voos['local_chegada'])
    df_novos_voos['empresa'] = le_empresa.inverse_transform(df_novos_voos['empresa'])
    df_novos_voos['cidade_destino'] = le_cidade.inverse_transform(df_novos_voos['cidade_destino'])

    return df_novos_voos

# Interface gráfica com tkinter
class FlightApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Visualizador de Voos")
        self.root.geometry("800x600")

        self.df = generate_data()
        self.X, self.y, self.le_local, self.le_empresa, self.le_cidade = preprocess_data(self.df)
        self.model, self.X_test, self.y_test = train_model(self.X, self.y)

        self.create_widgets()

    def create_widgets(self):
        self.table = ttk.Treeview(self.root, columns=('local_saida', 'local_chegada', 'horario_saida', 'empresa', 'cidade_destino', 'tempo_duracao', 'atraso_previsto'), show='headings')
        self.table.heading('local_saida', text='Local de Saída')
        self.table.heading('local_chegada', text='Local de Chegada')
        self.table.heading('horario_saida', text='Horário de Saída')
        self.table.heading('empresa', text='Empresa')
        self.table.heading('cidade_destino', text='Cidade Destino')
        self.table.heading('tempo_duracao', text='Tempo de Duração')
        self.table.heading('atraso_previsto', text='Atraso Previsto')

        self.table.pack(fill=tk.BOTH, expand=True)

        self.refresh_button = tk.Button(self.root, text="Gerar Novos Voos", command=self.generate_flights)
        self.refresh_button.pack(pady=10)

        self.generate_flights()

    def generate_flights(self):
        new_flights_df = generate_new_flights(self.le_local, self.le_empresa, self.le_cidade, self.model)

        for i in self.table.get_children():
            self.table.delete(i)

        for _, row in new_flights_df.iterrows():
            self.table.insert('', 'end', values=(row['local_saida'], row['local_chegada'], row['horario_saida'], row['empresa'], row['cidade_destino'], row['tempo_duracao'], row['atraso_previsto']))

if __name__ == "__main__":
    root = tk.Tk()
    app = FlightApp(root)
    root.mainloop()
