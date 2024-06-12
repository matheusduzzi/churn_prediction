#! pip install streamlit
!pip install joblib

import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Carregar o modelo
model = joblib.load('/content/churn_model.pkl')

# Função de pré-processamento
def preprocess_data(data, freq_encoding, one_hot_encoder):
    # Criar um dataframe com os dados de entrada
    df = pd.DataFrame([data], columns=['state', 'account_length', 'area_code', 'international_plan', 'voice_mail_plan',
                                       'number_vmail_messages', 'total_day_minutes', 'total_day_calls',
                                       'total_day_charge', 'total_eve_minutes', 'total_eve_calls',
                                       'total_eve_charge', 'total_night_minutes', 'total_night_calls',
                                       'total_night_charge', 'total_intl_minutes', 'total_intl_calls',
                                       'total_intl_charge', 'number_customer_service_calls'])

    # Label Encoding para 'international_plan' e 'voice_mail_plan'
    label_encoders = {
        'international_plan': LabelEncoder(),
        'voice_mail_plan': LabelEncoder()
    }

    df['international_plan'] = label_encoders['international_plan'].fit_transform(df['international_plan'])
    df['voice_mail_plan'] = label_encoders['voice_mail_plan'].fit_transform(df['voice_mail_plan'])

    # Frequency Encoding para 'state'
    df['state'] = df['state'].map(freq_encoding)

    # One-hot encoding para 'area_code'
    area_code_encoded = one_hot_encoder.transform(df[['area_code']]).toarray()
    area_code_columns = ['is_' + str(cat) for cat in one_hot_encoder.categories_[0]]
    area_code_df = pd.DataFrame(area_code_encoded, columns=area_code_columns)

    df = df.drop(columns=['area_code']).join(area_code_df)

    # Adicionar colunas faltantes (caso necessário)
    required_columns = [
        'account_length', 'number_vmail_messages', 'total_day_minutes', 'total_day_calls',
        'total_day_charge', 'total_eve_minutes', 'total_eve_calls', 'total_eve_charge',
        'total_night_minutes', 'total_night_calls', 'total_night_charge', 'total_intl_minutes',
        'total_intl_calls', 'total_intl_charge', 'number_customer_service_calls', 'state',
        'international_plan', 'voice_mail_plan'
    ] + area_code_columns

    for col in required_columns:
        if col not in df.columns:
            df[col] = 0

    # Garantir a ordem correta das colunas
    df = df[required_columns]

    return df.values[0]

# Função para fazer previsões
def predict_churn(data, freq_encoding, one_hot_encoder):
    processed_data = preprocess_data(data, freq_encoding, one_hot_encoder)
    prediction = model.predict([processed_data])
    return prediction[0]

# Carregar os dados de treino para calcular as frequências
train_data = pd.read_csv('/content/train.csv')
state_freq = train_data['state'].value_counts(normalize=True).to_dict()

# Treinar o OneHotEncoder
one_hot_encoder = OneHotEncoder()
one_hot_encoder.fit(train_data[['area_code']])

# Título da aplicação
st.title('Predição de Churn')

# Formulário para entrada de dados
st.header('Insira os dados do cliente:')
state = st.selectbox('Estado', ['OH', 'NJ', 'OK', 'MA'])
account_length = st.number_input('Comprimento da Conta', min_value=0, max_value=500, value=100)
area_code = st.selectbox('Código de Área', ['area_code_415', 'area_code_408', 'area_code_510'])
international_plan = st.selectbox('Plano Internacional', ['yes', 'no'])
voice_mail_plan = st.selectbox('Plano de Correio de Voz', ['yes', 'no'])
number_vmail_messages = st.number_input('Número de Mensagens de Voz', min_value=0, max_value=50, value=0)
total_day_minutes = st.number_input('Minutos de Dia Totais', min_value=0.0, max_value=500.0, value=0.0)
total_day_calls = st.number_input('Chamadas de Dia Totais', min_value=0, max_value=500, value=0)
total_day_charge = st.number_input('Custo do Dia Total', min_value=0.0, max_value=100.0, value=0.0)
total_eve_minutes = st.number_input('Minutos de Tarde Totais', min_value=0.0, max_value=500.0, value=0.0)
total_eve_calls = st.number_input('Chamadas de Tarde Totais', min_value=0, max_value=500, value=0)
total_eve_charge = st.number_input('Custo da Tarde Total', min_value=0.0, max_value=100.0, value=0.0)
total_night_minutes = st.number_input('Minutos de Noite Totais', min_value=0.0, max_value=500.0, value=0.0)
total_night_calls = st.number_input('Chamadas de Noite Totais', min_value=0, max_value=500, value=0)
total_night_charge = st.number_input('Custo da Noite Total', min_value=0.0, max_value=100.0, value=0.0)
total_intl_minutes = st.number_input('Minutos Internacionais Totais', min_value=0.0, max_value=50.0, value=0.0)
total_intl_calls = st.number_input('Chamadas Internacionais Totais', min_value=0, max_value=20, value=0)
total_intl_charge = st.number_input('Custo Internacional Total', min_value=0.0, max_value=10.0, value=0.0)
number_customer_service_calls = st.number_input('Número de Chamadas para o Serviço de Atendimento ao Cliente', min_value=0, max_value=10, value=0)

# Botão para fazer a previsão
if st.button('Prever Churn'):
    data = [state, account_length, area_code, international_plan, voice_mail_plan, number_vmail_messages,
            total_day_minutes, total_day_calls, total_day_charge, total_eve_minutes, total_eve_calls, total_eve_charge,
            total_night_minutes, total_night_calls, total_night_charge, total_intl_minutes, total_intl_calls,
            total_intl_charge, number_customer_service_calls]
    churn = predict_churn(data, state_freq, one_hot_encoder)
    st.write(f'O cliente irá churn: {churn}')

# Rodar a aplicação Streamlit
if __name__ == '__main__':
    st.set_option('deprecation.showfileUploaderEncoding', False)
