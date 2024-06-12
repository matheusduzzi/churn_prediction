import joblib
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Função para carregar o modelo
@st.cache_resource
def load_model():
    return joblib.load('model/churn_model.pkl')

# Carregar o modelo
model = load_model()

# Função de pré-processamento
def preprocess_data(data, freq_encoding, one_hot_encoder, label_encoders):
    # Label Encoding para 'international_plan' e 'voice_mail_plan'
    for col in ['international_plan', 'voice_mail_plan']:
        data[col] = label_encoders[col].transform(data[col])

    # Frequency Encoding para 'state'
    data['state'] = data['state'].map(freq_encoding)

    # One-hot encoding para 'area_code'
    area_code_encoded = one_hot_encoder.transform(data[['area_code']]).toarray()
    area_code_columns = ['is_' + str(cat) for cat in one_hot_encoder.categories_[0]]
    area_code_df = pd.DataFrame(area_code_encoded, columns=area_code_columns)

    data = data.drop(columns=['area_code']).join(area_code_df)

    # Adicionar colunas faltantes (caso necessário)
    required_columns = [
        'account_length', 'number_vmail_messages', 'total_day_minutes', 'total_day_calls',
        'total_day_charge', 'total_eve_minutes', 'total_eve_calls', 'total_eve_charge',
        'total_night_minutes', 'total_night_calls', 'total_night_charge', 'total_intl_minutes',
        'total_intl_calls', 'total_intl_charge', 'number_customer_service_calls', 'state',
        'international_plan', 'voice_mail_plan'
    ] + area_code_columns

    for col in required_columns:
        if col not in data.columns:
            data[col] = 0

    # Garantir a ordem correta das colunas
    data = data[required_columns]

    return data

# Função para fazer previsões
def predict_churn(data, freq_encoding, one_hot_encoder, label_encoders):
    processed_data = preprocess_data(data, freq_encoding, one_hot_encoder, label_encoders)
    prediction = model.predict(processed_data)
    return prediction

# Carregar os dados de treino para calcular as frequências
train_data = pd.read_csv('train.csv')
state_freq = train_data['state'].value_counts(normalize=True).to_dict()

# Treinar o OneHotEncoder e LabelEncoders
one_hot_encoder = OneHotEncoder()
one_hot_encoder.fit(train_data[['area_code']])

label_encoders = {
    'international_plan': LabelEncoder(),
    'voice_mail_plan': LabelEncoder()
}
for col in label_encoders.keys():
    label_encoders[col].fit(train_data[col])

# Título da aplicação
st.title('Predição de Churn')

# Upload de arquivo CSV
st.header('Faça o upload do arquivo CSV com os dados do cliente:')
uploaded_file = st.file_uploader("Escolha o arquivo CSV", type="csv")

if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)

    # Ordenar as colunas de acordo com o esperado pelo modelo
    required_columns = [
        'state', 'account_length', 'area_code', 'international_plan', 'voice_mail_plan',
        'number_vmail_messages', 'total_day_minutes', 'total_day_calls',
        'total_day_charge', 'total_eve_minutes', 'total_eve_calls',
        'total_eve_charge', 'total_night_minutes', 'total_night_calls',
        'total_night_charge', 'total_intl_minutes', 'total_intl_calls',
        'total_intl_charge', 'number_customer_service_calls'
    ]

    input_data = input_data[required_columns]

    predictions = predict_churn(input_data, state_freq, one_hot_encoder, label_encoders)
    input_data['Churn Prediction'] = predictions
    st.write('Previsões de Churn:')
    st.write(input_data)

if __name__ == '__main__':
    st.set_option('deprecation.showfileUploaderEncoding', False)
