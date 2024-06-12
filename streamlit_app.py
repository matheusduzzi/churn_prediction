import joblib
import pandas as pd
import streamlit as st
from sklearn.preprocessing import OneHotEncoder

# Função para carregar o modelo
@st.cache_resource
def load_model():
    return joblib.load('model/churn_model.pkl')

# Carregar o modelo
model = load_model()

# Função de pré-processamento
def preprocess_data(df_train):
  categorical = list(df_train.select_dtypes(['object']).columns)
  numerical = list(df_train.select_dtypes(['float64','int64']).columns)

  # desmembrando cada codigo de area para coluna
  enc_one = OneHotEncoder()
  x = df_train["area_code"].values.reshape(-1,1)
  val_train = enc_one.fit_transform(x).toarray()
  df_onehot_train = pd.DataFrame(val_train,columns=['is_'+str(enc_one.categories_[0][i]) for i in range(len(enc_one.categories_[0]))])

  df_c = df_train.copy()

  df_onehot_train = df_onehot_train.reset_index(drop=True)
  df_c = df_c.reset_index(drop=True)
  df_enc_train = pd.concat([df_onehot_train, df_c],axis=1)

  # exluindo a coluna modificada
  df_enc_train.drop("area_code",inplace=True,axis=1)
  frq_dis = df_enc_train.groupby('state').size()/len(df_train)
  df_enc_train["state"] = df_enc_train.state.map(frq_dis)

  numeric_min = df_enc_train[numerical].min().to_dict()
  numeric_max = df_enc_train[numerical].max().to_dict()

  for key in numeric_min.keys():
      df_enc_train[key] = round((df_enc_train[key] - numeric_min[key])/ (numeric_max[key]-numeric_min[key]),3)

  try:
      df_enc_train.drop("id",inplace=True,axis=1)
  except KeyError:
    print("A coluna 'id' não existe no DataFrame.") 

# Função para fazer previsões
def predict_churn(data):
    processed_data = preprocess_data(data)
    prediction = model.predict(processed_data)
    return prediction

# Título da aplicação
st.title('Predição de Churn')

# Upload de arquivo CSV
st.header('Faça o upload do arquivo CSV com os dados do cliente:')
uploaded_file = st.file_uploader("Escolha o arquivo CSV", type="csv")

if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)

    #predictions = predict_churn(input_data)
    #input_data['Churn Prediction'] = predictions
    st.write('Previsões de Churn:')
    st.write(input_data)

if __name__ == '__main__':
    st.set_option('deprecation.showfileUploaderEncoding', False)
