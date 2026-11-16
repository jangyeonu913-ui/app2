import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. 웹페이지 기본 설정
st.set_page_config(page_title="날씨별 자전거 대여량 예측", layout="wide")
st.title("🚲 날씨 데이터 기반 자전거 대여량 예측 (선형회귀)")
st.write("기온, 습도, 풍속 등 날씨 데이터를 바탕으로 자전거 대여량을 예측하는 AI 모델입니다.")

# 2. 데이터 불러오기
@st.cache_data
def load_data():
    bike_url = "https://raw.githubusercontent.com/jangyeonu913-ui/app2/refs/heads/main/bike.csv"
    weather_url = "https://raw.githubusercontent.com/jangyeonu913-ui/app2/refs/heads/main/weather.csv"
    
    bike = pd.read_csv(bike_url)
    weather = pd.read_csv(weather_url)
    return bike, weather

try:
    bike_df, weather_df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# 3. 사이드바에서 병합 기준 열 선택 (수동 매칭 기능)
st.sidebar.header("⚙️ 데이터 병합 설정")
st.sidebar.info("두 파일에서 서로 공통으로 들어있는 열(날짜, ID 등)을 각각 선택해주세요.")

bike_key = st.sidebar.selectbox("1️⃣ bike.csv 기준 열 선택:", bike_df.columns.tolist())
weather_key = st.sidebar.selectbox("2️⃣ weather.csv 기준 열 선택:", weather_df.columns.tolist())

# 데이터 병합 수행
bike_copy = bike_df.copy()
weather_copy = weather_df.copy()

bike_copy['join_key'] = bike_copy[bike_key].astype(str).str.strip()
weather_copy['join_key'] = weather_copy[weather_key].astype(str).str.strip()

df = pd.merge(bike_copy, weather_copy, on='join_key', suffixes=('', '_weather'))

# 병합된 데이터가 0개일 때 원본을 보여주어 진단할 수 있도록 함
if len(df) == 0:
    st.error(f"⚠️ 선택한 기준 열(`{bike_key}`와 `{weather_key}`)로 일치하는 데이터를 찾지 못했습니다 (현재 0개). 아래 원본 데이터를 참고하여 올바른 기준 열을 사이드바에서 다시 선택해주세요.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📁 bike.csv 원본 미리보기")
        st.dataframe(bike_df.head())
    with col2:
        st.subheader("📁 weather.csv 원본 미리보기")
        st.dataframe(weather_df.head())
    st.stop()

# 4. 데이터 미리보기
with st.expander("📊 병합된 원본 데이터 미리보기 (성공!)"):
    st.write(f"총 매칭된 데이터 행 개수: {len(df)}개")
    st.dataframe(df.head(10))

# 5. 모델 학습을 위한 변수 설정 (숫자형 데이터만 자동 추출)
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

if len(numeric_cols) < 2:
    st.warning("학습에 사용할 수 있는 숫자형 데이터가 충분하지 않습니다.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.header("⚙️ AI 모델 설정")
default_target = next((col for col in numeric_cols if 'count' in col.lower() or 'cnt' in col.lower() or '대여' in col), numeric_cols[-1])
target_col = st.sidebar.selectbox("예측할 목표 값 (Y):", numeric_cols, index=numeric_cols.index(default_target))

feature_options = [col for col in numeric_cols if col != target_col and col != 'join_key']
selected_features = st.sidebar.multiselect("학습에 사용할 날씨 조건 (X):", feature_options, default=feature_options[:min(5, len(feature_options))])

if not selected_features:
    st.warning("왼쪽 사이드바에서 최소 1개 이상의 날씨 조건(X)을 선택해주세요.")
    st.stop()

# 결측치 제거 및 데이터 분리
clean_df = df[selected_features + [target_col]].dropna()

if len(clean_df) < 5:
    st.error(f"⚠️ 유효한 행이 부족합니다 (현재 {len(clean_df)}개).")
    st.stop()

X = clean_df[selected_features]
y = clean_df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. 선형회귀 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 7. 결과 대시보드 출력
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 모델 성능 평가")
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    st.metric(label="결정계수 (R² Score)", value=f"{r2:.3f}")
    st.metric(label="평균제곱오차 (MSE)", value=f"{mse:,.1f}")
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_test, y_pred, alpha=0.5, color='royalblue')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)

with col2:
    st.subheader("🔮 직접 날씨 입력해서 예측해보기")
    input_data = {}
    for feat in selected_features:
        min_val = float(X[feat].min())
        max_val = float(X[feat].max())
        mean_val = float(X[feat].mean())
        if min_val == max_val:
            max_val += 1.0
        input_data[feat] = st.slider(f"{feat}", min_val, max_val, mean_val)
    
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    final_val = max(0, prediction)
    
    st.success(f"예상 {target_col}: {final_val:,.0f}")
