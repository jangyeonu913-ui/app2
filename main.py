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

# 2. 깃허브에서 데이터 불러오기 (데이터 타입 오류 방지 기능 포함)
@st.cache_data
def load_data():
    bike_url = "https://raw.githubusercontent.com/jangyeonu913-ui/app2/refs/heads/main/bike.csv"
    weather_url = "https://raw.githubusercontent.com/jangyeonu913-ui/app2/refs/heads/main/weather.csv"
    
    bike = pd.read_csv(bike_url)
    weather = pd.read_csv(weather_url)
    
    # 공통 컬럼 찾기
    common_cols = list(set(bike.columns) & set(weather.columns))
    
    if common_cols:
        key_col = common_cols[0]
        # 데이터 타입을 둘 다 문자열(str)로 맞춰서 타입 불일치 오류 방지
        bike[key_col] = bike[key_col].astype(str)
        weather[key_col] = weather[key_col].astype(str)
        df = pd.merge(bike, weather, on=key_col)
    else:
        left_key = bike.columns[0]
        right_key = weather.columns[0]
        bike[left_key] = bike[left_key].astype(str)
        weather[right_key] = weather[right_key].astype(str)
        df = pd.merge(bike, weather, left_on=left_key, right_on=right_key)
        
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# 3. 데이터 미리보기
with st.expander("📊 병합된 원본 데이터 미리보기"):
    st.dataframe(df.head(10))

# 4. 모델 학습을 위한 변수 설정 (숫자형 데이터만 자동 추출)
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

if len(numeric_cols) < 2:
    st.warning("학습에 사용할 수 있는 숫자형 데이터가 충분하지 않습니다.")
    st.stop()

st.sidebar.header("⚙️ 모델 설정")
# 예측할 목표 변수(Y) 선택
default_target = next((col for col in numeric_cols if 'count' in col.lower() or 'cnt' in col.lower() or '대여' in col), numeric_cols[-1])
target_col = st.sidebar.selectbox("예측할 목표 값 (Y):", numeric_cols, index=numeric_cols.index(default_target))

# 입력 변수(X) 선택
feature_options = [col for col in numeric_cols if col != target_col]
selected_features = st.sidebar.multiselect("학습에 사용할 날씨 조건 (X):", feature_options, default=feature_options)

if not selected_features:
    st.warning("왼쪽 사이드바에서 최소 1개 이상의 날씨 조건(X)을 선택해주세요.")
    st.stop()

# 결측치 제거 및 데이터 분리 (데이터 부족 에러 방지 안전장치 추가)
clean_df = df[selected_features + [target_col]].dropna()

if len(clean_df) < 10:
    st.error(f"⚠️ 병합된 데이터에 유효한 행이 부족합니다 (현재 {len(clean_df)}개). 두 CSV 파일의 공통 열(날짜 등) 데이터가 올바르게 매칭되는지 확인해주세요.")
    with st.expander("🔍 디버깅용 병합 데이터 확인"):
        st.write("bike.csv 및 weather.csv 병합 결과 크기:", df.shape)
        st.dataframe(df.head())
    st.stop()

X = clean_df[selected_features]
y = clean_df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 선형회귀 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 6. 결과 대시보드 출력
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 모델 성능 평가")
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    st.metric(label="결정계수 (R² Score)", value=f"{r2:.3f}")
    st.metric(label="평균제곱오차 (MSE)", value=f"{mse:,.1f}")
    
    # 실제값 vs 예측값 그래프
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
        input_data[feat] = st.slider(f"{feat}", min_val, max_val, mean_val)
    
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    
    st.success(f"### 예상 {target_col}: **{max(0
