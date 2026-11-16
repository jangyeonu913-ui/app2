import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 페이지 기본 설정
st.set_page_config(page_title="자전거 대여량 예측", layout="centered")
st.title("🚲 날씨에 따른 자전거 대여량 예측 (선형회귀)")

# 1. 깃허브 Raw 데이터 불러오기
bike_url = "https://raw.githubusercontent.com/jangyeonu913-ui/app2/refs/heads/main/bike.csv"
weather_url = "https://raw.githubusercontent.com/jangyeonu913-ui/app2/refs/heads/main/weather.csv"

@st.cache_data
def load_data():
    bike = pd.read_csv(bike_url)
    weather = pd.read_csv(weather_url)
    return bike, weather

try:
    bike_df, weather_df = load_data()
    
    st.subheader("1. 원본 데이터 확인")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**자전거 데이터 (bike.csv)**")
        st.dataframe(bike_df.head(5))
    with col2:
        st.write("**날씨 데이터 (weather.csv)**")
        st.dataframe(weather_df.head(5))

    # 2. 데이터 병합 설정 (사용자가 직접 공통 컬럼 선택 가능하도록 구성)
    st.subheader("2. 데이터 병합 및 모델 설정")
    common_cols = list(set(bike_df.columns).intersection(set(weather_df.columns)))
    
    if not common_cols:
        st.error("두 데이터에 공통으로 존재하는 열(예: 날짜, datetime 등)이 없습니다. 열 이름을 확인해주세요.")
    else:
        merge_key = st.selectbox("두 데이터를 합칠 기준 열(공통 열)을 선택하세요:", common_cols)
        merged_df = pd.merge(bike_df, weather_df, on=merge_key)
        
        # 숫자형 데이터만 추출
        numeric_cols = merged_df.select_dtypes(include=['number']).columns.tolist()
        
        target_col = st.selectbox("예측할 목표(타겟) 열을 선택하세요 (예: 대여량/count):", numeric_cols)
        feature_cols = st.multiselect(
            "예측에 사용할 날씨/조건 열(특성)을 선택하세요:", 
            [c for c in numeric_cols if c != target_col],
            default=[c for c in numeric_cols if c != target_col][:2]
        )

        # 3. 선형회귀 모델 학습 및 결과 출력
        if feature_cols and target_col:
            # 결측치(빈칸) 제거
            clean_df = merged_df[feature_cols + [target_col]].dropna()
            X = clean_df[feature_cols]
            y = clean_df[target_col]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.subheader("3. 선형회귀 모델 학습 결과")
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            st.write(f"- **결정계수 ($R^2$ Score):** `{r2:.4f}` (1에 가까울수록 예측력이 높음)")
            st.write(f"- **평균제곱오차 (MSE):** `{mse:.2f}`")

            # 4. 실시간 예측 시뮬레이터
            st.subheader("4. 직접 값을 입력해 대여량 예측해보기")
            input_data = {}
            for col in feature_cols:
                mean_val = float(X[col].mean())
                input_data[col] = st.number_input(f"{col} 입력", value=mean_val)

            if st.button("예측하기"):
                input_df = pd.DataFrame([input_data])
                prediction = model.predict(input_df)[0]
                st.success(f"예상되는 **{target_col}** 값은 약 **{prediction:.1f}** 입니다!")

except Exception as e:
    st.error(f"데이터를 불러오거나 처리하는 중 오류가 발생했습니다: {e}")
