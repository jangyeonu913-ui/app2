import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 페이지 기본 설정
st.set_page_config(page_title="자전거 대여량 예측 및 시각화", layout="centered")
st.title("🚲 날씨에 따른 자전거 대여량 예측 및 시각화")

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
        st.dataframe(bike_df.head(3))
    with col2:
        st.write("**날씨 데이터 (weather.csv)**")
        st.dataframe(weather_df.head(3))

    # 2. 데이터 병합 설정
    st.subheader("2. 데이터 병합 및 시각화")
    
    col_a, col_b = st.columns(2)
    with col_a:
        bike_key = st.selectbox("자전거 데이터의 기준 열 선택:", bike_df.columns, index=0)
    with col_b:
        default_weather_idx = list(weather_df.columns).index("일시") if "일시" in weather_df.columns else 0
        weather_key = st.selectbox("날씨 데이터의 기준 열 선택:", weather_df.columns, index=default_weather_idx)

    # 두 데이터 병합
    merged_df = pd.merge(bike_df, weather_df, left_on=bike_key, right_on=weather_key)
    
    st.write(f"병합된 데이터 미리보기 (총 데이터 수: {len(merged_df)}행)")
    st.dataframe(merged_df.head(3))

    # [그래프 추가 1] 날짜별 대여건수 변화 막대/선 그래프
    if '대여건수' in merged_df.columns and bike_key in merged_df.columns:
        st.markdown("### 📊 날짜별 자전거 대여건수 추이")
        chart_data = merged_df.set_index(bike_key)[['대여건수']]
        st.bar_chart(chart_data)  # 막대 그래프로 표현

    # 숫자형 데이터만 추출
    numeric_cols = merged_df.select_dtypes(include=['number']).columns.tolist()
    
    if not numeric_cols:
        st.error("분석할 수 있는 숫자형 데이터가 부족합니다.")
    else:
        target_col = st.selectbox("예측할 목표(타겟) 열을 선택하세요:", numeric_cols, index=numeric_cols.index('대여건수') if '대여건수' in numeric_cols else 0)
        
        feature_cols = st.multiselect(
            "예측에 사용할 날씨/조건 열(특성)을 선택하세요:", 
            [c for c in numeric_cols if c != target_col],
            default=[c for c in numeric_cols if c != target_col][:1]
        )

        # [그래프 추가 2] 선택한 특성과 대여량 간의 관계 산점도
        if feature_cols:
            st.markdown("### 📈 선택한 날씨 요인과 대여건수의 관계")
            for f_col in feature_cols:
                st.write(f"**{f_col} vs {target_col}**")
                st.scatter_chart(merged_df[[f_col, target_col]], x=f_col, y=target_col)

        # 3. 선형회귀 모델 학습 및 결과 출력
        if feature_cols and target_col:
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

            # [그래프 추가 3] 실제 값 vs 예측 값 비교 그래프
            st.markdown("### 📉 테스트 데이터의 실제 값 vs 예측 값 비교")
            result_df = pd.DataFrame({'실제값': y_test.values, '예측값': y_pred}).reset_index(drop=True)
            st.line_chart(result_df)

            # 4. 실시간 예측 시뮬레이터
            st.subheader("4. 직접 값을 입력해 대여량 예측해보기")
            input_data = {}
            for col in feature_cols:
                mean_val = float(X[col].mean())
                input_data[col] = st.number_input(f"{col} 입력", value=mean_val)

            if st.button("예측하기"):
                input_df = pd.DataFrame([input_data])
                prediction = model.predict(input_df)[0]
                st.success(f"예상되는 **{target_col}** 값은 약 **{prediction:.1f}** 건 입니다!")

except Exception as e:
    st.error(f"데이터를 불러오거나 처리하는 중 오류가 발생했습니다: {e}")
