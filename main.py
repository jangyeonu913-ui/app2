import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 페이지 기본 설정
st.set_page_config(page_title="공공자전거 이용 분석 앱", layout="centered")
st.title("🚲 언제 공공자전거를 가장 많이 탈까? (날씨 분석 앱)")
st.write("날씨와 날짜 데이터를 분석하여 공공자전거 대여 수요를 파악하고 예측하는 앱입니다.")

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
    
    # 2. 데이터 병합 설정 (기준 열 선택)
    st.subheader("1. 데이터 연결 설정")
    col_a, col_b = st.columns(2)
    with col_a:
        bike_key = st.selectbox("자전거 데이터 기준 열:", bike_df.columns, index=0)
    with col_b:
        default_weather_idx = list(weather_df.columns).index("일시") if "일시" in weather_df.columns else 0
        weather_key = st.selectbox("날씨 데이터 기준 열:", weather_df.columns, index=default_weather_idx)

    # 두 데이터 병합
    merged_df = pd.merge(bike_df, weather_df, left_on=bike_key, right_on=weather_key)
    st.success(f"데이터 병합 완료! (총 {len(merged_df)}일간의 데이터)")

    # [핵심 분석 1] 대여 건수가 가장 많은 날 TOP 5 분석
    st.subheader("🏆 공공자전거 이용이 가장 활발했던 날 TOP 5")
    if '대여건수' in merged_df.columns and bike_key in merged_df.columns:
        top_bikes = merged_df.sort_values(by='대west' if 'west' in merged_df.columns else '대여건수', ascending=False).head(5) if '대여건수' not in merged_df.columns else merged_df.sort_values(by='대여건수', ascending=False).head(5)
        
        # 보기 깔끔하게 출력할 컬럼 선택
        display_cols = [c for c in [bike_key, '대여건수', '평균기온(℃)'] if c in merged_df.columns]
        st.dataframe(top_bikes[display_cols].reset_index(drop=True))

    # [핵심 분석 2] 날짜별 대여건수 추이 그래프
    st.subheader("📊 날짜별 자전거 대여건수 변화 추이")
    if '대여건수' in merged_df.columns:
        chart_data = merged_df.set_index(bike_key)[['대여건수']]
        st.line_chart(chart_data)

    # [핵심 분석 3] 선형회귀를 통한 날씨-대여량 관계 분석
    st.subheader("🤖 날씨 조건에 따른 대여량 예측 (선형회귀 모델)")
    numeric_cols = merged_df.select_dtypes(include=['number']).columns.tolist()
    
    if '대여건수' in numeric_cols:
        target_col = '대여건수'
        feature_cols = st.multiselect(
            "분석 및 예측에 사용할 날씨 요인 선택:", 
            [c for c in numeric_cols if c != target_col],
            default=['평균기온(℃)'] if '평균기온(℃)' in numeric_cols else [c for c in numeric_cols if c != target_col][:1]
        )

        if feature_cols:
            clean_df = merged_df[feature_cols + [target_col]].dropna()
            X = clean_df[feature_cols]
            y = clean_df[target_col]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            st.info(f"💡 **모델 분석 결과**: 선택한 날씨 요인이 자전거 대여량을 약 **{r2*100:.1f}%** 설명할 수 있습니다 ($R^2$ 점수: {r2:.4f})")

            # 산점도 그래프
            for f_col in feature_cols:
                st.write(f"📈 **[{f_col}]과 [대여건수]의 상관관계 산점도**")
                st.scatter_chart(merged_df[[f_col, target_col]], x=f_col, y=target_col)

            # [핵심 분석 4] 실시간 예측 시뮬레이터
            st.subheader("🔮 미래 날씨 입력해서 자전거 대여량 예측해보기")
            input_data = {}
            for col in feature_cols:
                mean_val = float(X[col].mean())
                input_data[col] = st.number_input(f"예측할 {col} 값 입력", value=mean_val)

            if st.button("대여량 예측하기"):
                input_df = pd.DataFrame([input_data])
                prediction = model.predict(input_df)[0]
                st.success(f"✨ 해당 날씨 조건에서의 예상 자전거 대여 건수는 약 **{max(0, prediction):.0f}건** 입니다!")

except Exception as e:
    st.error(f"데이터를 처리하는 동안 오류가 발생했습니다: {e}")
