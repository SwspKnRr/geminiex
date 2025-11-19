import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from datetime import date, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------
# 1. 공통 함수 및 설정
# ---------------------------------------------------------
st.set_page_config(page_title="AI 주식 트레이딩 봇", layout="wide")
st.title("🤖 AI 주식 트레이딩 솔루션")
st.markdown("---")

# 사이드바 공통 입력
st.sidebar.header("기본 설정")
ticker = st.sidebar.text_input("종목 코드 (예: AAPL, TSLA, 005930.KS)", "AAPL")

# ---------------------------------------------------------
# 2. 데이터 로드 함수들
# ---------------------------------------------------------
@st.cache_data
def load_5m_data(ticker):
    # yfinance 제약: 5분봉은 최대 60일치만 가져올 수 있음
    data = yf.download(ticker, interval="5m", period="60d")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.reset_index(inplace=True)
    
    # Timezone 제거 (Prophet 오류 방지)
    if 'Datetime' in data.columns:
        data['Datetime'] = pd.to_datetime(data['Datetime']).dt.tz_localize(None)
    return data

@st.cache_data
def load_daily_data_with_indicators(ticker):
    # 1. 주가 데이터 (최근 5년)
    end_date = date.today().strftime("%Y-%m-%d")
    start_date = (date.today() - timedelta(days=365*5)).strftime("%Y-%m-%d")
    
    df = yf.download(ticker, start=start_date, end=end_date)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # 2. 거시 지표: 미국 10년물 국채 금리 (^TNX), VIX 지수 (^VIX)
    macro_tickers = ['^TNX', '^VIX']
    macro_data = yf.download(macro_tickers, start=start_date, end=end_date)['Close']
    if isinstance(macro_data.columns, pd.MultiIndex):
        macro_data.columns = macro_data.columns.get_level_values(0)
    
    # 데이터 병합 (날짜 기준)
    df = df.join(macro_data)
    df.rename(columns={'^TNX': 'Interest_Rate', '^VIX': 'VIX'}, inplace=True)
    
    # 3. 보조 지표 계산 (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 4. 이동평균선 괴리율
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['Disparity'] = (df['Close'] / df['MA20']) * 100
    
    # 결측치 제거
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    return df

# ---------------------------------------------------------
# 3. 탭 구성
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["⏱️ 5분봉 단기 예측 (Real-time)", "📈 상승 확률 분석 (AI Classifier)"])

# =========================================================
# TAB 1: 5분봉 단기 예측
# =========================================================
with tab1:
    st.subheader(f"⏱️ {ticker} 5분봉 기반 단기 흐름 예측")
    st.info("💡 참고: 무료 API 제한으로 최근 60일 간의 5분봉 데이터만 사용합니다.")

    if st.button("단기 예측 실행 (Tab 1)"):
        with st.spinner('5분봉 데이터 다운로드 및 학습 중...'):
            try:
                df_5m = load_5m_data(ticker)
                
                if df_5m.empty:
                    st.error("데이터를 가져올 수 없습니다.")
                else:
                    # Prophet 데이터셋 준비
                    df_prophet = df_5m[['Datetime', 'Close']].rename(columns={'Datetime': 'ds', 'Close': 'y'})
                    
                    # 모델 학습 (Intraday 설정)
                    m = Prophet(changepoint_prior_scale=0.05, daily_seasonality=True) 
                    m.fit(df_prophet)
                    
                    # 향후 12개 구간(60분) 예측
                    future = m.make_future_dataframe(periods=12, freq='5min')
                    forecast = m.predict(future)
                    
                    # 시각화
                    fig = plot_plotly(m, forecast)
                    fig.update_layout(title=f"{ticker} 향후 60분 주가 흐름 예측")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 최근 데이터 표시
                    st.write("최근 5분봉 데이터:")
                    st.dataframe(df_5m.tail())
                    
            except Exception as e:
                st.error(f"에러 발생: {e}")

# =========================================================
# TAB 2: 상승 확률 분석 (분류 모델)
# =========================================================
with tab2:
    st.subheader(f"📈 {ticker} 내일 상승 확률 분석 (머신러닝)")
    st.markdown("""
    **학습 원리:**
    과거 5년치 데이터를 바탕으로 **RSI, 금리, 공포지수(VIX), 이동평균 괴리율**을 학습합니다.
    이와 비슷한 패턴일 때 주가가 **'다음 날 올랐는지/내렸는지'**를 AI가 판단합니다.
    """)
    
    if st.button("상승 확률 계산 (Tab 2)"):
        with st.spinner('거시 지표 수집 및 AI 모델 학습 중...'):
            try:
                df_daily = load_daily_data_with_indicators(ticker)
                
                if df_daily.empty:
                    st.error("데이터 부족으로 분석 불가")
                else:
                    # 1. 타겟 생성 (내일 주가가 올랐으면 1, 아니면 0)
                    # shift(-1)은 다음날 종가를 현재 행으로 가져옴
                    df_daily['Target'] = (df_daily['Close'].shift(-1) > df_daily['Close']).astype(int)
                    
                    # 마지막 행은 내일 데이터가 없으므로 학습에서 제외하되, 예측용(오늘 데이터)으로 따로 저장
                    valid_data = df_daily.iloc[:-1].copy() # 학습용
                    last_row = df_daily.iloc[[-1]].copy()   # 오늘 데이터 (내일 예측용)
                    
                    # 2. 학습에 사용할 피처(Feature) 선택
                    features = ['RSI', 'Interest_Rate', 'VIX', 'Disparity']
                    
                    X = valid_data[features]
                    y = valid_data['Target']
                    
                    # 3. 랜덤 포레스트 모델 학습
                    model = RandomForestClassifier(n_estimators=100, random_state=42, min_samples_split=10)
                    model.fit(X, y)
                    
                    # 4. 정확도 점검 (최근 20% 데이터로 테스트)
                    split = int(len(X) * 0.8)
                    X_train, X_test = X.iloc[:split], X.iloc[split:]
                    y_train, y_test = y.iloc[:split], y.iloc[split:]
                    
                    model_eval = RandomForestClassifier(n_estimators=100, random_state=42)
                    model_eval.fit(X_train, y_train)
                    preds = model_eval.predict(X_test)
                    acc = accuracy_score(y_test, preds)
                    
                    # 5. 내일 상승 확률 예측 (predict_proba)
                    # predict_proba 결과는 [하락확률, 상승확률] 형태
                    prediction_prob = model.predict_proba(last_row[features])[0][1]
                    
                    # --- 결과 UI ---
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(label="모델 과거 적중률", value=f"{acc*100:.1f}%")
                    
                    with col2:
                        color = "red" if prediction_prob >= 0.5 else "blue"
                        st.metric(label="내일 상승 확률", value=f"{prediction_prob*100:.1f}%")
                        
                    with col3:
                        signal = "매수 추천 (Strong Buy)" if prediction_prob > 0.6 else \
                                 "매도/관망 (Sell/Hold)" if prediction_prob < 0.4 else "중립 (Neutral)"
                        st.metric(label="AI 판단", value=signal)
                        
                    # 피처 중요도 시각화
                    st.markdown("#### AI가 중요하게 생각한 지표")
                    import_df = pd.DataFrame({
                        'Feature': features,
                        'Importance': model.feature_importances_
                    }).sort_values(by='Importance', ascending=False)
                    
                    st.bar_chart(import_df.set_index('Feature'))
                    
                    st.write("---")
                    st.write("##### 분석에 사용된 최근 데이터 (지표 포함)")
                    st.dataframe(df_daily.tail())

            except Exception as e:
                st.error(f"분석 중 오류 발생: {e}")
                st.write("상세 에러:", e)