import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pytz
from prophet import Prophet
from datetime import date, timedelta, datetime
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
# ---------------------------------------------------------
# 2. 데이터 로드 함수들
# ---------------------------------------------------------
# ---------------------------------------------------------
# 2. 데이터 로드 함수들
# ---------------------------------------------------------
@st.cache_data
def load_intraday_data(ticker, interval):
    # yfinance 제약: 5분봉은 최대 60일치만 가져올 수 있음
    # [수정] prepost=True 옵션을 추가하여 프리장/애프터장 데이터 포함
    data = yf.download(ticker, interval=interval, period="60d", prepost=True)
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
        
    data.reset_index(inplace=True)
    
    # Timezone 처리
    if 'Datetime' in data.columns:
        dt_col = data['Datetime']
        
        # 1. Datetime 컬럼이 이미 tz-aware인지 확인 (캐시 충돌 방지)
        if dt_col.dt.tz is None:
            dt_col = dt_col.dt.tz_localize('UTC')
        
        # 2. Datetime_kst 컬럼 생성 (KST로 변환)
        data['Datetime_kst'] = dt_col.dt.tz_convert('Asia/Seoul')
        
        # 3. Prophet 사용을 위해 기존 Datetime 컬럼에서 Timezone 정보 제거
        data['Datetime'] = dt_col.dt.tz_localize(None)

    # 타임존 처리를 위한 Pandas 타입 변경
    if 'Datetime_kst' in data.columns:
        data['Datetime_kst'] = pd.to_datetime(data['Datetime_kst'])
        
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

# ... (상단 import 및 공통 설정, load_5m_data 함수 등은 그대로) ...

# ---------------------------------------------------------
# 3. 탭 구성 및 사이드바 옵션 선언 [수정 및 추가]
# ---------------------------------------------------------

# 사이드바 옵션을 탭 외부에 먼저 선언
st.sidebar.markdown("---")
st.sidebar.header("단기 예측 옵션")

# 분봉 선택 옵션
interval_choice = st.sidebar.radio("분봉 선택", ('5m', '1m'))

# 분봉에 따라 캔들 개수 기본값 조정
if interval_choice == '1m':
    default_candles = 500
    max_candles = 2000
else:
    default_candles = 200
    max_candles = 1000
    
display_candlesticks = st.sidebar.slider("차트 표시 캔들 개수", 50, max_candles, default_candles, step=50)
refresh_interval = st.sidebar.slider("자동 새로고침 간격 (초)", 30, 300, 60, step=30)

# 탭 생성
tab1, tab2 = st.tabs(["⏱️ 단기 예측", "📈 상승 확률 분석"])

# =========================================================
# TAB 1: 5분봉 단기 예측 (Real-time) - 범위 및 분봉 옵션 추가
# =========================================================
# =========================================================
# TAB 1: 5분봉 단기 예측 (Real-time) - 수정된 TAB 1
# =========================================================
with tab1:
    st.subheader(f"⏱️ {ticker} {interval_choice} 단기 흐름 예측 (프리장/애프터장 포함)")
    st.info("💡 참고: 데이터 로드 시 **프리장(Pre-market) 및 애프터장(After-hours) 데이터가 포함**되어 학습됩니다.")

    placeholder = st.empty()
    
    # ---------------------------------------------------------
    # 자동 새로고침 로직
    # ---------------------------------------------------------
    while True:
        with placeholder.container():
            kst_now = datetime.now(pytz.timezone('Asia/Seoul'))
            st.write(f"마지막 업데이트 (KST): {kst_now.strftime('%Y-%m-%d %H:%M:%S')} - 현재 분봉: {interval_choice}")
            
            with st.spinner(f'{interval_choice} 데이터 다운로드 및 예측 모델 학습 중...'):
                try:
                    # [수정 없음] 변경된 함수 이름 및 인자 사용
                    df_5m = load_intraday_data(ticker, interval_choice) 
                    
                    if df_5m.empty:
                        st.error("데이터를 가져올 수 없습니다. 종목 코드 또는 시장 상황을 확인해주세요.")
                        st.stop()

                    # 표시할 캔들 개수만큼 데이터 슬라이싱 (최상위 변수 사용)
                    df_5m_display = df_5m.tail(display_candlesticks).copy() 
                    
                    # Prophet 예측 설정 (60분 예측을 목표로)
                    time_step = int(interval_choice.replace('m', ''))
                    prediction_periods = int(60 / time_step)
                    
                    # Prophet 데이터셋 준비 (이하 Prophet 로직 동일)
                    df_prophet = df_5m[['Datetime_kst', 'Close']].rename(columns={'Datetime_kst': 'ds', 'Close': 'y'})
                    
                    if df_prophet['ds'].dt.tz is not None:
                        df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)
                        
                    df_prophet['y'] = pd.to_numeric(df_prophet['y'], errors='coerce')
                    
                    # 모델 학습
                    m = Prophet(changepoint_prior_scale=0.05, daily_seasonality=True, seasonality_mode='additive') 
                    m.fit(df_prophet)
                    
                    # 미래 데이터 프레임 생성
                    future = m.make_future_dataframe(periods=prediction_periods, freq=interval_choice)
                    forecast = m.predict(future)
                    
                    # forecast 데이터에 KST 타임존 정보를 부여
                    forecast['ds_kst'] = forecast['ds'].dt.tz_localize('Asia/Seoul')


                    # ---------------------------------------------
                    # 차트 그리기 (최근 캔들 + 미래 예측 라인)
                    # ---------------------------------------------
                    # 1) x축을 전부 "KST 기준 타임존 제거된 datetime"으로 통일
                    df_5m_display = df_5m_display.copy()
                    df_5m_display['x'] = df_5m_display['Datetime_kst'].dt.tz_localize(None)

                    # Prophet 쪽도 같은 스케일 사용 (이미 KST 기준 naive datetime)
                    df_prophet = df_prophet.copy()
                    forecast = forecast.copy()
                    forecast['x'] = pd.to_datetime(forecast['ds'])  # ds 그대로 사용

                    # 2) "미래 예측 구간"만 따로 분리
                    last_hist_time = df_prophet['ds'].max()
                    future_mask = forecast['ds'] > last_hist_time
                    forecast_future = forecast.loc[future_mask].copy()

                    # ---------------------------------------------
                    # Figure 생성
                    # ---------------------------------------------
                    fig = go.Figure()

                    # (1) 최근 캔들 차트
                    fig.add_trace(go.Candlestick(
                        x=df_5m_display['x'],
                        open=df_5m_display['Open'],
                        high=df_5m_display['High'],
                        low=df_5m_display['Low'],
                        close=df_5m_display['Close'],
                        name='최근 가격'
                    ))

                    # (2) 미래 예측 라인 (과거 구간은 안 그림)
                    if not forecast_future.empty:
                        fig.add_trace(go.Scatter(
                            x=forecast_future['x'],
                            y=forecast_future['yhat'],
                            mode='lines',
                            line=dict(width=2, dash='dot'),
                            name='예측 가격'
                        ))

                    # (3) 현재가 수평선
                    current_close_price = df_5m_display['Close'].iloc[-1]
                    fig.add_hline(
                        y=current_close_price,
                        line_dash="dash",
                        line_color="black",
                        annotation_text=f"현재가: {current_close_price:.2f}",
                        annotation_position="bottom right"
                    )

                    # (4) +5분 / +30분 예측 값 표시
                    future_start_idx = forecast[forecast['ds'] > last_hist_time].index

                    if not future_start_idx.empty:
                        future_start_idx = future_start_idx[0]
                        time_step = int(interval_choice.replace('m', ''))

                        idx_5 = future_start_idx + int(5 / time_step) - 1
                        idx_30 = future_start_idx + int(30 / time_step) - 1

                        if idx_5 < len(forecast):
                            price_5 = forecast.loc[idx_5, 'yhat']
                            t_5 = forecast.loc[idx_5, 'x']
                            fig.add_annotation(
                                x=t_5,
                                y=price_5,
                                text=f"+5분: {price_5:.2f}",
                                showarrow=True,
                                arrowhead=2,
                                arrowcolor='green',
                                font=dict(color='green', size=12),
                                xshift=10,
                                yshift=10,
                                bgcolor="rgba(255,255,255,0.7)"
                            )

                        if idx_30 < len(forecast):
                            price_30 = forecast.loc[idx_30, 'yhat']
                            t_30 = forecast.loc[idx_30, 'x']
                            fig.add_annotation(
                                x=t_30,
                                y=price_30,
                                text=f"+30분: {price_30:.2f}",
                                showarrow=True,
                                arrowhead=2,
                                arrowcolor='green',
                                font=dict(color='green', size=12),
                                xshift=10,
                                yshift=10,
                                bgcolor="rgba(255,255,255,0.7)"
                            )

                    # (5) 레이아웃 (군더더기 다 빼고 깔끔하게)
                    fig.update_layout(
                        title=f"📈 {ticker} {interval_choice} 실시간 차트 및 단기 예측 (표시 캔들: {len(df_5m_display)}개)",
                        xaxis_title="시간 (KST)",
                        yaxis_title="가격",
                        xaxis_rangeslider_visible=False,
                        hovermode="x unified",
                        xaxis=dict(
                            range=[
                                df_5m_display['x'].min(),
                                forecast['x'].max() if not forecast.empty else df_5m_display['x'].max()
                            ]
                        )
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    
                    st.write("---")
                    st.write(f"##### 최근 {interval_choice} 데이터 (마지막 업데이트 시점):")
                    st.dataframe(df_5m_display.tail())
                    
                except Exception as e:
                    st.error(f"단기 예측 중 오류 발생: {e}")
                    st.write("상세 에러:", e)
        
        # 지정된 새로고침 간격만큼 대기
        import time
        time.sleep(refresh_interval)

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