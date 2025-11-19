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
@st.cache_data
def load_5m_data(ticker):
    # yfinance 제약: 5분봉은 최대 60일치만 가져올 수 있음
    data = yf.download(ticker, interval="5m", period="60d")
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
        
    data.reset_index(inplace=True)
    
    # Timezone 처리
    if 'Datetime' in data.columns:
        dt_col = data['Datetime']
        
        # 1. Datetime 컬럼이 이미 tz-aware인지 확인
        if dt_col.dt.tz is None:
            # tz-aware가 아니면 (대부분의 경우 yfinance가 처음 다운로드할 때) UTC를 부여
            dt_col = dt_col.dt.tz_localize('UTC')
        
        # 2. Datetime_kst 컬럼 생성 (KST로 변환)
        # 이미 tz-aware이므로 tz_convert를 사용
        data['Datetime_kst'] = dt_col.dt.tz_convert('Asia/Seoul')
        
        # 3. Prophet 사용을 위해 기존 Datetime 컬럼에서 Timezone 정보 제거 (tz_localize(None) 사용)
        data['Datetime'] = dt_col.dt.tz_localize(None)

    # 타임존 처리를 위한 Pandas 타입 변경 (혹시 모를 오류 방지)
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
# 3. 탭 구성
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["⏱️ 5분봉 단기 예측 (Real-time)", "📈 상승 확률 분석 (AI Classifier)"])

# ... (상단 import, load_5m_data 함수 등은 그대로) ...

# =========================================================
# TAB 1: 5분봉 단기 예측 (Real-time) - 범위 조절 옵션 추가
# =========================================================
with tab1:
    st.subheader(f"⏱️ {ticker} 5분봉 기반 단기 흐름 예측")
    st.info("💡 참고: 무료 API 제한으로 최근 60일 간의 5분봉 데이터만 사용합니다.")

    # 1. UI 옵션 추가: 표시할 캔들 개수 설정
    st.sidebar.markdown("---")
    st.sidebar.header("단기 예측 옵션")
    display_candlesticks = st.sidebar.slider("차트 표시 캔들 개수", 50, 600, 150, step=50) # 50개 ~ 600개, 기본값 150
    refresh_interval = st.sidebar.slider("자동 새로고침 간격 (초)", 30, 300, 60, step=30)
    
    placeholder = st.empty()
    
    # ---------------------------------------------------------
    # 자동 새로고침 로직
    # ---------------------------------------------------------
    while True:
        with placeholder.container():
            kst_now = datetime.now(pytz.timezone('Asia/Seoul'))
            st.write(f"마지막 업데이트 (KST): {kst_now.strftime('%Y-%m-%d %H:%M:%S')}")
            
            with st.spinner('5분봉 데이터 다운로드 및 예측 모델 학습 중...'):
                try:
                    df_5m = load_5m_data(ticker)
                    
                    if df_5m.empty:
                        st.error("데이터를 가져올 수 없습니다. 종목 코드 또는 시장 상황을 확인해주세요.")
                        st.stop()

                    # -------------------------------------------------------
                    # [핵심 수정] 표시할 캔들 개수만큼 데이터 슬라이싱
                    # -------------------------------------------------------
                    df_5m_display = df_5m.tail(display_candlesticks).copy() 
                    
                    # Prophet 학습용 데이터는 전체 데이터를 사용하고, 시각화만 슬라이싱된 데이터 사용
                    # (혹은 학습 데이터도 슬라이싱해서 속도를 높일 수도 있으나, 정확도를 위해 전체 사용)
                    
                    # Prophet 데이터셋 준비 (학습은 전체 데이터로)
                    df_prophet = df_5m[['Datetime_kst', 'Close']].rename(columns={'Datetime_kst': 'ds', 'Close': 'y'})
                    
                    # Prophet 학습 전, 'ds' 컬럼의 타임존 정보를 최종적으로 제거합니다.
                    if df_prophet['ds'].dt.tz is not None:
                        df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)
                        
                    df_prophet['y'] = pd.to_numeric(df_prophet['y'], errors='coerce')
                    
                    # 모델 학습
                    m = Prophet(changepoint_prior_scale=0.05, daily_seasonality=True, seasonality_mode='additive') 
                    m.fit(df_prophet)
                    
                    # 향후 12개 구간(60분) 예측
                    future = m.make_future_dataframe(periods=12, freq='5min')
                    forecast = m.predict(future)
                    
                    # forecast 데이터에 KST 타임존 정보를 부여
                    forecast['ds_kst'] = forecast['ds'].dt.tz_localize('Asia/Seoul')


                    # --- 차트 그리기 (캔들스틱 + 예측선) ---
                    fig = go.Figure()

                    # 1. 과거 캔들스틱 차트 (슬라이싱된 데이터 사용: df_5m_display)
                    fig.add_trace(go.Candlestick(
                        x=df_5m_display['Datetime_kst'],
                        open=df_5m_display['Open'],
                        high=df_5m_display['High'],
                        low=df_5m_display['Low'],
                        close=df_5m_display['Close'],
                        name='과거 주가',
                        increasing_line_color='red',
                        decreasing_line_color='blue'
                    ))

                    # 2. Prophet 예측 선 및 신뢰 구간 (전체 데이터 사용)
                    # 예측선
                    fig.add_trace(go.Scatter(
                        x=forecast['ds_kst'],
                        y=forecast['yhat'],
                        mode='lines',
                        line=dict(color='purple', width=2, dash='dot'),
                        name='예측 주가'
                    ))
                    # 신뢰 구간 (생략)
                    fig.add_trace(go.Scatter(
                        x=forecast['ds_kst'], y=forecast['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
                    ))
                    fig.add_trace(go.Scatter(
                        x=forecast['ds_kst'], y=forecast['yhat_lower'], mode='lines', line=dict(width=0), fill='tonexty', 
                        fillcolor='rgba(128,0,128,0.05)', name='예측 신뢰구간', hoverinfo='skip'
                    ))


                    # 3. 현재 종가 기준으로 수평 점선 추가
                    current_close_price = df_5m_display['Close'].iloc[-1]
                    fig.add_hline(
                        y=current_close_price, 
                        line_dash="dash", 
                        line_color="black", 
                        annotation_text=f"현재가: {current_close_price:.2f}", 
                        annotation_position="bottom right"
                    )

                    # 4. 예측값 텍스트 표시 (5분 뒤, 30분 뒤)
                    # ... (이하 텍스트 표시 로직은 변경 없음) ...
                    future_start_idx = forecast[forecast['ds'] > df_prophet['ds'].max()].index
                    
                    if not future_start_idx.empty:
                        future_start_idx = future_start_idx[0]
                        future_5min_idx = future_start_idx
                        future_30min_idx = future_start_idx + 5 

                        if future_30min_idx < len(forecast):
                            price_5min = forecast.loc[future_5min_idx, 'yhat']
                            price_30min = forecast.loc[future_30min_idx, 'yhat']
                            
                            time_5min_kst = forecast.loc[future_5min_idx, 'ds_kst'].strftime('%H:%M')
                            time_30min_kst = forecast.loc[future_30min_idx, 'ds_kst'].strftime('%H:%M')

                            # 5분 뒤 예상가 텍스트
                            fig.add_annotation(
                                x=forecast.loc[future_5min_idx, 'ds_kst'], y=price_5min,
                                text=f"+5분: {price_5min:.2f} ({time_5min_kst})",
                                showarrow=True, arrowhead=2, arrowcolor='green',
                                font=dict(color='green', size=12, family='Arial'),
                                xshift=10, yshift=10, bgcolor="rgba(255,255,255,0.7)"
                            )
                            # 30분 뒤 예상가 텍스트
                            fig.add_annotation(
                                x=forecast.loc[future_30min_idx, 'ds_kst'], y=price_30min,
                                text=f"+30분: {price_30min:.2f} ({time_30min_kst})",
                                showarrow=True, arrowhead=2, arrowcolor='green',
                                font=dict(color='green', size=12, family='Arial'),
                                xshift=10, yshift=10, bgcolor="rgba(255,255,255,0.7)"
                            )
                        
                    # --- 레이아웃 설정 ---
                    fig.update_layout(
                        title=f"📈 {ticker} 5분봉 실시간 차트 및 단기 예측 (표시 캔들: {len(df_5m_display)}개)",
                        xaxis_rangeslider_visible=False,
                        # ... (이하 레이아웃 설정은 동일) ...
                        
                        # 실제 데이터 영역과 예측 데이터 영역 배경색 구분
                        shapes=[
                            dict( # 예측 영역 배경색 (연한 초록)
                                type="rect",
                                xref="x", yref="paper",
                                x0=df_prophet['ds'].max(), 
                                y0=0, x1=forecast['ds_kst'].max(), y1=1,
                                fillcolor="rgba(0,255,0,0.05)", layer="below", line_width=0
                            ),
                            dict( # 과거 데이터 영역 배경색 (연한 파랑)
                                type="rect",
                                xref="x", yref="paper",
                                # [수정] 차트 표시 시작점을 슬라이싱된 데이터의 시작점으로 설정
                                x0=df_5m_display['Datetime_kst'].min(), 
                                y0=0, x1=df_prophet['ds'].max(), y1=1,
                                fillcolor="rgba(0,0,255,0.05)", layer="below", line_width=0
                            )
                        ],
                        xaxis=dict(
                            # [추가] 차트의 x축 시작점을 슬라이싱된 데이터의 시작점으로 강제 설정하여 깔끔하게 보임
                            range=[df_5m_display['Datetime_kst'].min(), forecast['ds_kst'].max()]
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.write("---")
                    st.write("##### 최근 5분봉 데이터 (마지막 업데이트 시점):")
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