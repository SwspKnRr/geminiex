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

# =========================================================
# TAB 1: 5분봉 단기 예측 (Real-time) - UI 최종 개선
# =========================================================
with tab1:
    st.subheader(f"⏱️ {ticker} 5분봉 기반 단기 흐름 예측")
    st.info("💡 참고: 무료 API 제한으로 최근 60일 간의 5분봉 데이터만 사용합니다. 차트는 일정 시간마다 새로고침됩니다.")

    refresh_interval = st.slider("자동 새로고침 간격 (초)", 30, 300, 60, step=30)
    
    placeholder = st.empty() # 실시간 차트를 업데이트할 빈 공간
    
    # ---------------------------------------------------------
    # 자동 새로고침 로직
    # ---------------------------------------------------------
    while True: # 무한 루프 (Streamlit 앱이 종료될 때까지 반복)
        with placeholder.container():
            # KST 현재 시간 표시
            kst_now = datetime.now(pytz.timezone('Asia/Seoul'))
            st.write(f"마지막 업데이트 (KST): {kst_now.strftime('%Y-%m-%d %H:%M:%S')}")
            
            with st.spinner('5분봉 데이터 다운로드 및 예측 모델 학습 중...'):
                try:
                    df_5m = load_5m_data(ticker)
                    
                    if df_5m.empty:
                        st.error("데이터를 가져올 수 없습니다. 종목 코드 또는 시장 상황을 확인해주세요.")
                        st.stop()
                    
                    # Prophet 데이터셋 준비
                    # df_5m 데이터프레임의 'Datetime' 컬럼을 'ds'로, 'Close' 컬럼을 'y'로 리네임합니다.
                    df_prophet = df_5m[['Datetime_kst', 'Close']].rename(columns={'Datetime_kst': 'ds', 'Close': 'y'})
                    
                    # 데이터가 문자열이 아닌 숫자형인지 최종 확인
                    df_prophet['y'] = pd.to_numeric(df_prophet['y'], errors='coerce')
                    
                    # 모델 학습
                    m = Prophet(changepoint_prior_scale=0.05, daily_seasonality=True, seasonality_mode='additive') 
                    m.fit(df_prophet)
                    
                    # 향후 12개 구간(60분) 예측 (5분 간격)
                    future = m.make_future_dataframe(periods=12, freq='5min')
                    forecast = m.predict(future)
                    
                    # Prophet의 forecast['ds'] 컬럼이 UTC이므로 KST로 변환
                    # yfinance 데이터는 이미 tz_localize(None)으로 타임존 제거되었으므로, forecast만 KST로 변경
                    # 실제 시각화 시점은 KST로 맞춰주기 위함
                    forecast['ds_kst'] = forecast['ds'].dt.tz_localize('UTC').dt.tz_convert('Asia/Seoul')
                    df_5m['Datetime_kst'] = df_5m['Datetime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Seoul')


                    # --- 차트 그리기 (캔들스틱 + 예측선) ---
                    fig = go.Figure()

                    # 1. 과거 캔들스틱 차트 (실제 데이터)
                    fig.add_trace(go.Candlestick(
                        x=df_5m['Datetime_kst'], # KST 시간 사용
                        open=df_5m['Open'],
                        high=df_5m['High'],
                        low=df_5m['Low'],
                        close=df_5m['Close'],
                        name='과거 주가',
                        increasing_line_color='red', # 양봉: 빨간색
                        decreasing_line_color='blue' # 음봉: 파란색
                    ))

                    # 2. Prophet 예측 선 (점선)
                    fig.add_trace(go.Scatter(
                        x=forecast['ds_kst'], # KST 시간 사용
                        y=forecast['yhat'],
                        mode='lines',
                        line=dict(color='purple', width=2, dash='dot'), # 보라색 점선
                        name='예측 주가'
                    ))

                    # 3. 예측 신뢰 구간 (연한 회색 배경)
                    fig.add_trace(go.Scatter(
                        x=forecast['ds_kst'], # KST 시간 사용
                        y=forecast['yhat_upper'],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    fig.add_trace(go.Scatter(
                        x=forecast['ds_kst'], # KST 시간 사용
                        y=forecast['yhat_lower'],
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty', 
                        fillcolor='rgba(128,0,128,0.05)', # 연한 보라색 계열로 신뢰구간 채우기
                        name='예측 신뢰구간',
                        hoverinfo='skip'
                    ))

                    # 4. 현재 종가 기준으로 수평 점선 추가
                    current_close_price = df_5m['Close'].iloc[-1]
                    fig.add_hline(
                        y=current_close_price, 
                        line_dash="dash", 
                        line_color="black", 
                        annotation_text=f"현재가: {current_close_price:.2f}", 
                        annotation_position="bottom right"
                    )

                    # 5. 예측값 텍스트 표시 (5분 뒤, 30분 뒤)
                    # 현재 시간 다음 5분 뒤 (1칸 뒤)
                    # df_prophet['ds'].max()는 마지막 실제 데이터의 시간
                    # forecast['ds']에서 이 시간 이후 첫 번째 인덱스 찾기
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
                                x=forecast.loc[future_5min_idx, 'ds_kst'], 
                                y=price_5min,
                                text=f"+5분: {price_5min:.2f} ({time_5min_kst})",
                                showarrow=True,
                                arrowhead=2,
                                arrowcolor='green',
                                font=dict(color='green', size=12, family='Arial'),
                                xshift=10, yshift=10,
                                bgcolor="rgba(255,255,255,0.7)"
                            )
                            # 30분 뒤 예상가 텍스트
                            fig.add_annotation(
                                x=forecast.loc[future_30min_idx, 'ds_kst'], 
                                y=price_30min,
                                text=f"+30분: {price_30min:.2f} ({time_30min_kst})",
                                showarrow=True,
                                arrowhead=2,
                                arrowcolor='green',
                                font=dict(color='green', size=12, family='Arial'),
                                xshift=10, yshift=10,
                                bgcolor="rgba(255,255,255,0.7)"
                            )
                        
                    # --- 레이아웃 설정 (첨부 이미지 UI와 유사하게) ---
                    fig.update_layout(
                        title=f"📈 {ticker} 5분봉 실시간 차트 및 단기 예측 (KST)",
                        xaxis_rangeslider_visible=False, # 하단 Range Slider 제거
                        xaxis_title="시간 (KST)",
                        yaxis_title="가격",
                        height=600,
                        hovermode="x unified",
                        template="plotly_white", # 흰색 배경 테마 (이미지와 유사)
                        margin=dict(l=20, r=20, t=50, b=20),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        plot_bgcolor='white', # 차트 내부 배경색
                        paper_bgcolor='white', # 차트 외부 배경색
                        # 실제 데이터 영역과 예측 데이터 영역 배경색 구분
                        shapes=[
                            dict( # 예측 영역 배경색 (연한 초록)
                                type="rect",
                                xref="x", yref="paper",
                                x0=df_prophet['ds'].max(), 
                                y0=0, 
                                x1=forecast['ds_kst'].max(), 
                                y1=1,
                                fillcolor="rgba(0,255,0,0.05)", # 연한 초록색
                                layer="below", 
                                line_width=0
                            ),
                            dict( # 과거 데이터 영역 배경색 (연한 파랑)
                                type="rect",
                                xref="x", yref="paper",
                                x0=df_prophet['ds_kst'].min(), 
                                y0=0, 
                                x1=df_prophet['ds_kst'].max(), 
                                y1=1,
                                fillcolor="rgba(0,0,255,0.05)", # 연한 파란색
                                layer="below", 
                                line_width=0
                            )
                        ]
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.write("---")
                    st.write("##### 최근 5분봉 데이터 (마지막 업데이트 시점):")
                    st.dataframe(df_5m.tail())
                    
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