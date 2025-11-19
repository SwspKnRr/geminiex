import streamlit as st
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from datetime import date

# ---------------------------------------------------------
# 1. 기본 설정 및 UI 구성 (UI Module)
# ---------------------------------------------------------
st.set_page_config(page_title="주가 예측 앱", layout="wide")
st.title("📈 AI 기반 주가 예측 프로그램")

# 사이드바 설정
st.sidebar.header("사용자 입력")
selected_stock = st.sidebar.text_input("종목 코드 입력 (예: AAPL, 005930.KS)", "AAPL")
n_years = st.sidebar.slider("예측할 기간 (년)", 1, 4)
period = n_years * 365

# ---------------------------------------------------------
# 2. 데이터 수집 모듈 (Data Loader)
# ---------------------------------------------------------
@st.cache_data # 데이터를 캐싱하여 속도 향상
def load_data(ticker):
    # 오늘 날짜까지 데이터 다운로드
    data = yf.download(ticker, start="2015-01-01", end=date.today().strftime("%Y-%m-%d"))
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('데이터를 불러오는 중...')
try:
    data = load_data(selected_stock)
    # 데이터가 비어있는 경우 예외 처리
    if data.empty:
        data_load_state.text('데이터 로드 실패! 종목 코드를 확인해주세요.')
        st.stop()
    else:
        data_load_state.text('데이터 로드 완료!')
except Exception as e:
    data_load_state.text(f'오류 발생: {e}')
    st.stop()

# ---------------------------------------------------------
# 3. 데이터 시각화 모듈 (Visualization - Raw Data)
# ---------------------------------------------------------
st.subheader(f"📊 {selected_stock} 과거 데이터 차트")

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="시가(Open)"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="종가(Close)"))
    fig.layout.update(title_text='시계열 데이터 (Time Series Data)', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

plot_raw_data()

# 최근 데이터 5행 보여주기
with st.expander("최근 데이터 보기"):
    st.write(data.tail())

# ---------------------------------------------------------
# 4. 예측 모델링 모듈 (Prediction Engine - Prophet)
# ---------------------------------------------------------
st.subheader(f"🔮 {n_years}년 후 주가 예측")

# Prophet 모델 학습을 위한 데이터 전처리
# Prophet은 컬럼명이 'ds' (날짜)와 'y' (값)이어야 함
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# 모델 학습
m = Prophet()
m.fit(df_train)

# 미래 데이터 프레임 생성
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# ---------------------------------------------------------
# 5. 예측 결과 시각화 (Visualization - Prediction)
# ---------------------------------------------------------
# Prophet 내장 Plotly 기능 사용
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1, use_container_width=True)

st.write("예측 데이터 상세 컴포넌트 (Trend, Weekly, Yearly)")
fig2 = m.plot_components(forecast)
st.pyplot(fig2) # Matplotlib 기반 차트