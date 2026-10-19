import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. 페이지 설정 및 제목 (아이콘과 브라우저 탭 제목 설정)
st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 영화 유형 나누기")

# 2. 데이터 불러오기 및 전처리 함수
@st.cache_data
def load_and_preprocess_data():
    url = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"
    df_raw = pd.read_csv(url, encoding="utf-8")
    
    # 필수 속성 결측치 제거 및 첫 주 관객/스크린 수가 0 이하인 영화 제외
    df_clean = df_raw.dropna(subset=['first_scrn', 'first_week_audi', 'total_audi', 'days_in_top10']).copy()
    df_clean = df_clean[
        (df_clean['first_week_audi'] > 0) & 
        (df_clean['first_scrn'] > 0) & 
        (df_clean['total_audi'] > 0)
    ]
    
    # 속성 가공 (상용로그 및 롱런 지수 생성)
    df_clean['first_scrn_log'] = np.log10(df_clean['first_scrn'])
    df_clean['total_audi_log'] = np.log10(df_clean['total_audi'])
    df_clean['long_run_index'] = (df_clean['total_audi'] / df_clean['first_week_audi']).clip(upper=20)
    
    return df_raw, df_clean

df_raw, df = load_and_preprocess_data()

# 3. 전체 편수 및 묶은 편수 한 줄 표시
st.markdown(f"**전체 영화 수:** {len(df_raw):,}편 &nbsp;|&nbsp; **군집 분석에 사용된 영화 수:** {len(df):,}편")
st.write("---")

# 4. 묶는 데 사용할 속성 선택 UI (기본값: 4개 모두 선택)
feature_dict = {
    "스크린 수 (상용로그)": "first_scrn_log",
    "누적 관객 (상용로그)": "total_audi_log",
    "10위권 일수": "days_in_top10",
    "롱런 지수": "long_run_index"
}

st.subheader("1. 묶음에 사용할 속성 선택")
selected_feature_names = st.multiselect(
    "군집화에 사용할 속성을 2개 이상 선택하세요 (기본: 4개 모두 선택):",
    options=list(feature_dict.keys()),
    default=list(feature_dict.keys())
)

if len(selected_feature_names) < 2:
    st.warning("⚠️ 묶음을 나누기 위해서는 속성을 최소 2개 이상 선택해야 합니다.")
    st.stop()

selected_feature_cols = [feature_dict[name] for name in selected_feature_names]

# 5. 표준화 및 K-평균 군집화 (난수 고정 random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[selected_feature_cols])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
raw_clusters = kmeans.fit_predict(X_scaled)
df['raw_cluster'] = raw_clusters

# 누적 관객(total_audi) 평균이 큰 순서대로 ㉮, ㉯, ㉰ 라벨 부여
cluster_audi_means = df.groupby('raw_cluster')['total_audi'].mean().sort_values(ascending=False)
label_map = {old_label: new_label for old_label, new_label in zip(cluster_audi_means.index, ['㉮', '㉯', '㉰'])}
df['cluster'] = df['raw_cluster'].map(label_map)

cluster_order = ['㉮', '㉯', '㉰']

st.write("---")

# 6. 2차원 산점도 시각화 (Plotly)
st.subheader("2. 2차원 산점도 시각화")
col2d_1, col2d_2 = st.columns(2)
with col2d_1:
    x_axis_2d = st.selectbox("2D 가로축(X축) 속성
