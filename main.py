import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. 페이지 설정 및 제목 (브라우저 탭 아이콘 및 제목)
st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 영화 유형 나누기")

# 2. 데이터 불러오기 및 예외 처리 강화
@st.cache_data
def load_and_preprocess_data():
    url = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"
    try:
        df_raw = pd.read_csv(url, encoding="utf-8")
    except Exception as e:
        st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
        return None, None
    
    # 필수 열 숫자형 변환 (이상치 및 문자열 안전 처리)
    cols_to_num = ['first_scrn', 'first_week_audi', 'total_audi', 'days_in_top10']
    for col in cols_to_num:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
    
    # 필수 속성 결측치 제거 및 0 이하 값 제거
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

if df_raw is None or df is None or len(df) == 0:
    st.error("처리할 유효한 영화 데이터가 없습니다.")
    st.stop()

# 3. 전체 편수 및 묶은 편수 한 줄 표시
st.markdown(f"**전체 영화 수:** {len(df_raw):,}편 &nbsp;|&nbsp; **군집 분석에 사용된 영화 수:** {len(df):,}편")
st.write("---")

# 4. 묶는 데 사용할 속성 선택 UI
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

n_clusters = min(3, len(df))
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df['raw_cluster'] = kmeans.fit_predict(X_scaled)

# 누적 관객(total_audi) 평균이 큰 순서대로 ㉮, ㉯, ㉰ 라벨 부여
cluster_audi_means = df.groupby('raw_cluster')['total_audi'].mean().sort_values(ascending=False)
labels = ['㉮', '㉯', '㉰'][:n_clusters]
label_map = {old_label: new_label for old_label, new_label in zip(cluster_audi_means.index, labels)}
df['cluster'] = df['raw_cluster'].map(label_map)

cluster_order = labels

st.write("---")

# 6. 2차원 산점도 시각화
st.subheader("2. 2차원 산점도 시각화")
col2d_1, col2d_2 = st.columns(2)
with col2d_1:
    x_axis_2d = st.selectbox("2D 가로축(X축) 속성 선택:", options=selected_feature_names, index=0, key="2d_x")
with col2d_2:
    y_idx = 1 if len(selected_feature_names) > 1 else 0
    y_axis_2d = st.selectbox("2D 세로축(Y축) 속성 선택:", options=selected_feature_names, index=y_idx, key="2d_y")

fig_2d = px.scatter(
    df,
    x=feature_dict[x_axis_2d],
    y=feature_dict[y_axis_2d],
    color='cluster',
    category_orders={'cluster': cluster_order},
    hover_name='movieNm',
    hover_data=['cluster', feature_dict[x_axis_2d], feature_dict[y_axis_2d]],
    labels={feature_dict[x_axis_2d]: x_axis_2d, feature_dict[y_axis_2d]: y_axis_2d, 'cluster': '묶음'},
    title=f"2D 산점도 ({x_axis_2d} vs {y_axis_2d})"
)
fig_2d.update_layout(height=500)
st.plotly_chart(fig_2d, use_container_width=True)

st.write("---")

# 7. 3차원 산점도 시각화
st.subheader("3. 3차원 산점도 시각화")
if len(selected_feature_names) < 3:
    st.info("💡 3차원 산점도를 확인하려면 묶는 데 사용할 속성을 3개 이상 선택해 주세요.")
else:
    col3d_1, col3d_2, col3d_3 = st.columns(3)
    with col3d_1:
        x_axis_3d = st.selectbox("3D X축 속성 선택:", options=selected_feature_names, index=0, key="3d_x")
    with col3d_2:
        y_axis_3d = st.selectbox("3D Y축 속성 선택:", options=selected_feature_names, index=1, key="3d_y")
    with col3d_3:
        z_axis_3d = st.selectbox("3D Z축 속성 선택:", options=selected_feature_names, index=2, key="3d_z")

    fig_3d = px.scatter_3d(
        df,
        x=feature_dict[x_axis_3d],
        y=feature_dict[y_axis_3d],
        z=feature_dict[z_axis_3d],
        color='cluster',
        category_orders={'cluster': cluster_order},
        hover_name='movieNm',
        hover_data=['cluster'],
        labels={
            feature_dict[x_axis_3d]: x_axis_3d,
            feature_dict[y_axis_3d]: y_axis_3d,
            feature_dict[z_axis_3d]: z_axis_3d,
            'cluster': '묶음'
        },
        title=f"3D 산점도 ({x_axis_3d} / {y_axis_3d} / {z_axis_3d})"
    )
    fig_3d.update_traces(marker=dict(size=3))
    fig_3d.update_layout(height=600)
    st.plotly_chart(fig_3d, use_container_width=True)

st.write("---")

# 8. 묶음별 요약 표 (원래 단위 평균)
st.subheader("4. 묶음별 요약 및 속성 평균 (원래 단위)")

summary_list = []
for c_label in cluster_order:
    sub_df = df[df['cluster'] == c_label]
    cnt = len(sub_df)
    if cnt > 0:
        avg_scrn = sub_df['first_scrn'].mean()
        avg_audi = sub_df['total_audi'].mean()
        avg_top10 = sub_df['days_in_top10'].mean()
        avg_longrun = sub_df['long_run_index'].mean()
    else:
        avg_scrn = avg_audi = avg_top10 = avg_longrun = 0
    
    summary_list.append({
        "묶음": c_label,
        "편수": f"{cnt:,}편",
        "스크린 수 평균": f"{avg_scrn:,.1f}개",
        "누적 관객 평균": f"{avg_audi:,.0f}명",
        "10위권 일수 평균": f"{avg_top10:,.1f}일",
        "롱런 지수 평균": f"{avg_longrun:,.2f}"
    })

summary_df = pd.DataFrame(summary_list)
st.dataframe(summary_df, use_container_width=True, hide_index=True)

st.write("---")

# 9. 묶음별 누적 관객수 Top 5 영화
st.subheader("5. 묶음별 누적 관객수 상위 5개 영화")

cols = st.columns(len(cluster_order))
for idx, c_label in enumerate(cluster_order):
    with cols[idx]:
        st.markdown(f"### 묶음 {c_label}")
        top5 = df[df['cluster'] == c_label].sort_values(by='total_audi', ascending=False).head(5)
        if len(top5) > 0:
            top5_display = top5[['movieNm', 'total_audi', 'openDt']].reset_index(drop=True)
            top5_display.index += 1
            top5_display.columns = ['영화 제목', '누적 관객수', '개봉일']
            top5_display['누적 관객수'] = top5_display['누적 관객수'].apply(lambda x: f"{x:,.0f}명")
            st.table(top5_display)
        else:
            st.write("해당하는 영화가 없습니다.")
