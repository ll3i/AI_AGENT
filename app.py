import os
import json
import requests
from io import BytesIO
import pandas as pd
import streamlit as st
import base64
from PIL import Image, ImageDraw, ImageFont
import sys

# Load .env if present (local dev)
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass

# Streamlit Cloud secrets → env var fallback
try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

# Ensure run_solution is importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    import run_solution
except ImportError:
    st.error("run_solution.py 파일을 찾을 수 없습니다. 경로를 확인해주세요.")

# --------------------------
# Page config
# --------------------------
st.set_page_config(page_title="Visual Inspection Results", layout="wide")

# --------------------------
# Pretendard Font & Pixie-Style CSS
# --------------------------
st.markdown("""
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');

/* ── Global Font ── */
html, body, [class*="st-"], .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stSidebar"],
.stMarkdown, .stMarkdown p, .stMarkdown span,
label, input, textarea, button, select {
    font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ── White background ── */
.stApp, [data-testid="stAppViewContainer"] > section {
    background: #ffffff !important;
}
[data-testid="stAppViewContainer"] > section > div { background: #ffffff !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #fafafa !important;
    border-right: 1px solid #e8e8e8 !important;
}

/* ── Typography ── */
h1 {
    color: #000 !important;
    font-size: 40px !important;
    font-weight: 700 !important;
    line-height: 130% !important;
    letter-spacing: -0.1px !important;
}
h2 {
    color: #000 !important;
    font-size: 28px !important;
    font-weight: 600 !important;
    line-height: 140% !important;
}
h3 {
    color: #000 !important;
    font-size: 24px !important;
    font-weight: 600 !important;
    line-height: 140% !important;
}
h4 {
    color: #000 !important;
    font-size: 18px !important;
    font-weight: 600 !important;
}
p, li, span {
    color: #474747 !important;
    font-weight: 400 !important;
    line-height: 160% !important;
    letter-spacing: -0.1px !important;
}

/* ── Tabs (Pixie nav style) ── */
.stTabs [data-baseweb="tab-list"] {
    background: #ffffff !important;
    border-bottom: 1px solid #e8e8e8 !important;
    gap: 0px !important;
}
.stTabs [data-baseweb="tab"] {
    font-size: 16px !important;
    font-weight: 600 !important;
    color: #000 !important;
    padding: 12px 24px !important;
    border-radius: 0 !important;
}
.stTabs [aria-selected="true"] {
    color: #1454FE !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    background: #1454FE !important;
    height: 3px !important;
}

/* ── Metric cards (F5F5F5 card style) ── */
[data-testid="stMetric"] {
    background: #F5F5F5 !important;
    border: none !important;
    border-radius: 20px !important;
    padding: 20px 24px !important;
    min-height: 125px !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: flex-start !important;
}
[data-testid="stMetricLabel"] p {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #000 !important;
    text-transform: none !important;
}
[data-testid="stMetricValue"] {
    font-weight: 700 !important;
    color: #000 !important;
}
[data-testid="stMetricDelta"] {
    color: #1454FE !important;
}

/* ── Buttons (Pixie blue) ── */
.stButton > button {
    font-weight: 600 !important;
    border-radius: 12px !important;
    border: 1px solid #e8e8e8 !important;
    padding: 8px 20px !important;
}
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    background: #1454FE !important;
    color: white !important;
    border: none !important;
}
.stButton > button:hover {
    border-color: #1454FE !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #F5F5F5 !important;
    border-radius: 16px !important;
    font-weight: 600 !important;
    color: #000 !important;
    border: none !important;
    padding: 12px 16px !important;
}
.streamlit-expanderHeader p,
[data-testid="stExpander"] summary span p {
    font-size: 15px !important;
    color: #000 !important;
    padding-left: 4px !important;
}
/* Expander icon fix */
[data-testid="stExpander"] summary svg {
    flex-shrink: 0 !important;
    margin-right: 8px !important;
}
[data-testid="stExpander"] summary {
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
}
[data-testid="stExpander"] summary > span {
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    overflow: visible !important;
}
[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    overflow: visible !important;
}

/* ── Data editor / DataFrame ── */
[data-testid="stDataEditor"], [data-testid="stDataFrame"] {
    border-radius: 16px !important;
    overflow: hidden !important;
}

/* ── Chat ── */
[data-testid="stChatMessage"] {
    border-radius: 20px !important;
    background: #ffffff !important;
    border: 1px solid #e8e8e8 !important;
    padding: 16px 20px !important;
    margin-bottom: 12px !important;
}
[data-testid="stChatMessage"] [data-testid="stChatMessageAvatar"] {
    background-color: transparent !important;
    font-size: 22px !important;
    border-radius: 0 !important;
    width: 32px !important;
    height: 32px !important;
    align-items: flex-start !important;
}
[data-testid="stChatMessage"] .stMarkdown p {
    color: #222222 !important;
    font-size: 15px !important;
}

/* ── Custom card class ── */
.pixie-card {
    background: #F5F5F5;
    border-radius: 24px;
    padding: 28px 32px;
    margin-bottom: 16px;
}
.pixie-card-white {
    background: #ffffff;
    border-radius: 20px;
    padding: 21px 28px;
    margin-top: 12px;
}
.pixie-accent { color: #1454FE !important; font-weight: 600 !important; }
.pixie-label { color: #FE3F14 !important; font-size: 14px !important; font-weight: 400 !important; }
.pixie-section-title {
    color: #000; font-size: 24px; font-weight: 600;
    line-height: 33.6px; margin-bottom: 8px;
}
.pixie-desc {
    color: #474747; font-size: 16px; font-weight: 400;
    line-height: 24px;
}

/* ── Column overflow fix ── */
[data-testid="stColumn"] {
    overflow: visible !important;
}

/* ── Divider ── */
hr { border: none !important; height: 1px !important; background: #e8e8e8 !important; }

/* ── Hide defaults ── */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# --------------------------
# Helpers
# --------------------------
def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def safe_json(val):
    try:
        return json.loads(val)
    except:
        return {}

def draw_boxes(img, roi_meta, show_roi, show_sub, disp_width):
    """Draws ROI and Sub boxes on the given PIL Image based on JSON metadata."""
    if not isinstance(roi_meta, dict):
        return img
        
    orig_w, orig_h = img.size
    factor_w = disp_width / orig_w
    factor_h = disp_width / orig_h
    new_h = int(orig_h * factor_w)
    
    # 렌더링 품질을 위해 박스 그리기 전 이미지를 먼저 리사이즈
    img = img.resize((disp_width, new_h), Image.LANCZOS)
    draw = ImageDraw.Draw(img)
    
    # 디스플레이 크기에 맞춰 폰트 사이즈 조절 (기준 450px일때 15~18정도)
    font_size = max(10, int(18 * (disp_width / 450.0)))
    try:
        font = ImageFont.truetype("arial.ttf", size=font_size)
    except IOError:
        font = ImageFont.load_default()

    # ROI 박스 (빨간색)
    if show_roi and "roi_box" in roi_meta:
        box = roi_meta["roi_box"]
        x1, y1, x2, y2 = box.get("x1", 0), box.get("y1", 0), box.get("x2", 0), box.get("y2", 0)
        
        # 비율 적용
        x1, y1, x2, y2 = int(x1 * factor_w), int(y1 * factor_h), int(x2 * factor_w), int(y2 * factor_h)
        
        # 선 두께 (화면 큰 정도에 비례, 최소 2)
        line_w = max(2, int(2 * (disp_width / 450.0)))
        draw.rectangle([x1, y1, x2, y2], outline="red", width=line_w)
            
    # 서브 박스 (초록색)
    if show_sub and "sub_boxes_in_roi" in roi_meta and "roi_box" in roi_meta:
        rx1, ry1 = roi_meta["roi_box"].get("x1", 0), roi_meta["roi_box"].get("y1", 0)
        sub_boxes = roi_meta["sub_boxes_in_roi"]
        
        for i, sb in enumerate(sub_boxes):
            sx1, sy1, sx2, sy2 = sb.get("x1", 0), sb.get("y1", 0), sb.get("x2", 0), sb.get("y2", 0)
            
            # Sub-박스는 ROI 범위 내 상대좌표이므로 rx1, ry1을 더해줌
            ox1 = int((rx1 + sx1) * factor_w)
            oy1 = int((ry1 + sy1) * factor_h)
            ox2 = int((rx1 + sx2) * factor_w)
            oy2 = int((ry1 + sy2) * factor_h)
            
            line_w = max(2, int(2 * (disp_width / 450.0)))
            draw.rectangle([ox1, oy1, ox2, oy2], outline="#00FF00", width=line_w)
            
            # 박스 좌상단에 H1, H2, H3 텍스트 그리기
            label_text = f"H{i+1}"
            
            # Use textbbox to get text dimensions
            if hasattr(draw, "textbbox"):
                bbox = draw.textbbox((ox1, oy1), label_text, font=font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
            else:
                tw, th = font.getsize(label_text)
            
            # 텍스트 가독성을 위해 배경 상자 그리고 표시
            draw.rectangle([ox1, oy1, ox1 + tw + 4, oy1 + th + 4], fill="#00FF00")
            draw.text((ox1 + 2, oy1 + 2), label_text, fill="black", font=font)

    return img

# --------------------------
# UI Sidebar
# --------------------------
with st.sidebar:
    st.title("Web service (Streamlit)")
    st.markdown("---")
    
    st.markdown("**경로 설정**")
    default_explain_path = "explanations.csv" if os.path.exists("explanations.csv") else os.path.join(os.path.dirname(os.path.abspath(__file__)), "explanations.csv")
    explain_path_raw = st.text_input("explanations.csv 경로", value=default_explain_path)
    explain_path = explain_path_raw.strip(' "\'')
    if os.path.isdir(explain_path):
        explain_path = os.path.join(explain_path, "explanations.csv")
    
    default_img_path = "images" if os.path.exists("images") else os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
    image_folder_raw = st.text_input("이미지 폴더 (images) 경로", value=default_img_path)
    image_folder = image_folder_raw.strip(' "\'')
    
    st.markdown("---")
    st.markdown("**시각화 옵션**")
    show_roi = st.checkbox("ROI 박스(roi_box)", value=True)
    show_sub = st.checkbox("서브 박스(sub_boxes_in_roi)", value=True)
    
    st.markdown("---")
    st.markdown("**표시 크기**")
    disp_width = st.slider("표시 이미지 폭 (px)", min_value=200, max_value=900, value=450, step=10)
    st.caption("※ 박스는 원본 크기에 그린 뒤, 표시용으로만 축소합니다.")


# --------------------------
# Main Panel
# --------------------------
st.markdown("<h1>Inspection Web Platform</h1>", unsafe_allow_html=True)

# --------------------------
# Session State Initialization
# --------------------------
if "current_id" not in st.session_state:
    st.session_state["current_id"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "reval_result" not in st.session_state:
    st.session_state["reval_result"] = None
if "erp_sync_logs" not in st.session_state:
    st.session_state["erp_sync_logs"] = []

df = load_csv(explain_path)

if df is None:
    st.warning(f"`explanations.csv` 파일을 다음 경로에서 찾을 수 없습니다: `{explain_path}`\n경로를 확인해주세요.")
    st.stop()

# 파싱 가능한지 검사
required_cols = ["id", "label", "confidence", "valid", "used_vote", "checks", "roi_meta"]
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    st.error(f"`explanations.csv`에 필수 컬럼이 없습니다: {missing_cols}")
    st.stop()

# Pre-compute shared data
df_abnormal = df[df["label"] == 1].reset_index(drop=True)
total = len(df)
abnormal = len(df_abnormal)
normal = total - abnormal
avg_conf = df["confidence"].mean() if "confidence" in df.columns else 0.0

parsed_list = df["checks"].apply(safe_json) if "checks" in df.columns else pd.Series([{}]*total)
pkg_damage = sum(1 for p in parsed_list if p.get("package_intact") is False)
def _is_faulty(p, key):
    return 1 if p.get(key) in ["not_connected", "uncertain"] else 0
h1_faults = sum(_is_faulty(p, "hole1_status") for p in parsed_list)
h2_faults = sum(_is_faulty(p, "hole2_status") for p in parsed_list)
h3_faults = sum(_is_faulty(p, "hole3_status") for p in parsed_list)

test_csv_path = os.path.join(os.path.dirname(explain_path), "test.csv")
if not os.path.exists(test_csv_path):
    test_csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test.csv")
test_df = load_csv(test_csv_path)

# Tabs
tabs = st.tabs([
    "사진 검사", 
    "통합 대시보드", 
    "튜닝 랩", 
    "데일리 보고서", 
    "실시간 모니터링", 
    "ERP 연동"
])
tab_vis, tab_dash, tab_tuning, tab_report, tab_alert, tab_erp = tabs

# -----------------------------------------------------
# TAB 1: 사진 검사 + VQA
# -----------------------------------------------------
with tab_vis:
    if len(df_abnormal) == 0:
        st.info("label=1(비정상)으로 판단된 이미지가 없습니다.")
    else:
        # 섹션 헤더
        st.markdown('<span style="color:#1454FE; font-size:18px; font-weight:600;">불량 판독 결과</span> <span style="color:#000; font-size:18px; font-weight:600;">시각화 및 AI 분석</span>', unsafe_allow_html=True)
        
        selected_id = st.selectbox("불량 ID 선택", options=df_abnormal["id"].tolist(), key="vis_id", label_visibility="collapsed")
        
        if st.session_state.get("current_id") != selected_id:
            st.session_state["current_id"] = selected_id
            st.session_state["chat_history"] = []
            st.session_state["reval_result"] = None

        row = df_abnormal[df_abnormal["id"] == selected_id].iloc[0]
        checks_data = safe_json(row["checks"])
        roi_meta_data = safe_json(row["roi_meta"])
        use_enhance = checks_data.get("use_enhance", False)
        
        img = None
        img_url = None
        if test_df is not None and "id" in test_df.columns:
            row_test = test_df[test_df["id"] == selected_id]
            if not row_test.empty:
                img_url = row_test.iloc[0]["img_url"]
                try:
                    b64_full, _, _, _, _, meta = run_solution.make_images_b64(img_url, use_enhance=use_enhance)
                    img_data = base64.b64decode(b64_full)
                    img = Image.open(BytesIO(img_data)).convert("RGB")
                except Exception as e:
                    st.error(f"전처리 이미지 생성 에러: {e}")

        # === 상단: 이미지(L) + 메타데이터(R) ===
        col_img, col_meta = st.columns([1.3, 1], gap="large")
        
        with col_img:
            if img:
                drawn_img = draw_boxes(img, meta, show_roi, show_sub, disp_width)
                st.image(drawn_img, use_container_width=True)
            
            # 검사 결과 요약 카드
            st.markdown(f'''
            <div class="pixie-card">
                <div style="display:flex; gap:24px; flex-wrap:wrap;">
                    <div style="flex:1; min-width:100px;">
                        <span class="pixie-label">{row["confidence"]:.4f}</span>
                        <div style="color:#000; font-size:16px; font-weight:600; line-height:24px;">확신도</div>
                    </div>
                    <div style="flex:1; min-width:100px;">
                        <span class="pixie-label">{"Yes" if row["used_vote"] else "No"}</span>
                        <div style="color:#000; font-size:16px; font-weight:600; line-height:24px;">다수결 재검사</div>
                    </div>
                    <div style="flex:1; min-width:100px;">
                        <span class="pixie-label">{"On" if use_enhance else "Off"}</span>
                        <div style="color:#000; font-size:16px; font-weight:600; line-height:24px;">화질 보정</div>
                    </div>
                </div>
                <div class="pixie-card-white">
                    <div style="display:flex; align-items:center; gap:13px; margin-bottom:8px;">
                        <div style="width:16px; height:21px; background:#1454FE; border-radius:3px;"></div>
                        <span style="color:#000 !important; font-size:18px; font-weight:700;">판독 요약</span>
                    </div>
                    <span style="color:#000 !important; font-size:16px; font-weight:400; line-height:24px;">{row.get("reason", "판독 정보 없음")}</span>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col_meta:
            # LLM 제어판
            st.subheader("🚀 AI 실시간 재판독")
            st.caption("선택된 샘플의 판독을 LLM에게 다시 요청합니다.")
            if st.button("🚀 재판독 실행", use_container_width=True, type="primary"):
                with st.spinner("API 서버 응답 중..."):
                    try:
                        st.session_state["reval_result"] = run_solution.observe_once(img_url, use_enhance=use_enhance)
                    except Exception as e:
                        st.error(f"API 에러: {e}")
            if st.session_state["reval_result"]:
                st.success("재평가 완료!")
                st.json(st.session_state["reval_result"])
            
            st.markdown("---")
            
            # 메타데이터 JSON (expander 대신 checkbox 토글 사용)
            st.subheader("상세 메타데이터")
            if st.checkbox("checks 데이터 보기", key="show_checks"):
                st.json(checks_data)
            if st.checkbox("roi_meta 데이터 보기", key="show_roi"):
                st.json(roi_meta_data)

        # === 하단: VQA 챗봇 ===
        st.markdown('---')
        st.markdown(f'''
        <div class="pixie-card">
            <div class="pixie-section-title">💬 시각 지능(VQA) 챗봇</div>
            <p class="pixie-desc">현재 보고 계신 이미지와 판독 결과를 기반으로 AI에게 자유롭게 질문하세요.</p>
        </div>
        ''', unsafe_allow_html=True)
        
        chat_box = st.container(height=350)
        with chat_box:
            for msg in st.session_state["chat_history"]:
                avatar_icon = "👤" if msg["role"] == "user" else "🤖"
                with st.chat_message(msg["role"], avatar=avatar_icon): st.markdown(msg["content"])
        
        if prompt := st.chat_input("이 이미지의 손상 여부가 궁금하신가요?"):
            st.session_state["chat_history"].append({"role": "user", "content": prompt})
            with chat_box:
                with st.chat_message("user", avatar="👤"): st.markdown(prompt)
                with st.chat_message("assistant", avatar="🤖"):
                    with st.spinner("분석 중..."):
                        try:
                            if img_url:
                                current_reason = st.session_state["reval_result"].get("reason", "") if st.session_state["reval_result"] else row.get("reason", "")
                                sys_msg = (
                                    "You are an expert semiconductor inspection bot.\n"
                                    f"Current defect explanation: {current_reason}\n"
                                    "Answer in Korean."
                                )
                                full_64, ann_64, _, _, _, _ = run_solution.make_images_b64(img_url, use_enhance=use_enhance)
                                messages = [
                                    {"role": "system", "content": sys_msg},
                                    {"role": "user", "content": [
                                        {"type": "text", "text": prompt},
                                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{full_64}"}},
                                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{ann_64}"}},
                                    ]}
                                ]
                                answer = run_solution._post_chat(messages, timeout=60, max_retries=1)
                                st.markdown(answer)
                                st.session_state["chat_history"].append({"role": "assistant", "content": answer})
                        except Exception as e:
                            st.error(f"오류: {e}")

# -----------------------------------------------------
# TAB 2: 통합 대시보드
# -----------------------------------------------------
with tab_dash:
    st.subheader("📊 통합 데이터 관리 대시보드")
    st.markdown("`explanations.csv` 에 누적된 전체 검사 데이터를 기반으로 품질 현황을 요약합니다.")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Inspections", f"{total:,} 건")
    m2.metric("Normal (정상)", f"{normal:,} 건", "목표수량대비", delta_color="normal")
    m3.metric("Abnormal (불량)", f"{abnormal:,} 건", f"불량률: {abnormal/total*100:.1f}%" if total else "0%", delta_color="inverse")
    m4.metric("Average Confidence", f"{avg_conf*100:.1f} %")
    
    st.markdown("---")

    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("1. 양품 vs 불량 비율 (Target vs Defect)")
        pie_data = pd.DataFrame({
            "판정 결과": ["Normal (0)", "Abnormal (1)"],
            "수량": [normal, abnormal]
        }).set_index("판정 결과")
        if total > 0:
            st.bar_chart(pie_data)
        else:
            st.info("데이터가 없습니다.")
            
    with c2:
        st.markdown("2. API 요청 검사 유효성 (Success/Fail)")
        if "valid" in df.columns:
            valid_counts = df["valid"].value_counts().reset_index()
            valid_counts.columns = ["유효성(Valid)", "수량"]
            valid_counts.set_index("유효성(Valid)", inplace=True)
            st.bar_chart(valid_counts)

    st.markdown("---")
    c3, c4 = st.columns(2)
    
    with c3:
        st.markdown("3. 세부 불량 요인 분석 (Defect Breakdown)")
        st.caption("어떤 부위에서 불량이 가장 자주 일어나는지 부품별로 필터링한 데이터입니다.")
        defect_data = pd.DataFrame({
            "부위별 고장 빈도": ["패키지 외형 손상", "H1 (좌측) 핀 단선", "H2 (중앙) 핀 단선", "H3 (우측) 핀 단선"],
            "누적 횟수": [pkg_damage, h1_faults, h2_faults, h3_faults]
        }).set_index("부위별 고장 빈도")
        st.bar_chart(defect_data, color="#FF4B4B")
        
    with c4:
        st.markdown("4. 검사 확신도(Confidence) 추세")
        st.caption("AI 판단 모델이 각 샘플에서 보여준 신뢰도 편차입니다.")
        if "confidence" in df.columns:
            st.line_chart(df["confidence"], color="#00FF00")

    st.markdown("---")
    c5, c6 = st.columns(2)
    
    with c5:
        st.markdown("5. 난이도별 시스템 처리량 (다수결 발동 횟수)")
        st.caption("confidence 기준 미달로 집중 재판독(used_vote)을 거친 케이스.")
        if "used_vote" in df.columns:
            vote_counts = df["used_vote"].value_counts(dropna=False).rename({True: "복합 재검사(True)", False: "일반 판별(False)"}).reset_index()
            vote_counts.columns = ["분류", "수량"]
            vote_counts.set_index("분류", inplace=True)
            st.bar_chart(vote_counts)

    with c6:
        st.markdown("6. 혼동 자가교정 시도 (CoVe 발동)")
        st.caption("uncertain 판정으로 인해 시스템 내 추가 교정 로직이 사용된 빈도.")
        if "checks" in df.columns:
            cove_count = sum(1 for p in parsed_list if p.get("cove_used", False) is True)
            nocove_count = len(parsed_list) - cove_count
            cove_data = pd.DataFrame({"사용여부": ["CoVe 활성", "미사용"], "횟수": [cove_count, nocove_count]}).set_index("사용여부")
            st.bar_chart(cove_data)

# -----------------------------------------------------
# TAB 4: 🎛️ 튜닝 랩 (Threshold & Vision)
# -----------------------------------------------------
with tab_tuning:
    st.markdown("## 🎛️ 파라미터 / 튜닝 ")
    st.markdown("이곳에서는 인공지능의 합격 판정 컷오프(임계값)와 이미지 전처리 필터 성능을 즉각적으로 조절하고 시뮬레이션 해볼 수 있습니다.")
    
    st.markdown("---")
    st.subheader("1. AI 판정 임계값(Threshold) 시뮬레이터")
    st.caption("Confidence 임계값을 극단적으로 조절하면 수율(합격률)이 어떻게 변동되는지 실험해 보세요.")
    
    if "confidence" in df.columns:
        conf_min = float(df["confidence"].min())
        conf_max = float(df["confidence"].max())
        conf_mean = float(df["confidence"].mean())
        default_thresh = round(max(conf_min, min(conf_mean, conf_max)), 2)
    else:
        conf_min, conf_max, default_thresh = 0.0, 1.0, 0.5

    sim_thresh = st.slider(
        "가상 Confidence 커트라인 설정",
        min_value=0.0, max_value=1.0,
        value=default_thresh, step=0.01,
        help=f"현재 데이터의 confidence 범위: {conf_min:.2f} ~ {conf_max:.2f} (평균 {conf_mean:.2f})"
    )

    if "confidence" in df.columns:
        strict_fails = len(df[(df["label"] == 1) | (df["confidence"] < sim_thresh)])
        strict_pass = len(df) - strict_fails
        
        c_s1, c_s2 = st.columns(2)
        c_s1.metric("시뮬레이션 후 양품(Pass)", f"{strict_pass} 개", f"{strict_pass - normal} 개", delta_color="normal")
        c_s2.metric("시뮬레이션 후 불량(Fail)", f"{strict_fails} 개", f"{strict_fails - abnormal} 개", delta_color="inverse")
        
        sim_data = pd.DataFrame({"상태": ["새로운 양품(Pass)", "새로운 불량(Fail)"], "수량": [strict_pass, strict_fails]}).set_index("상태")
        st.bar_chart(sim_data, color="#86b4ff")
    else:
        st.info("데이터에 confidence 컬럼이 없습니다.")

    st.markdown("---")
    st.subheader("2. 비전 프로세싱 놀이터")
    st.caption("현재 적용된 고대비 보정알고리즘(CLAHE) 및 샤프닝 계수를 조작하여 원본과 어떻게 차이가 나는지 즉석에서 확인합니다.")
    
    val_ids = [r for r in df["id"].tolist() if str(r).startswith("TEST_")] # TEST 셋트만
    if val_ids:
        v_id = st.selectbox("비교할 이미지 선택", val_ids[:20]) # Limit for performance
        
        c_v1, c_v2 = st.columns([1, 2])
        with c_v1:
            st.markdown("**⚙️ 처리 옵션 필터 설정**")
            v_clip = st.slider("CLAHE Clip Limit (대비)", 0.5, 5.0, 2.0, 0.1)
            v_sharp = st.slider("Sharpen Factor (선명도)", 0.5, 5.0, 1.8, 0.1)
            test_run = st.button("비교 렌더링 확인")
            
        with c_v2:
            if test_run:
                with st.spinner("이미지 인출 및 비전 연산 중..."):
                    try:
                        v_df = load_csv(test_csv_path)
                        t_url = v_df[v_df["id"] == v_id].iloc[0]["img_url"]
                        
                        raw_bytes = run_solution._download_image_bytes(t_url)
                        raw_pil = Image.open(BytesIO(raw_bytes)).convert("RGB").resize((224,224))
                        
                        import numpy as np
                        from PIL import ImageEnhance
                        arr = np.array(raw_pil)
                        gray = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.uint8)
                        eq = run_solution._clahe_gray_np(gray, clip_limit=v_clip, grid=(8,8))
                        ratio = (eq.astype(np.float32) + 1.0) / (gray.astype(np.float32) + 1.0)
                        ratio = np.clip(ratio, 0.7, 1.6)[..., None]
                        arr2 = np.clip(arr.astype(np.float32) * ratio, 0, 255).astype(np.uint8)
                        img2 = Image.fromarray(arr2)
                        enh_pil = ImageEnhance.Sharpness(img2).enhance(v_sharp)
                        
                        tc1, tc2 = st.columns(2)
                        tc1.image(raw_pil, caption="[RAW] 원본 이미지", use_column_width=True)
                        tc2.image(enh_pil, caption=f"[ENHANCED] 커스텀 설정", use_column_width=True)
                    except Exception as e:
                        st.error(f"처리 오류: {e}")

# -----------------------------------------------------
# TAB 5: 📝 LLM 스마트 일일 보고서 자동 생성
# -----------------------------------------------------
with tab_report:
    st.subheader("📝 데일리 품질 요약 LLM 보고서")
    st.markdown("오늘자 전체 데이터를 종합하여 LLM 시스템이 경영진 보고용 텍스트를 기안해 줍니다.")

    if st.button("✨ 오늘자 보고서 자동 작성 (GPT)"):
        with st.spinner("AI가 데이터를 종합하여 보고서를 쓰고 있습니다..."):
            stats_str = f"전체:{total}건, 정상:{normal}건, 불량:{abnormal}건, 평균신뢰도:{avg_conf:.2f}. 파손유형패키지:{pkg_damage}, H1단선:{h1_faults}, H2단선:{h2_faults}, H3단선:{h3_faults}"
            prompt = [
                {"role": "system", "content": "You are a factory manager robot. Summarize the following daily inspection stats in professional Korean markdown report format. Identify the most critical defect origin."},
                {"role": "user", "content": stats_str}
            ]
            try:
                api_key = os.environ.get("OPENAI_API_KEY", "")
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                payload = {"model": run_solution.MODEL, "messages": prompt, "max_completion_tokens": 4096}
                r = requests.post(run_solution.BRIDGE_URL, headers=headers, json=payload, timeout=60)
                if r.status_code != 200:
                    raise RuntimeError(f"status={r.status_code}, body={r.text[:300]}")
                report_txt = r.json()["choices"][0]["message"]["content"].strip()
                st.session_state["daily_report"] = report_txt
            except Exception as e:
                st.error(f"LLM API 실패: {e}")

    if "daily_report" in st.session_state:
        st.success("보고서 생성 완료")
        st.markdown(st.session_state["daily_report"])
        st.download_button("문서로 다운로드 (.md)", st.session_state["daily_report"], file_name="daily_report.md")

# -----------------------------------------------------
# TAB 6: 🚨 라인 실시간 스트리밍 모니터링 시뮬레이터
# -----------------------------------------------------
with tab_alert:
    st.subheader("🚨 라인 실시간 모니터링 시뮬레이터")
    st.caption("초당 1건씩 데이터가 들어오는 가상 환경에서, 불량률이 치솟을 때 알람이 발생하는 현장 콘솔을 시뮬레이션 합니다.")
    
    if st.button("▶️ 실시간 스트리밍 시뮬레이션 가동 (30초)", type="primary"):
        alert_placeholder = st.empty()
        chart_placeholder = st.empty()
        import time
        import numpy as np
        
        window = []
        # 의도적으로 중간에 불량이 몰리게 셋팅된 가짜 배열
        fake_labels = [0]*10 + [1]*5 + [0,1,0,0,1] + [0]*10
        
        for val in fake_labels:
            window.append(val)
            if len(window) > 15: window.pop(0) # 15칸 표본 윈도우
            rate = sum(window) / len(window) * 100
            
            with chart_placeholder.container():
                st.line_chart(window, color="#ff7b7b" if rate >= 30 else "#5555ff")
                
            if rate >= 30:
                alert_placeholder.error(f"⚠️ [비상] 최근 15건 중 불량률 {rate:.1f}% 돌파! 라인 정지 및 즉각 확인 요망!")
            else:
                alert_placeholder.success(f"✅ 라인 스루풋 정상 유지 중 (불량률 {rate:.1f}%)")
            time.sleep(0.5)
            
        alert_placeholder.info("시뮬레이션이 종료되었습니다.")

# -----------------------------------------------------
# TAB 7: 🏢 ERP 연동 (Mock)
# -----------------------------------------------------
with tab_erp:
    st.subheader("🏢 ERP 데이터 연계 및 승인 (Mock)")
    st.markdown("현재 모델이 판별한 불량(Abnormal) 내역을 ERP/MES 시스템으로 전송하거나 추출합니다.")
    
    df_abnormal_erp = df[df["label"] == 1].copy()
    
    if df_abnormal_erp.empty:
        st.info("업데이트할 불량 항목이 없습니다.")
    else:
        # 체크박스(선택열) 추가
        df_abnormal_erp.insert(0, "Select", True)
        
        st.markdown("**[ 선택 전송 테이블 ]** 전체 전송이 아닌 확인된 항목만 골라서 전송할 수 있습니다.")
        edited_df = st.data_editor(
            df_abnormal_erp[["Select", "id", "confidence", "valid", "reason"]],
            column_config={
                "Select": st.column_config.CheckboxColumn("선택", default=True),
                "reason": st.column_config.TextColumn("판독 상세(Reason)", width="large")
            },
            disabled=["id", "confidence", "valid", "reason"],
            hide_index=True,
            use_container_width=True
        )
        
        selected_rows = edited_df[edited_df["Select"] == True]
        
        c_btn1, c_btn2 = st.columns([1, 5])
        with c_btn1:
            if st.button("📤 선택 항목 ERP로 전송"):
                if selected_rows.empty:
                    st.warning("선택된 항목이 없습니다!")
                else:
                    import datetime
                    now_str = datetime.datetime.now().strftime("%H:%M:%S")
                    with st.spinner("가상의 ERP 엔드포인트(http://mock-erp.local/api/sync)로 전송 중..."):
                        # 추출하여 payload 생성
                        payload = df_abnormal_erp[df_abnormal_erp["id"].isin(selected_rows["id"])].to_dict(orient="records")
                        
                        target_ids = selected_rows["id"].tolist()
                        sample_str = ", ".join(target_ids[:3]) + (" 등" if len(target_ids) > 3 else "")
                        
                        st.session_state["erp_sync_logs"].append(f"[{now_str}] ERP 전송 완료 (총 {len(target_ids)}건) - {sample_str}")
                        st.success(f"데이터 {len(payload)}건 전송 성공!")
                        
                        if st.checkbox("전송된 Payload JSON 요약 보기", key="show_erp_payload"):
                            st.json(payload[:2]) # Show up to 2 for brevity
                            if len(payload) > 2:
                                st.caption(f"...and {len(payload)-2} more items.")
        
        with c_btn2:
            export_df = df_abnormal_erp[df_abnormal_erp["id"].isin(selected_rows["id"])]
            csv_data = export_df.drop(columns=["Select"]).to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 선택된 데이터 CSV 백업",
                data=csv_data,
                file_name="erp_backup_abnormal.csv",
                mime="text/csv"
            )
            
        st.markdown("---")
        st.markdown("**동기화 로그 (Sync History)**")
        log_box = st.container(height=150)
        for log in reversed(st.session_state["erp_sync_logs"]):
            log_box.text(log)
