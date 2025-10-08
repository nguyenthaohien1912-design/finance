import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Báo Cáo Tài Chính",
    layout="wide"
)

st.title("Ứng dụng Phân Tích Báo Cáo Tài Chính 📊")

# --- Khởi tạo Session State cho Chat và Dữ liệu Context ---
# Lịch sử tin nhắn cho khung chat
if "messages" not in st.session_state:
    st.session_state.messages = []
# Dữ liệu phân tích (dạng markdown string) làm bối cảnh cho AI chat
if "analysis_data_context" not in st.session_state:
    st.session_state.analysis_data_context = ""
# Client Chat API để duy trì lịch sử hội thoại
if "chat_client" not in st.session_state:
    st.session_state.chat_client = None

# --- Hàm tính toán chính (Sử dụng Caching để Tối ưu hiệu suất) ---
@st.cache_data
def process_financial_data(df):
    """Thực hiện các phép tính Tăng trưởng và Tỷ trọng."""
    
    # Đảm bảo các giá trị là số để tính toán
    numeric_cols = ['Năm trước', 'Năm sau']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 1. Tính Tốc độ Tăng trưởng
    df['Tốc độ tăng trưởng (%)'] = (
        (df['Năm sau'] - df['Năm trước']) / df['Năm trước'].replace(0, 1e-9)
    ) * 100

    # 2. Tính Tỷ trọng theo Tổng Tài sản
    tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]
    
    if tong_tai_san_row.empty:
        raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN'.")

    tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0]
    tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0]

    # Xử lý giá trị 0 cho mẫu số
    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    # Tính tỷ trọng với mẫu số đã được xử lý
    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100
    
    return df

# --- Hàm gọi API Gemini cho Nhận xét Tóm tắt Ban đầu (Chức năng 5) ---
def get_ai_analysis(data_for_ai, api_key):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash' 

        prompt = f"""
        Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Dựa trên các chỉ số tài chính sau, hãy đưa ra một nhận xét khách quan, ngắn gọn (khoảng 3-4 đoạn) về tình hình tài chính của doanh nghiệp. Đánh giá tập trung vào tốc độ tăng trưởng, thay đổi cơ cấu tài sản và khả năng thanh toán hiện hành.
        
        Dữ liệu thô và chỉ số:
        {data_for_ai}
        """

        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except KeyError:
        return "Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY'. Vui lòng kiểm tra cấu hình Secrets trên Streamlit Cloud."
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"


# --- Chia giao diện thành 2 tab chính ---
analysis_tab, chat_tab = st.tabs(["📊 Phân Tích Dữ Liệu", "💬 Hỏi Đáp với AI"])


with analysis_tab:
    # --- Chức năng 1: Tải File ---
    uploaded_file = st.file_uploader(
        "1. Tải file Excel Báo cáo Tài chính (Chỉ tiêu | Năm trước | Năm sau)",
        type=['xlsx', 'xls']
    )

    if uploaded_file is not None:
        try:
            df_raw = pd.read_excel(uploaded_file)
            
            # Tiền xử lý: Đảm bảo chỉ có 3 cột quan trọng
            df_raw.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau']
            
            # Xử lý dữ liệu
            df_processed = process_financial_data(df_raw.copy())

            if df_processed is not None:
                
                # --- Chức năng 2 & 3: Hiển thị Kết quả ---
                st.subheader("2. Tốc độ Tăng trưởng & 3. Tỷ trọng Cơ cấu Tài sản")
                st.dataframe(df_processed.style.format({
                    'Năm trước': '{:,.0f}',
                    'Năm sau': '{:,.0f}',
                    'Tốc độ tăng trưởng (%)': '{:.2f}%',
                    'Tỷ trọng Năm trước (%)': '{:.2f}%',
                    'Tỷ trọng Năm sau (%)': '{:.2f}%'
                }), use_container_width=True)
                
                # --- Chức năng 4: Tính Chỉ số Tài chính ---
                st.subheader("4. Các Chỉ số Tài chính Cơ bản")
                
                thanh_toan_hien_hanh_N = "N/A"
                thanh_toan_hien_hanh_N_1 = "N/A"

                try:
                    # Lấy Tài sản ngắn hạn
                    tsnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                    tsnh_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                    # Lấy Nợ ngắn hạn
                    no_ngan_han_N = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]  
                    no_ngan_han_N_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                    # Tính toán, kiểm tra chia cho 0
                    if no_ngan_han_N != 0:
                        thanh_toan_hien_hanh_N = tsnh_n / no_ngan_han_N
                    if no_ngan_han_N_1 != 0:
                        thanh_toan_hien_hanh_N_1 = tsnh_n_1 / no_ngan_hanh_N_1
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            label="Chỉ số Thanh toán Hiện hành (Năm trước)",
                            value=f"{thanh_toan_hien_hanh_N_1:.2f} lần" if isinstance(thanh_toan_hien_hanh_N_1, (int, float)) else thanh_toan_hien_hanh_N_1
                        )
                    with col2:
                        # Tính delta chỉ khi cả 2 giá trị đều là số
                        delta_val = None
                        if isinstance(thanh_toan_hien_hanh_N, (int, float)) and isinstance(thanh_toan_hien_hanh_N_1, (int, float)):
                            delta_val = f"{thanh_toan_hien_hanh_N - thanh_toan_hien_hanh_N_1:.2f}"
                        
                        st.metric(
                            label="Chỉ số Thanh toán Hiện hành (Năm sau)",
                            value=f"{thanh_toan_hien_hanh_N:.2f} lần" if isinstance(thanh_toan_hien_hanh_N, (int, float)) else thanh_toan_hien_hanh_N,
                            delta=delta_val
                        )
                        
                except IndexError:
                    st.warning("Thiếu chỉ tiêu 'TÀI SẢN NGẮN HẠN' hoặc 'NỢ NGẮN HẠN' để tính chỉ số.")
                except ZeroDivisionError:
                    st.warning("Không thể tính chỉ số Thanh toán Hiện hành do Nợ Ngắn Hạn bằng 0.")

                # --- Cập nhật Context Dữ liệu cho Chat ---
                # Lưu toàn bộ bảng phân tích vào session state để làm context cho khung chat
                st.session_state.analysis_data_context = df_processed.to_markdown(index=False)
                
                # --- Chức năng 5: Nhận xét AI Tóm tắt Ban đầu ---
                st.subheader("5. Nhận xét Tình hình Tài chính (AI Tóm tắt)")
                
                # Chuẩn bị dữ liệu đầy đủ để gửi cho AI
                data_for_ai = pd.DataFrame({
                    'Chỉ tiêu': [
                        'Toàn bộ Bảng phân tích', 
                        'Thanh toán hiện hành (N-1)', 
                        'Thanh toán hiện hành (N)'
                    ],
                    'Giá trị': [
                        st.session_state.analysis_data_context, # Sử dụng dữ liệu đã lưu
                        f"{thanh_toan_hien_hanh_N_1}" if isinstance(thanh_toan_hien_hanh_N_1, (int, float)) else thanh_toan_hien_hanh_N_1, 
                        f"{thanh_toan_hien_hanh_N}" if isinstance(thanh_toan_hien_hanh_N, (int, float)) else thanh_toan_hien_hanh_N
                    ]
                }).to_markdown(index=False) 

                if st.button("Yêu cầu AI Tóm tắt và Nhận xét (Tạo mới)"):
                    api_key = st.secrets.get("GEMINI_API_KEY") 
                    
                    if api_key:
                        with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                            ai_result = get_ai_analysis(data_for_ai, api_key)
                            st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                            st.info(ai_result)
                    else:
                        st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")

        except ValueError as ve:
            st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
        except Exception as e:
            st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file.")

    else:
        st.info("Vui lòng tải lên file Excel để bắt đầu phân tích.")


with chat_tab:
    st.header("💬 Trợ Lý Tài Chính AI - Hỏi Đáp")
    api_key = st.secrets.get("GEMINI_API_KEY")

    if not api_key:
        st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")
    else:
        # --- Khởi tạo Chat Client: KIỂM TRA VÀ TẠO LẠI NẾU CẦN ---
        # Kiểm tra: Nếu chat_client chưa được khởi tạo hoặc đã bị đánh dấu lỗi (False)
        if st.session_state.chat_client is None or st.session_state.chat_client is False:
            try:
                client = genai.Client(api_key=api_key)
                
                # System Instruction cho Chat
                system_instruction = (
                    "Bạn là một Trợ lý Tài chính AI chuyên nghiệp. Nhiệm vụ của bạn là phân tích và trả lời các câu hỏi về dữ liệu tài chính "
                    "đã được cung cấp. Luôn sử dụng dữ liệu trong bối cảnh hiện tại để trả lời. Câu trả lời phải bằng Tiếng Việt và chuyên nghiệp."
                )
                
                # Sử dụng 'config' để truyền System Instruction
                config = {"system_instruction": system_instruction}
                
                # Khởi tạo chat client và lưu vào session state
                st.session_state.chat_client = client.chats.create(
                    model="gemini-2.5-flash",
                    config=config 
                )
                st.info("Chat Client đã được khởi tạo thành công.")
            except Exception as e:
                # Nếu có lỗi, đánh dấu chat_client là False để hiển thị cảnh báo
                st.error(f"Không thể khởi tạo Chat Client: {e}")
                st.session_state.chat_client = False 

        if st.session_state.chat_client:
            # 1. Hiển thị lịch sử tin nhắn
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # 2. Xử lý Input của người dùng
            if prompt := st.chat_input("Hỏi về báo cáo tài chính của bạn..."):
                
                # Thêm tin nhắn người dùng vào lịch sử
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Thêm bối cảnh dữ liệu vào đầu prompt nếu có
                context = st.session_state.analysis_data_context
                
                # Nếu có context (dữ liệu đã được tải lên và xử lý)
                if context:
                    # Gắn bối cảnh dữ liệu vào prompt để AI trả lời chính xác
                    context_prompt = (
                        f"Dữ liệu tài chính hiện tại để phân tích là: \n\n{context}\n\n"
                        f"Dựa vào dữ liệu này và lịch sử hội thoại, trả lời câu hỏi sau: {prompt}"
                    )
                else:
                    context_prompt = prompt

                # Gửi yêu cầu đến AI
                with st.chat_message("assistant"):
                    with st.spinner("Đang nghĩ..."):
                        try:
                            # Gửi prompt (có thể kèm context) đến chat client
                            # Sửa lỗi: Đảm bảo chat_client còn hoạt động trước khi gửi
                            if st.session_state.chat_client: 
                                response = st.session_state.chat_client.send_message(context_prompt)
                                st.markdown(response.text)
                                st.session_state.messages.append({"role": "assistant", "content": response.text})
                            else:
                                raise Exception("Chat client đã bị đóng.")

                        except APIError as e:
                            error_msg = f"Lỗi gọi Gemini Chat API: {e}. Vui lòng kiểm tra Khóa API hoặc thử lại."
                            st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
                        except Exception as e:
                            # Bắt lỗi "Cannot send a request, as the client has been closed."
                            error_msg = f"Lỗi không xác định khi gọi AI: {e}. Có thể chat client đã bị đóng, vui lòng thử lại."
                            st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
                            # Đặt lại chat_client để nó được khởi tạo lại ở lần chạy tiếp theo
                            st.session_state.chat_client = None 

            # Thông báo nếu chưa có dữ liệu và chưa có tin nhắn
            if not st.session_state.analysis_data_context and not st.session_state.messages:
                    st.info("Vui lòng tải file và phân tích dữ liệu ở tab **'Phân Tích Dữ Liệu'** trước. Khi có dữ liệu, AI sẽ sử dụng nó làm bối cảnh.")
        elif st.session_state.chat_client is False:
             st.warning("Chatbot không khả dụng do lỗi khởi tạo Client API.")
