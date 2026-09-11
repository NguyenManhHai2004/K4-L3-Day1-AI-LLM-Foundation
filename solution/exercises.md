# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đế### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Khi temperature = 0.0, phản hồi mang tính xác định (deterministic) cao, luôn chọn từ có xác suất lớn nhất nên nội dung rất an toàn, chuẩn mực và ít thay đổi; khi tăng lên 0.5 – 1.0, câu trả lời trở nên linh hoạt, giàu chi tiết và tự nhiên hơn; nhưng khi lên 1.5, mô hình chọn cả các từ có xác suất thấp khiến câu văn lan man, lủng củng và tăng mạnh nguy cơ sinh thông tin sai lệch (hallucination).

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Nên đặt temperature ở mức thấp từ 0.0 đến 0.2 vì chatbot chăm sóc khách hàng cần ưu tiên tính chính xác, nhất quán và tuân thủ đúng dữ liệu/chính sách của doanh nghiệp, giúp loại bỏ tính ngẫu nhiên, tránh bịa đặt thông tin và đảm bảo hai khách hàng hỏi cùng một vấn đề sẽ luôn nhận được câu trả lời đồng nhất.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> GPT-4o ($0.010/1K token output) đắt hơn GPT-4o-mini ($0.0006/1K token) khoảng 16.67 lần (tương đương ~$105/ngày so với ~$6.3/ngày cho 10.5M token); GPT-4o xứng đáng với chi phí cho các bài toán phức tạp đòi hỏi suy luận logic nhiều bước hoặc phân tích hợp đồng pháp lý, còn mini nên dùng cho các tác vụ khối lượng lớn lặp lại như chatbot trả lời FAQ hay phân loại ý định người dùng.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Khác biệt ở chỗ persona giáo viên tiểu học dùng câu ngắn, từ ngữ giản dị và ẩn dụ trực quan dễ hiểu, trong khi persona chuyên gia tài chính dùng văn phong học thuật, cấu trúc chặt chẽ với nhiều thuật ngữ chuyên sâu; system prompt đóng vai trò như lệnh mẫu thiết lập khung tham chiếu toàn cục, giúp mô hình tự động điều chỉnh đối tượng mục tiêu, văn phong diễn đạt và độ sâu kiến thức mà không cần người dùng mô tả lại ở từng câu hỏi.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Số token thực tế từ tiktoken (khoảng 180–210 token cho 100 từ tiếng Việt) chênh lệch cao hơn từ 35% đến 55% so với công thức ước lượng thô (100 / 0.75 ≈ 133 token); nguyên nhân là do BPE tokenizer được huấn luyện chủ yếu trên tiếng Anh nên từ vựng tiếng Anh thường nguyên 1 token, trong khi tiếng Việt có nhiều nguyên âm ghép thanh điệu (UTF-8 đa byte) không có sẵn trong từ điển nên bị phân rã thành nhiều sub-words hoặc byte đơn lẻ (mất 1.5–2.5 token/từ).

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất trong các ứng dụng tương tác thời gian thực với người dùng cuối (như chatbot tư vấn, trợ lý viết code, hỏi đáp trực tiếp) khi câu trả lời dài để tối ưu chỉ số Time-To-First-Token (TTFT) giúp người dùng đọc ngay thay vì chờ màn hình tải; ngược lại, non-streaming phù hợp hơn cho các tác vụ xử lý theo lô (batch processing), gọi ngầm trong background, trích xuất dữ liệu có cấu trúc (JSON để parse sang code/DB), hoặc khi cần chạy bộ lọc kiểm duyệt an toàn trên toàn bộ văn bản hoàn chỉnh trước khi trả về.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> So với delay cố định, exponential backoff tăng dần thời gian chờ sau mỗi lần thử (0.1s -> 0.2s -> 0.4s...) giúp server đang quá tải có thời gian đệm giải phóng tài nguyên để tự phục hồi; nếu hàng nghìn client cùng dùng một mức delay cố định, tất cả sẽ đồng loạt gửi lại request cùng lúc sau mỗi chu kỳ 1 giây, gây ra hiện tượng bão retry (thundering herd / retry storms) tiếp tục đánh sập server thay vì giúp nó ổn định trở lại.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Persona: "Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt." — Lựa chọn "trợ giảng thân thiện" tạo cảm giác gần gũi, sẵn sàng hỗ trợ người học; yêu cầu "trả lời ngắn gọn" giúp tiết kiệm token, giảm độ trễ và tập trung đúng trọng tâm câu hỏi; còn chỉ định "bằng tiếng Việt" đảm bảo mô hình luôn phản hồi bằng ngôn ngữ mẹ đẻ của học viên ngay cả khi câu hỏi chứa thuật ngữ kỹ thuật tiếng Anh.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất là bộ nhớ chỉ lưu tối đa 3 lượt hội thoại gần nhất nên bot sẽ quên sạch ngữ cảnh ban đầu khi chat kéo dài; đề xuất cải thiện là cơ chế Tóm tắt bộ nhớ nghĩa là khi hội thoại vượt quá 3 lượt, trước khi cắt bỏ các message cũ, gọi một model phụ tóm tắt các thông tin cốt lõi thành 2–3 câu rồi lưu vào đầu context/system prompt làm bối cảnh dài hạn.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> **Hạn chế lớn nhất:** Trợ lý chỉ lưu giữ tối đa 3 lượt hội thoại gần nhất (sliding window 6 messages) và hoàn toàn không có bộ nhớ dài hạn (persistent memory). Khi cuộc trò chuyện kéo dài, bot sẽ quên các thông tin ngữ cảnh quan trọng đã nói từ đầu (như tên học viên, trình độ hiện tại, dự án đang làm dở).**Đề xuất cải thiện:** Triển khai cơ chế Tóm tắt bộ nhớ (Memory Summarization).**Cách triển khai:** Khi số lượt hội thoại chuẩn bị vượt quá ngưỡng 3 lượt, trước khi cắt bỏ các message cũ, hệ thống gửi các message đó đến một model nhỏ (như `gpt-4o-mini` hoặc `llama-3.1-8b`) kèm prompt: *"Hãy tóm tắt ngắn gọn các sự kiện và thông tin cá nhân quan trọng trong đoạn hội thoại này thành 2-3 câu"*. Nội dung tóm tắt này sẽ được lưu cố định vào đầu danh sách tin nhắn hoặc ghép vào system prompt làm "bối cảnh dài hạn", giúp bot vừa nhớ được thông tin xuyên suốt vừa không làm bùng nổ token.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
