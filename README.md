# Legal Entity NER

[Mã nguồn trên GitHub](https://github.com/lehthai22072005-collab/Legal_Entity_NER) · [Tải ZIP kèm checkpoint](https://github.com/lehthai22072005-collab/Legal_Entity_NER/archive/refs/heads/main.zip)

Trích xuất thực thể từ DOCX bằng PhoBERT-base kết hợp nhúng ký tự, tích chập theo từng kênh, đầu nhận dạng biên, đầu phân loại đoạn và Learned Marker Adapter. Bộ chọn đoạn sử dụng chính sách xây dựng từ Train. Ontology có 40 loại thực thể; mô hình hiện tại không có đầu trích xuất quan hệ. Nhãn chưa có mẫu Train không được coi là đã học được.

## Chạy giao diện web bằng Docker

Máy nhận cần cài Docker Engine kèm Docker Compose, hoặc Docker Desktop. Bản đóng gói dành cho Linux amd64, chạy suy luận bằng CPU; Docker Desktop trên Windows/macOS có thể chạy Linux containers. Lần dựng image đầu cần Internet để tải thư viện. Image đã dựng chứa checkpoint epoch 11 và tokenizer, nên lúc suy luận không tải mô hình từ Hugging Face.

Từ thư mục dự án đầy đủ (đã có `checkpoints/phobert_span_all369_v1/model.pt`):

```bash
docker compose up --build -d
```

Mở **http://localhost:8501**, chọn checkpoint, tải DOCX, nhấn **Trích xuất**, xem kết quả và tải JSON. Không cần cài Python, `.venv` hoặc CUDA trên máy chủ. Dừng bằng:

```bash
docker compose down
```

Cổng mặc định chỉ mở trên máy đang chạy. Nếu cổng 8501 đã được dùng, đặt `NER_PORT=8502` trong môi trường khi chạy Compose. Giới hạn RAM mặc định của container là 3 GiB, có thể đổi qua `NER_MEMORY_LIMIT` và `NER_MEMORY_SWAP_LIMIT`. Nên có khoảng 8 GB RAM để chạy demo cùng trình duyệt; văn bản dài cần nhiều thời gian và bộ nhớ hơn. Không có phép đo bảo đảm tốc độ cho mọi DOCX.

### Dùng image đã đóng gói, không dựng lại

Với tệp bàn giao `Legal_Entity_NER_Docker_Epoch11.tar.gz`, chạy:

```bash
docker load -i Legal_Entity_NER_Docker_Epoch11.tar.gz
docker run --rm -d --name legal-ner -p 127.0.0.1:8501:8501 legal-entity-ner:epoch11
```

Mở cùng địa chỉ ở trên. Dừng bằng `docker stop legal-ner`. Image này đã kèm trọng số, thư viện và mã chạy; không cần tải dataset để suy luận.

### Tải dự án từ GitHub rồi chạy Docker

Trọng số `model.pt` phải đi cùng mã nguồn. `.gitattributes` đã khai báo checkpoint epoch 11 dùng Git LFS và `.gitignore` cho phép theo dõi đúng checkpoint này. Tệp hơn 100 MiB phải được đẩy qua Git LFS, không commit như tệp Git thông thường.

**Người đăng dự án thực hiện một lần:** cài Git LFS, rồi trong repository đã khởi tạo chạy `git lfs install` trước khi `git add` và commit checkpoint. Cần push cả tệp `.gitattributes`, mã nguồn và đối tượng LFS thực tế; chỉ thêm tệp cấu hình ở local chưa đưa trọng số lên GitHub.

Để người dùng chọn **Code → Download ZIP**, giải nén rồi chỉ cần Docker, vào **Settings → General → Archives → Include Git LFS objects in archives** trên GitHub và bật tùy chọn này. Mặc định ZIP của GitHub chỉ chứa con trỏ LFS; phải bật tùy chọn trên sau khi checkpoint đã được push thành công.

Khi ZIP đã chứa đủ trọng số, người nhận giải nén, mở Terminal tại thư mục có `compose.yaml`, rồi chạy:

```bash
docker compose up --build -d
```

Nếu dùng `git clone`, máy nhận cần Git LFS hoặc GitHub Desktop để lấy cả trọng số. Với Git CLI, chạy `git lfs pull` trong repository sau khi clone, rồi chạy lệnh Docker ở trên. Không cần môi trường Python trên máy nhận.

Tệp `Legal_Entity_NER_Docker_Epoch11.tar.gz` nằm ngoài source là một cách chạy khác: image đã dựng sẵn để `docker load`, không bắt buộc có trong repository khi dựng từ source đầy đủ. Có thể cung cấp tệp này qua GitHub Releases.

Nguồn: [Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage), [đưa trọng số LFS vào ZIP tải từ GitHub](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/managing-repository-settings/managing-git-lfs-objects-in-archives-of-your-repository).

Docker build kiểm SHA-256 của checkpoint với `ACCEPTANCE_SNAPSHOT.json`; tệp thiếu, tệp con trỏ Git LFS hoặc trọng số khác sẽ làm build dừng. Dataset, bản xuất nhãn, notebook, lịch sử phiên, `.venv` và thông tin xác thực không được chép vào image. Bản Docker phục vụ suy luận, không tự huấn luyện lại.

Tham khảo: [triển khai Streamlit bằng Docker](https://docs.streamlit.io/deploy/tutorials/docker), [giới hạn tệp lớn của GitHub](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

### Kết quả kiểm tra bản Docker

Đã dựng image thành công, kiểm tra dependency, đối chiếu SHA-256 checkpoint epoch 11, chạy suy luận hai DOCX khi container không có mạng, và thực hiện upload → trích xuất → tải JSON bằng Firefox. JSON tải từ web khớp kết quả CLI. Tệp image nén đã qua `gzip -t` và nạp lại bằng `docker load`, giữ nguyên image ID. Đây là kiểm tra chức năng; không phải đánh giá F1 mới hoặc xác nhận ngữ nghĩa mọi nhãn. Trọng số và bộ dữ liệu không thay đổi.

## Dữ liệu

Bộ hiện hành có 369 văn bản riêng biệt: 314 Train và 55 Validation, chia theo nhóm nguồn với seed 20261117. Mọi cửa sổ của một văn bản thuộc cùng tập. Chưa có Test độc lập.

```text
data/
├── data_tho/       # Nguồn JSON/CSV và DOC, tổ chức theo 7 nhóm hình thức
├── dataset/
│   ├── train.json  # 314 bản ghi, mảng JSON
│   └── val.json    # 55 bản ghi, mảng JSON
└── source_docx/    # 369 DOCX: 367 bản gốc và 2 bản chuyển từ DOC
```

Khi chuẩn bị snapshot huấn luyện, 217 văn bản có DOCX và 152 bản lấy phần thân Word từ bản xuất nhãn. Sau đó đã bổ sung nguồn cho 152 bản, trong đó 146 khớp nguyên chuỗi và 6 chỉ khác khoảng trắng/xuống dòng. Các bản ghi và split huấn luyện được giữ nguyên. Phần thông tin bổ sung ngoài nội dung Word bị loại khỏi đầu vào và mục tiêu. Nhãn do người dùng cung cấp và kiểm tra không đồng nghĩa với Gold được đánh giá độc lập.

Ngày 20/11/2026, theo yêu cầu dọn nguồn, đã xóa 14 bản xuất trùng và hai bản xuất bị loại, cùng các CSV tương ứng. `data/data_tho/` còn 369 JSON và 369 CSV phục vụ bộ dữ liệu hiện hành. Số lượng 385 bản xuất trong hồ sơ chuẩn bị là số liệu trước khi dọn; Train, Validation, DOCX nguồn và checkpoint giữ nguyên.

`scripts/check_all_data.py` kiểm tra nguồn, hash, loại nhãn, offset và tách nhóm; kiểm tra kỹ thuật không xác nhận mọi nhãn đều đúng ngữ nghĩa.

## Lượt huấn luyện được sử dụng

Đã hoàn tất 11 epoch trên Tesla T4 và chốt checkpoint epoch 11. Cấu hình tối đa là 12 epoch, nhưng runtime bị ngắt trong epoch 12; không báo đã hoàn tất epoch đó. Hiện không có phiên huấn luyện hoặc watcher đang chạy.

Checkpoint ở `checkpoints/phobert_span_all369_v1/` gồm:

- `model.pt`: trọng số đã tinh chỉnh.
- `config.json`: kiến trúc và cấu hình.
- `training_state.json`: epoch, ngưỡng và giới hạn suy luận.
- `decoder_policy.json`: chính sách chọn đoạn từ Train.

Kết quả exact-span trên 55 Validation của epoch 11:

| Precision | Recall | Micro-F1 | Macro-F1 trên 21 nhãn có mẫu |
|---|---|---|---|
| 83,28% | 89,30% | 86,19% | 53,81% |

TP = 6.680, FP = 1.341, FN = 800; ngưỡng biên và đoạn đều 0,5. Nguồn số liệu là `reports/phobert_span_all369_v1/history.json` và bản tổng hợp `ACCEPTANCE_SNAPSHOT.json`. Chưa có bộ kết quả Train, Test độc lập hoặc kết quả tìm ngưỡng sau phục hồi để công bố. Không sử dụng điểm của những nhãn vắng mặt để nâng Macro-F1.

## Colab

```text
colab/
├── Train_PhoBERT_NER_All369.ipynb
├── Inputs_All369.zip
├── INPUT_MANIFEST.json
└── COLAB_HANDOFF.json  # Chỉ lưu trên máy làm việc, không đăng lên GitHub
```

Notebook giữ nguyên nội dung của bản Recovery thực sự dùng cho lượt gần nhất, chỉ đổi về tên ngắn ở trên. Gói đầu vào được giữ nguyên SHA-256 và chứa snapshot mã nguồn/dữ liệu đã dùng cho thí nghiệm. `INPUT_MANIFEST.json` ghi hash notebook hiện hành và hash bản đóng gói ban đầu. `COLAB_HANDOFF.json` chỉ lưu trên máy làm việc, không có trong bản GitHub; tệp này giữ liên kết phiên Colab đã kết thúc. Mã nguồn và dữ liệu để đối chiếu lượt huấn luyện nằm trong notebook, Inputs_All369.zip và INPUT_MANIFEST.json.

Các notebook nhập lại/đánh giá phục hồi chưa hoàn tất, log trình duyệt và ZIP checkpoint trùng đã được xóa. Trước khi xóa ZIP, đã đối chiếu đủ 7 tệp bên trong với các bản đã giải nén. Trọng số hiện hành vẫn ở `checkpoints/`. Các script theo dõi/truyền tệp chỉ phục vụ phiên Colab mới được thiết lập đầy đủ; không chạy lại chúng để tiếp tục phiên đã kết thúc.

Khi cần một thí nghiệm mới, dùng `scripts/build_colab_all.py --output-dir <thu_muc_moi>` để tránh ghi đè gói frozen. Notebook giữ cấu hình tối đa 12 epoch của thí nghiệm gốc; không được hiểu là lượt đã báo cáo hoàn tất 12 epoch.

## Báo cáo được giữ

`reports/phobert_span_all369_v1/` chứa năm tệp của lượt gần nhất:

- `history.json`: kết quả từng epoch từ 1 đến 11.
- `EXPERIMENT.json`: siêu tham số, phần cứng và hash mã nguồn.
- `PRETRAINING_ORIGIN.json`: nguồn trọng số PhoBERT ban đầu.
- `ACCEPTANCE_SNAPSHOT.json`: kết quả checkpoint đã chốt.
- `RUNTIME_INTERRUPTION.json`: bằng chứng gián đoạn và checkpoint được phục hồi; đây là ghi nhận lịch sử, xem trạng thái chốt ở ACCEPTANCE_SNAPSHOT.

Năm tệp ở gốc `reports/` mô tả dữ liệu của chính lượt này: `DATA_AUDIT.json`, `DUPLICATES.json`, `EXCLUDED_DOCUMENTS.json`, `LABEL_REVIEW.json` và `OFFSET_PROJECTIONS.json`. Đây là hồ sơ chuẩn bị nguồn, không phải kết quả huấn luyện. Các log chuẩn bị Colab và preflight phiên bản cũ đã xóa.

## Chạy trên máy

Đứng trong thư mục `Legal_Entity_NER`, dùng môi trường `.venv/` hoặc cài thư viện từ `requirements.txt`.

```bash
.venv/bin/python scripts/check_all_data.py
.venv/bin/python scripts/predict.py /duong/dan/vanban.docx --checkpoint checkpoints/phobert_span_all369_v1 --output ket_qua.json
.venv/bin/python -m streamlit run app.py
```

Đầu vào suy luận chỉ cần DOCX. Mọi đoạn kết quả gắn với tọa độ Python Unicode `[start,end)` trên chuỗi chuẩn hóa. Thuộc tính không có đoạn được chấp nhận mang giá trị null; null có thể phản ánh thông tin không có trong nguồn hoặc mô hình bỏ sót. Hệ thống không tra metadata website/Neo4j để điền đầu ra.

`src/` chứa các mô-đun cốt lõi; `scripts/` cung cấp các lệnh huấn luyện, đánh giá, kiểm tra và suy luận; `app.py` là giao diện Streamlit. Không có thư mục `tests/` trong bản bàn giao hiện hành theo yêu cầu trước đó. Những con số kiểm thử kỹ thuật của snapshot cũ không phải chỉ số F1.
