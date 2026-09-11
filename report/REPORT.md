# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):  "class_id": 468, "class_name": "cab", "rank": 1, "score": 0.510915, "taxonomy_name": "ImageNet-1K".
- Record này mô tả toàn ảnh như thế nào?  Record naỳ mô tả toàn ảnh có khả năng thuộc lớp 'Cab' cao nhất với 'score' ₫ 0,510915.
- Ai định nghĩa class list mà checkpoint có thể dự đoán? Dataset/taxonomy dùng để huấn luyện checkpoint định nghĩa class list mà checkpoint có thể dự đoán.
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy? ID là mã số định danh cho mỗi lớp hữu ích cho việc xử lý dữ liệu và tương thích với các hệ thống khác nhau. Class Name cung cấp tên dễ đọc, dễ hiểu cho con người, giúp diễn giải kết quả và làm việc với các nhãn một cách trực quan. Taxonomy name chỉ định nguồn gốc của danh sách lớp (ví dụ: ImageNet-1K, COCO-80).
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì? Nếu ảnh có nhiều chủ thể thì guideline cần quy định chọn ra cách phân loại nhãn cho toàn bộ ảnh.
- Vì sao model score không phải ground truth? model score là độ tin tưởng của mô hình về dư đoán của nó. Còn ground truth là nhãn đúng do con người gán dựa trên sự hiểu biết của chuyên gia.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
 "class_name": "person", "score": 0.890623,"bbox_xyxy": [
    394.04,
    144.97,
    466.8,
    339.73
  ],  "bbox_width": 72.76, "bbox_height": 194.76.
- Diễn giải vị trí box bằng lời: 
Box có vị trí bên phải của hình ảnh với xmin là 394,04 trên tổng chiều rộng là 640,  ymin là 144,97 trên tổng chiều cao là 428
- So sánh số prediction ở hai threshold: 
Với threshold=0.20 thì mô hình phát hiện được 17 vật thể còn với threshold=0.60 thì mô hình phát hiện được 6 vật thể điều này cho thấy khi tăng threshold thì số lượng prediction ở các score thấp hơn sẽ bị loại nên số lượng prediction sẽ giảm. Khi ngưỡng tin cậy (threshold) giảm xuống thì số lượng prediction tăng lên cho thấy mô hình sẽ chấp nhận cả những dự đoán mà nó ít tự tin hơn.
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem? 
Khi ngưỡng Threshold giảm xuống thì mô hình có khẳ năng phát hiện được nhiều vật thể hơn bao gồm các đối tượng có độ tim cậy thấp. Vì vậy khi giảm ngưỡng threshold sẽ làm các dự đoán dễ bị sai hơn nên khối lượng công việc của reviewer cũng sẽ tăng lên. Ngược lại nếu tăng Threshold thì độ chính xác cao hơn nhưng mô hình sẽ bỏ sót các vật thể.
- Đề xuất một quy tắc box chặt: 
Box cần phải được bao sát toàn bộ vật thể, không nên lấy quá nhiều phần nền bao quanh
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định? 
Với các object bị cắt  hoặc che khuất thì guideline cần quy định rõ cách xử lý đối. Có thể chỉ bao quanh phần vật nhìn thấy hoặc ước tính.   

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`): 
"instance_id": "kitchen-001", "class_name": "person", "score": 0.890623, "polygon_point_count": 120, 
  "polygon_xy": [
    [394.0, 150.0], [394.0, 151.0], [395.0, 152.0], ... 
  ]
- Polygon bổ sung chi tiết gì so với box? 
Polygon sẽ cung cấp chi tiết về đường biên và hình dạng của vật thể hơn so với box (là hình chữ nhật bao quanh đối tượng và có thể bao gồm cả phần nên không phải là đối tượng). Polygon giúp chính xác hơn trong các đối tượng có hình dạng đặc biệt.
- `instance_id` dùng để làm gì và không phải loại ID nào? 
'instance_id' dùng để định danh đối tượng cụ thể trong hình ảnh, giúp phân biệt nhiều đồi tượng cùng một loại class. instance_id không phải là ID của loại đối tượng hay id của ảnh.
- Đề xuất một quy tắc biên mask:
Mask cần phải bám sát phần nhìn thấy của vật thể và hạn chế tối đa phần nền không phải vật thể. 
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
Guideline cần quy định cách xử lý vùng mờ, vùng tiếp xúc giữa các vật thể và phần bị che khuất; nếu không thể xác định rõ ranh giới thì cần đánh dấu không chắc chắn hoặc escalation để reviewer quyết định.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh |Class ID/ Class Name  | Ảnh quá phức tạp/mờ. Score cao nhưng lớp không phù hợp.  | Gán nhãn đúng nhất cho toàn bộ ảnh theo guideline. Chọn nhãn đại diện theo quy tắc (đối tượng lớn nhất/chính). Gán nhãn 'không xác định' hoặc bỏ qua nếu không phù hợp. |Tính chính xác của nhãn so với nội dung ảnh. Tính nhất quán với guideline (ảnh nhiều đối tượng, mơ hồ). Không có lỗi chính tả/sai ID lớp.  |
| Phát hiện vật thể |Box (xyxy): [x_min, y_min, x_max, y_max]; Class ID ; Class Name | Hộp sai vị trí/kích thước (quá rộng/hẹp, cắt đối tượng, bao gồm nền). Sai lớp cho hộp. Hộp cho nền (false positive). Vật thể bị che khuất/cắt mép.  | Vẽ hộp cho tất cả đối tượng thuộc các lớp. Đảm bảo hộp ôm sát, không quá rộng/cắt. Gán đúng lớp cho hộp. Xử lý che khuất/cắt mép theo guideline  | Tất cả đối tượng được gán nhãn (không thiếu). Vị trí/kích thước hộp chính xác (độ chặt). Nhãn lớp chính xác. Không có hộp giả/trùng lặp. Tuân thủ guideline che khuất/cắt mép. |
| Instance segmentation |Polygon (danh sách các điểm [[x1, y1], ..., [xn, yn]] tính bằng pixel); Instance ID (chuỗi duy nhất); Class ID (số nguyên); Class Name (chuỗi)  |Thiếu mask (false negative). Mask trùng lặp. Mask không theo sát biên dạng (thô, bao nền, cắt đối tượng). Sai lớp cho mask. Mask cho nền (false positive). Vùng biên mờ, đối tượng tiếp xúc, che khuất.| Vẽ mask dạng polygon cho tất cả đối tượng. Đảm bảo mask theo sát biên dạng ở cấp độ pixel. Gán đúng lớp và instance_id duy nhất. Xử lý vùng mờ/tiếp xúc/che khuất theo guideline. | Tất cả đối tượng được gán mask. Hình dạng mask chính xác (độ chặt pixel). Nhãn lớp và instance_id chính xác/duy nhất. Không có mask giả. Tuân thủ guideline cho trường hợp đặc biệt (mờ, tiếp xúc, che khuất). |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu: Không chia sẻ, tải xuống hoặc sử dụng dữ liệu/ảnh chứa thông tin cá nhân hoặc dữ liệu nhạy cảm ngoài phạm vi được phép.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho: giảng viên/người phụ trách khóa học hoặc người quản lý dữ liệu.

## 6. Danh sách bằng chứng

- [x ] `classification_predictions.json`
- [x ] `detection_predictions.json`
- [x ] `segmentation_predictions.json`
- [x ] `IMAGE_ATTRIBUTION.md`
- [x ] `visuals/classification_top5.png`
- [x ] `visuals/detection_predictions.png`
- [x ] `visuals/segmentation_prediction.png`
- [x ] Ô validation cuối notebook báo `PASS`.
- [x ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
