# Báo cáo Lab Day 1 — Văn Thành Huy — 2A202602763

## 1\. Thiết lập

Forest CoverType, 54 đặc trưng và 7 lớp. Chia dữ liệu theo metadata cố định: train gốc 464809, eval 116203. Tách validation phân tầng với seed 42: train 371847, val 92962. Chuẩn hoá 10 cột số bằng thống kê train; giữ nguyên 44 cột nhị phân. Mean sau chuẩn hoá gần 0, độ lệch chuẩn gần 1. Accuracy đoán lớp đa số trên validation là 0,4876.

M-base: 54→256→128→7, ReLU, 47879 tham số, logits chưa softmax. Baseline: CE, SGD momentum 0.9, lr=0.1, batch 512, 20 epoch, He, FP32, không dropout/clipping. Learning rate chọn từ các pilot 5 epoch trên validation: lr=0,01 cho macro-F1 0,6420 (`lr-search-0.01`), lr=0,03 cho 0,7279 (`lr-search-0.03`), lr=0,1 cho 0,7756 (`lr-search-0.1`). Mọi checkpoint trong các thí nghiệm chính được chọn bằng validation loss thấp nhất, không phải macro-F1 cao nhất.

## 2\. Kiểm tra ban đầu và độ nhiễu

Loss bước 0: 2.2691, so với ln(7)=1.9459. He không đảm bảo dự đoán đều cho các lớp. Gradient của mọi tham số khác 0. Học thuộc 20 mẫu sau 11 bước: loss=0.008686, accuracy=100.0%.

!\[](figures/health\_overfit20.png)

|exp\_id|val\_acc|val\_macro\_f1|best\_epoch|
|-|-|-|-|
|base-s1|0.9069|0.8410|18|
|base-s2|0.9113|0.8639|20|
|base-s3|0.9109|0.8579|20|

Baseline macro-F1: 0.8543 ± 0.0119; mốc tham khảo 2σ=0.0238. Chỉ 3 seed nên đây là ước lượng thô, không phải kiểm định thống kê. Với `base-s1`, train/val loss giảm từ 0,4585/0,4634 ở epoch 1 xuống 0,2153/0,2365 ở epoch 20. Val loss thấp nhất ở epoch 18; chọn checkpoint này thay vì epoch cuối. Accuracy validation của cả ba seed đều vượt xa mốc đoán đa số 0,4876.

## 3\. Thí nghiệm

### loss

**Dự đoán trước khi chạy:** MSE có thể học chậm hơn CE với cùng learning rate và số epoch.

|exp\_id|val\_macro\_f1|delta\_vs\_base\_s1|seconds\_per\_epoch|
|-|-|-|-|
|loss-mse|0.7443|-0.0966|1.4075|

Kết quả phù hợp dự đoán: MSE đạt macro-F1 0,7443, thấp hơn CE ở `base-s1` 0,0966 với cùng lr=0,1, batch 512 và 20 epoch. Chênh lệch lớn hơn mốc nhiễu tham khảo 0,0238, nhưng MSE chưa được tune learning rate riêng. CE nhận logit trực tiếp; MSE qua softmax có gradient khác và có thể suy giảm khi xác suất bão hoà. Không so độ lớn loss MSE với loss CE; kết luận ở đây dựa trên macro-F1.

!\[](figures/compare\_loss.png)

### optimizer

**Dự đoán trước khi chạy:** Adam lr=0.001 có thể hội tụ nhanh hơn SGD momentum; đây là so sánh cấu hình, chưa phải so sánh optimizer đã tune công bằng.

|exp\_id|val\_macro\_f1|delta\_vs\_base\_s1|seconds\_per\_epoch|
|-|-|-|-|
|opt-adam-lr1e-3|0.8476|0.0067|1.4620|

Adam lr=0,001 đạt macro-F1 0,8476, hơn `base-s1` 0,0067; chênh lệch nhỏ hơn mốc 0,0238 nên chưa có bằng chứng đủ mạnh về ưu thế. Adam điều chỉnh bước cập nhật theo các moment của gradient. Thí nghiệm đồng thời đổi optimizer và learning rate; chưa tune hai optimizer với ngân sách ngang nhau, nên không kết luận Adam thắng tổng quát hoặc hội tụ nhanh hơn chỉ từ điểm cuối.

!\[](figures/compare\_optimizer.png)

### hparam

**Dự đoán trước khi chạy:** Batch 1024 có thể giảm thời gian mỗi epoch, nhưng ít bước cập nhật hơn có thể làm điểm validation thấp hơn.

|exp\_id|val\_macro\_f1|delta\_vs\_base\_s1|seconds\_per\_epoch|
|-|-|-|-|
|hparam-batch1024|0.8166|-0.0244|0.6969|

Batch 1024 giảm thời gian/epoch từ 1,2745 giây (`base-s1`) xuống 0,6969 giây, khoảng 45,3%, nhưng macro-F1 giảm 0,0244. Với 371847 mẫu, mỗi epoch có 727 bước ở batch 512 và 364 bước ở batch 1024; 20 epoch tương ứng 14540 và 7280 bước. Kết quả phù hợp dự đoán về đánh đổi tốc độ và chất lượng trong ngân sách epoch này. Không tách được tác dụng của batch khỏi tác dụng của số bước cập nhật khi chỉ giữ số epoch cố định.

!\[](figures/compare\_hparam.png)

### dropout

**Dự đoán trước khi chạy:** Dropout 0.2 có thể giảm quá khớp; nếu baseline chưa quá khớp thì có thể làm việc học chậm hơn.

|exp\_id|val\_macro\_f1|delta\_vs\_base\_s1|seconds\_per\_epoch|
|-|-|-|-|
|dropout-p02|0.8166|-0.0244|1.4374|

Dropout 0,2 không cải thiện trong cấu hình này: macro-F1 giảm từ 0,8410 xuống 0,8166. Ở epoch cuối, baseline có train/val loss 0,2153/0,2365, gap 0,0212; dropout có 0,2693/0,2778, gap 0,0084. Dropout giảm gap nhưng làm cả train và val loss cao hơn, phù hợp với việc regularization làm khó tối ưu hơn. Gap nhỏ hơn không tự động đồng nghĩa mô hình tốt hơn; dữ liệu chưa cho thấy cần thêm dropout để cải thiện điểm.

!\[](figures/compare\_dropout.png)

### clipping

**Dự đoán trước khi chạy:** Clipping có thể không giúp ở learning rate ổn định; ngưỡng thấp cũng có thể làm học chậm. Learning rate gấp 5 baseline có thể gây dao động hoặc phân kỳ. Clipping có thể ổn định cập nhật ở learning rate cao, nhưng không đảm bảo cứu được huấn luyện.

|exp\_id|val\_macro\_f1|delta\_vs\_base\_s1|seconds\_per\_epoch|
|-|-|-|-|
|clip-normal-lr|0.8255|-0.0155|1.3849|
|highlr-no-clip|0.8408|-0.0002|1.2945|
|highlr-with-clip|0.8513|0.0103|1.3881|

Ngưỡng clip là 0,283431. Ở lr=0,1, clipping kích hoạt khoảng 99,99% các bước có gradient hữu hạn và macro-F1 giảm 0,0155, cho thấy ngưỡng này hạn chế cập nhật quá thường xuyên. Ở lr=0,5, cả hai run đều không phân kỳ; bật clipping kích hoạt khoảng 76,81% các bước và tăng F1 từ 0,8408 (`highlr-no-clip`) lên 0,8513 (`highlr-with-clip`), tăng 0,0104 trên cùng seed. Vì run không clip vẫn ổn định, không thể nói clipping đã cứu một run phân kỳ. Kiểm tra 3 seed bên dưới không xác nhận ưu thế trung bình so với baseline.

!\[](figures/compare\_clipping.png)

### amp

**Dự đoán trước khi chạy:** BF16 có thể ổn định hơn FP16 nhờ miền biểu diễn rộng hơn. FP16 có thể giảm bộ nhớ và thời gian, nhưng MLP nhỏ có thể không nhanh hơn do chi phí quản lý AMP. FP32 làm mốc thời gian và bộ nhớ cho pipeline AMP.

|exp\_id|val\_macro\_f1|delta\_vs\_base\_s1|seconds\_per\_epoch|
|-|-|-|-|
|amp-bf16|0.8469|0.0059|1.5435|
|amp-fp16|0.8451|0.0042|1.7948|
|amp-fp32|0.8410|0.0000|1.2886|

Thời gian phần train trung bình theo log: FP32 1,2359 giây/epoch, FP16 1,7407 giây (chậm hơn khoảng 40,8%), BF16 1,4977 giây (chậm hơn khoảng 21,2%). Peak allocated gần như bằng nhau, khoảng 161,12 MiB. Logit được chuyển FP32 để tính loss, dữ liệu và trọng số vẫn FP32, nên AMP không giảm toàn bộ bộ nhớ pipeline. Với MLP nhỏ, chi phí autocast, GradScaler và quản lý batch có thể vượt lợi ích tính toán. Các chênh lệch F1 +0,0042/+0,0059 nhỏ hơn mốc nhiễu. Thời gian chỉ đo một run cho mỗi cấu hình; BF16 chạy được không chứng minh hỗ trợ native trên GPU.

!\[](figures/compare\_amp.png)

### init

**Dự đoán trước khi chạy:** Xavier có thể tạo scale kích hoạt khác He, ảnh hưởng loss ban đầu và tốc độ hội tụ. Khởi tạo zero khiến các lớp ẩn không học được đặc trưng; bias đầu ra có thể vẫn học phân bố lớp.

|exp\_id|val\_macro\_f1|delta\_vs\_base\_s1|seconds\_per\_epoch|
|-|-|-|-|
|init-xavier|0.8374|-0.0036|1.2831|
|init-zeros|0.0936|-0.7473|1.2966|

He dùng variance 2/fan\_in, phù hợp ReLU; Xavier dùng 2/(fan\_in+fan\_out). Khởi tạo zero làm các lớp ẩn không học được đặc trưng, dù bias đầu ra vẫn có thể học phân bố lớp.

!\[](figures/compare\_init.png)

### Xác nhận cấu hình ứng viên

|seed|baseline\_f1|candidate\_f1|paired\_delta|
|-|-|-|-|
|1|0.8410|0.8513|0.0103|
|2|0.8639|0.8542|-0.0097|
|3|0.8579|0.8463|-0.0116|

Baseline trung bình F1=0.8543; ứng viên=0.8506. Chọn `base-s1` theo quy tắc trung bình validation, seed nộp cố định 1. Không chọn seed bằng eval. Ứng viên hơn baseline ở seed 1 nhưng kém ở seed 2 và 3; trung bình giảm 0,0037. Vì vậy giữ baseline, không chỉ chọn kết quả một seed thuận lợi.

## 4\. Đánh giá cuối

Nạp checkpoint có validation loss thấp nhất. Các điểm eval lấy từ script chấm chính thức; không chỉnh cấu hình sau khi xem eval. Cấu hình cuối chính là baseline nên mức cải thiện eval so với baseline là 0. Macro-F1 eval 0,8427 gần validation 0,8410 (chênh 0,0018); accuracy eval 0,9043 so với validation 0,9069. Chỉ đánh giá seed nộp 1 nên không có ước lượng nhiễu eval qua nhiều seed.

|role|exp\_id|val\_macro\_f1|eval\_macro\_f1|eval\_accuracy|
|-|-|-|-|-|
|baseline|base-s1|0.8410|0.8427|0.9043|
|final|base-s1|0.8410|0.8427|0.9043|

### Phân tích lỗi theo lớp

|cls|support|precision|recall|f1|
|-|-|-|-|-|
|0|42368|0.9106|0.8947|0.9026|
|1|56661|0.9083|0.9342|0.9211|
|2|7151|0.8655|0.9171|0.8905|
|3|549|0.8227|0.7523|0.7859|
|4|1899|0.8485|0.6251|0.7198|
|5|3473|0.8418|0.7244|0.7787|
|6|4102|0.9324|0.8708|0.9005|

!\[](figures/eval\_confusion.png)

Lớp 4 (Aspen) khó nhất: F1 0,7198, recall 0,6251, precision 0,8485. Có 606/1899 mẫu lớp 4 bị nhầm sang lớp 1, tương đương 31,9%. Lớp 5 cũng có recall thấp (0,7244), thường nhầm sang lớp 2 (675 mẫu). Lớp 1 có F1 cao nhất (0,9211). Lớp ít mẫu và đặc trưng chồng lấn là giả thuyết giải thích, chưa phải quan hệ nhân quả được chứng minh. Nếu có thêm thời gian, thử weighted CE, chọn trọng số và cấu hình bằng validation thay vì eval.

## 5\. Câu hỏi dẫn dắt

1. Adam lr=0,001 đạt F1 0,8476, SGD momentum lr=0,1 đạt 0,8410 trên seed 1. Chênh lệch 0,0067 nhỏ hơn 2σ=0,0238. Chưa tune learning rate với ngân sách ngang nhau nên chưa kết luận optimizer nào thắng.
2. Dropout 0,2 giảm gap loss từ 0,0212 xuống 0,0084 nhưng giảm F1 0,0244. Trong bài này gap nhỏ hơn không mang lại điểm tốt hơn; nên thử dropout khi có bằng chứng quá khớp trên validation, không mặc định bật.
3. Clipping giới hạn gradient lớn. Tần suất clip 99,99% ở lr thường đi kèm giảm F1; ở lr cao, tần suất 76,81% đi kèm tăng F1 trên seed 1, nhưng lợi ích không giữ được qua cả 3 seed.
4. FP16 và BF16 chậm hơn FP32 khoảng 40,8% và 21,2% trong phần train. Mạng nhỏ và việc giữ dữ liệu/trọng số FP32 có thể làm chi phí quản lý AMP lớn hơn lợi ích. Chưa đo riêng từng thành phần chi phí; BF16 chạy được không xác nhận hỗ trợ native.
5. Zero khiến lớp ẩn không có gradient hữu ích qua ReLU; kết quả F1 0,0936 xác nhận thất bại dù loss vẫn giảm nhờ bias đầu ra. He phù hợp ReLU theo variance 2/fan_in; Xavier có variance 2/(fan_in+fan_out). Kết quả He/Xavier ở đây chênh ít hơn nhiễu seed.
6. Nếu loss không giảm sau 2000 bước: (a) kiểm tra nhãn, shape và thống kê dữ liệu; (b) kiểm tra gradient, zero\_grad/backward/optimizer.step; (c) thử học thuộc 20 mẫu và thử learning rate. Các phép kiểm tra này lần lượt giúp phát hiện lỗi đầu vào, lỗi cập nhật và vấn đề tối ưu.

## 6\. Hạn chế

Chỉ 3 seed cho baseline và ứng viên; các thí nghiệm còn lại chủ yếu một seed. Pilot learning rate chỉ 5 epoch. Cùng epoch nhưng khác batch tạo ngân sách cập nhật khác nhau. Thời gian phụ thuộc tải GPU và chưa đo lặp nhiều lần. Không suy ra ưu thế chắc chắn từ delta nhỏ. Một số checkpoint tốt nhất ở epoch cuối nên chưa kiểm tra đầy đủ lợi ích của ngân sách huấn luyện dài hơn. Kết quả bất ngờ là AMP chậm hơn FP32 và ứng viên clipping tốt trên seed 1 nhưng không hơn baseline trung bình. Nếu có thêm thời gian, ưu tiên tune optimizer công bằng, thử weighted CE và xác nhận các cấu hình tốt qua nhiều seed.

## 7\. File và thời gian

Bài nộp gồm code/lab.ipynb, các module Python, experiments.xlsx, predictions\_eval.csv, eval\_result.json, figures/ và results/.

Tổng thời gian pipeline ghi nhận cho các run trong bảng: 477.5 giây; không gồm setup, vẽ hình và xuất file.



