# Hướng Dẫn Sử Dụng

## Giải Thích Tác Dụng của Các File và Thư Mục

### 1. **Thư Mục: CheckPoint**
   - **Mô tả**: Lưu trữ các checkpoint của tham số mô hình với `model_data_Adam_tuning.npz` là tham số mô hình sau khi được tinh chỉnh bởi bộ tối ưu hóa Adam. Các checkpoint còn lại là kết quả từ quá trình huấn luyện sử dụng SGD (Stochastic Gradient Descent) với tham số(`model_data_Adam_tuning.npz`) đã được tinh chỉnh bởi Adam.
   - **Tác dụng**: Cung cấp các trạng thái mô hình đã được lưu để tiếp tục huấn luyện hoặc đánh giá, đảm bảo tính liên tục và hiệu quả trong quá trình phát triển.

### 2. **Thư Mục: All_Module_Squeenet**
   - **Mô tả**: Chứa tất cả các module cấu thành kiến trúc SqueezeNet, bao gồm convolution, Fire Module (gồm các lớp squeeze và expand convolution), ReLU, max pooling, softmax,Global average pooling cả forward lẫn backward
   - **Tác dụng**: 
     - Cung cấp các thành phần cơ bản để xây dựng và triển khai mô hình SqueezeNet.
     - Cho phép so sánh hiệu năng của các module được triển khai trên CUDA, NumPy và PyTorch.

### 3. **File: Cuda_Adam_Tunning_Parameter.ipynb**
   - **Mô tả**: Chứa code CUDA xây dựng mô hình SqueezeNet và sử dụng Adam optimizer để tinh chỉnh tham số sau 5 epoch.
   - **Tác dụng**:
     - Giúp mô hình vượt qua giai đoạn khó khăn ban đầu mà SGD không thể học được.
     - Tối ưu hóa tham số ban đầu nhờ đặc tính thích nghi tốc độ học của Adam.
     - Tạo checkpoint chất lượng cao (`model_fintuning_by_Adam.npz`) để sử dụng trong các giai đoạn huấn luyện tiếp theo.

### 4. **File: Cuda_SGD_optimizer.ipynb**
   - **Mô tả**: Chứa code CUDA xây dựng mô hình SqueezeNet với bộ tối ưu hóa SGD, so sánh với triển khai PyTorch sau 20 lần cập nhật tham số.
   - **Tác dụng**:
     - Đánh giá hiệu suất của SGD trên CUDA so với PyTorch, xác định độ tin cậy và hiệu quả của mã CUDA.

### 5. **File: Cuda_Training_SGD.ipynb**
   - **Mô tả**: Chứa code CUDA xây dựng mô hình SqueezeNet và sử dụng SGD optimizer để huấn luyện, khởi tạo từ tham số đã được tinh chỉnh bởi Adam.
   - **Tác dụng**:
     - Tiếp tục huấn luyện với tham số đã được tối ưu bởi Adam, tận dụng SGD để đạt độ chính xác cao hơn , hội tụ tốt hơn.
     - Kết hợp ưu điểm của Adam (tinh chỉnh ban đầu) và SGD (huấn luyện dài hạn).
### 6. **Fire: inference.ipynb**:
    - **Mô tả**: Chứa code CUDA inference mô hình SqueezeNet 
    - **Tác dụng**: Kiểm tra độ chính xác của mô hình khi áp dụng checkpoint được huấn luyện, xem tốc độ inference
### Ghi Chú: Lý Do Sử Dụng Kết Hợp SGD và Adam
   - **Lý do 1**: SGD gặp khó khăn trong giai đoạn đầu, không thể vượt qua vùng không học được.
   - **Lý do 2**: Adam sau 5 epoch giúp mô hình vượt qua giai đoạn khó khăn, nhưng từ epoch thứ 5 trở đi, triển khai Adam của nhóm gặp vấn đề gradient vanishing, làm giảm độ chính xác.
   - **Lý do 3**: Triển khai SGD của nhóm đảm bảo tương đồng 100% với PyTorch, cho phép sự học mạnh mẽ của mô hình sau khi vượt qua giai đoạn khó khăn nhờ Adam.
## Hướng Dẫn Sử Dụng
**Lưu ý**: Cần có GPU để chạy các file này.

### 1. **Kiểm Tra Độ Chính Xác của Mô Hình**
   - **Mô tả**: So sánh độ chính xác giữa triển khai CUDA và PyTorch với SGD optimizer.
   - **Hướng dẫn**:
     - Chạy file `Cuda_SGD_optimizer.ipynb`.
     - Kết quả ở cuối file sẽ hiển thị độ sai số `rtol >= 10⁻⁸`, chứng minh trọng số sau 20 lần cập nhật bởi CUDA và PyTorch có độ tương đồng lên đến 99.999999%.

### 2. **Huấn Luyện Lại Từ Đầu**
   - **Bước 1**: Tải dataset từ: [Tomato Diseases Dataset](https://www.kaggle.com/datasets/luisolazo/tomato-diseases).
   - **Bước 2**: Trong file `Cuda_Adam_Tunning_Parameter.ipynb`, thay đường dẫn `/kaggle/input/tomato-diseases` bằng đường dẫn đến dataset đã tải.
   - **Bước 3**: Chạy file `Cuda_Adam_Tunning_Parameter.ipynb` để tinh chỉnh tham số bằng Adam.
   - **Bước 4**: Tìm checkpoint `model_fintuning_by_Adam.npz` được lưu bởi hàm `np.savez(f'model_fintuning_by_Adam.npz', weights=weights_np, m_adam=M_adam_np, v_adam=V_adam_np)` trong file trên.
   - **Bước 5**: Trong file `Cuda_Training_SGD.ipynb`:
     - Thay đường dẫn `/kaggle/input/tomato-diseases` bằng đường dẫn đến dataset.
     - Thay đường dẫn `/kaggle/input/epoch_30_sgd/other/default/1/model_data_SGD_epoch_25.npz` bằng `model_fintuning_by_Adam.npz` từ bước 4.
     - Chạy file để huấn luyện tiếp với SGD.

### 3. **Kiểm Tra Độ Chính Xác và Tốc Độ của quá trình Inference**
   - **Bước 1**: Tải dataset từ: [Tomato Diseases Dataset](https://www.kaggle.com/datasets/luisolazo/tomato-diseases).
   - **Bước 2**: Trong file `inference.ipynb`:
     - Thay đường dẫn `/kaggle/input/tomato-diseases` bằng đường dẫn đến dataset đã tải.
     - Thay đường dẫn `/kaggle/input/epoch_30_sgd/other/default/1/model_data_SGD_epoch_25.npz` bằng `squeezenet_manual_state_dict_2.pth` trong thư mục checkpoint,
     - **Lưu ý**: Nếu sử dụng checkpoint NumPy từ quá trình huấn luyện CUDA, thay đoạn code:
       ```python
       state = torch.load("/kaggle/input/squeeznetinfer/pytorch/default/1/squeezenet_manual_state_dict_2.pth", map_location="cpu")
       weights_np = {k: v.cpu().numpy() for k, v in state.items()}
       ```
       bằng:
       ```python
       load_state_5 = np.load("/kaggle/input/{LinkCheckPoint}.npz", allow_pickle=True)
       weights_np = {k: v for k, v in load_state_5["weights"].item().items()}
       ```
       trong đó `LinkCheckPoint` là đường dẫn đến checkpoint được lưu bởi mô hình huấn luyện bằng CUDA.
## Lưu Ý
- Đảm bảo môi trường có GPU để chạy các file.
- Kiểm tra kỹ các đường dẫn file và dataset trước khi chạy.
- Kết quả quá trình huân luyện của nhóm vẫn chưa hội tụ có thể đạt >90% accuracy, nhưng vì lý do thời gian và dung lượng GPU nên nhóm dừng ở epochs 45.
- Kết quả chi tiết và biểu đồ hiệu suất được lưu trong file `Presentation - SqueezeNet for Leaf Disease Detection.pdf`
