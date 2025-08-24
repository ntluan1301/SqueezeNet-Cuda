# SqueezeNet-Cuda
Implement SqueezeNet with a forward-backward optimizer from scratch using CUDA programming.

## Giải Thích Tác Dụng của Các File

### 1. **File: Cuda_Adam_Tunning_Parameter.ipynb**
   - **Mô tả**: Chứa code CUDA xây dựng mô hình SqueezeNet và sử dụng Adam optimizer để tinh chỉnh tham số sau 5 epoch.
   - **Tác dụng**:
     - Giúp mô hình vượt qua giai đoạn khó khăn ban đầu mà SGD không thể học được.
     - Tối ưu hóa tham số ban đầu nhờ đặc tính thích nghi tốc độ học của Adam.
     - Tạo checkpoint chất lượng cao (`model_fintuning_by_Adam.npz`) để sử dụng trong các giai đoạn huấn luyện tiếp theo

### 2. **File: Cuda_Training_SGD.ipynb**
   - **Mô tả**: Chứa code CUDA xây dựng mô hình SqueezeNet và sử dụng SGD optimizer để huấn luyện, khởi tạo từ tham số đã được tinh chỉnh bởi Adam.
   - **Tác dụng**:
     - Tiếp tục huấn luyện với tham số đã được tối ưu bởi Adam, tận dụng SGD để đạt độ chính xác cao hơn , hội tụ tốt hơn.
     - Kết hợp ưu điểm của Adam (tinh chỉnh ban đầu) và SGD (huấn luyện dài hạn).
### Ghi Chú: Lý Do Sử Dụng Kết Hợp SGD và Adam
   - **Lý do 1**: SGD gặp khó khăn trong giai đoạn đầu, không thể vượt qua vùng không học được.
   - **Lý do 2**: Adam sau 5 epoch giúp mô hình vượt qua giai đoạn khó khăn, nhưng từ epoch thứ 5 trở đi, triển khai Adam của nhóm gặp vấn đề gradient vanishing, làm giảm độ chính xác.
   - **Lý do 3**: Triển khai SGD của mình đảm bảo tương đồng 100% với PyTorch, cho phép sự học mạnh mẽ của mô hình sau khi vượt qua giai đoạn khó khăn nhờ Adam.
## Hướng Dẫn Sử Dụng
**Lưu ý**: Cần có GPU để chạy các file này.

###**Huấn Luyện Lại Từ Đầu**
   - **Bước 1**: Tải dataset từ: [Tomato Diseases Dataset](https://www.kaggle.com/datasets/luisolazo/tomato-diseases).
   - **Bước 2**: Trong file `Cuda_Adam_Tunning_Parameter.ipynb`, thay đường dẫn `/kaggle/input/tomato-diseases` bằng đường dẫn đến dataset đã tải.
   - **Bước 3**: Chạy file `Cuda_Adam_Tunning_Parameter.ipynb` để tinh chỉnh tham số bằng Adam.
   - **Bước 4**: Tìm checkpoint `model_fintuning_by_Adam.npz` được lưu bởi hàm `np.savez(f'model_fintuning_by_Adam.npz', weights=weights_np, m_adam=M_adam_np, v_adam=V_adam_np)` trong file trên.
   - **Bước 5**: Trong file `Cuda_Training_SGD.ipynb`:
     - Thay đường dẫn `/kaggle/input/tomato-diseases` bằng đường dẫn đến dataset.
     - Thay đường dẫn `/kaggle/input/epoch_30_sgd/other/default/1/model_data_SGD_epoch_25.npz` bằng `model_fintuning_by_Adam.npz` từ bước 4.
     - Chạy file để huấn luyện tiếp với SGD.
## Lưu Ý
- Đảm bảo môi trường có GPU để chạy các file.
- Kiểm tra kỹ các đường dẫn file và dataset trước khi chạy.
- Kết quả chi tiết và biểu đồ hiệu suất được lưu trong file `Presentation - SqueezeNet for Leaf Disease Detection.pdf`
