import scipy
import math
import numpy as np
import scipy.signal
import scipy.io.wavfile
import matplotlib.pyplot as plt

"""  1. Hàm framing:
     Chia tín hiệu mẫu vào các overlapping frames
     Args:
         sig            (array) : tín hiệu mẫu (Nx1).
         fs               (int) : tần số lấy mẫu.
         win_len        (float) : chiều dài cửa số (tính bằng giây).
                                  mặc định là 0.02s
         win_hop        (float) : bước chuyển giữa 2 cửa số liên tiếp
                                  mặc định là 0.01s
     Returns:
         array of frames.
         frame length.
"""
def framing(sig, fs, win_len, win_hop):
     # chuyển từ cửa số (giây) sang mẫu (samples)
     signal_length = len(sig)                       # độ dài tín hiệu gốc
     frame_length = win_len * fs                    # tính độ dài khung
     frame_step = win_hop * fs                      # tính bước nhảy khung
     frames_overlap = frame_length - frame_step     # tính độ dài giữa 2 khung chồng nhau

     # tính số frame và số mẫu bị dư ra
     num_frames = np.abs(signal_length - frames_overlap) // np.abs(frame_length - frames_overlap)
     rest_samples = np.abs(signal_length - frames_overlap) % np.abs(frame_length - frames_overlap)

     # pad_signal chứa tín hiệu mẫu gốc và thêm các phần tử 0 để đảm bảo khi chia
     # các frame có số phần tử bằng nhau.     
     if rest_samples != 0: 
         pad_signal_length = int(frame_step - rest_samples)
         z = np.zeros((pad_signal_length))
         pad_signal = np.append(sig, z)
         num_frames += 1
     else:
         pad_signal = sig

     # đảm bảo các chỉ số đều là số nguyên
     frame_length = int(frame_length)
     frame_step = int(frame_step)
     num_frames = int(num_frames)

     # chia các tín hiệu vào các frame
     idx1 = np.tile(np.arange(0, frame_length), (num_frames, 1))
     idx2 = np.tile(np.arange(0, num_frames * frame_step, frame_step),
                    (frame_length, 1)).T
     
     # indices có dạng: [[           0,                1,......frame_step - 1],
     #                   [  frame_step,   frame_step + 1,....2*frame_step - 1]
     #                   [2*frame_step, 2*frame_step + 1,....3*frame_step - 1]
     #                   [...........  ...........    ......  .............. ]]
     indices = idx1 + idx2
     frames = pad_signal[indices.astype(np.int32, copy=False)]
     return frames, frame_length, frames_overlap


""" 2. Hàm calculate_STE:
    Tính Short-time Energy (STE) của các khung tín hiệu được chia sẵn.
    Args:
         frames            (array) : các khung tín hiệu đã được chia.
    Returns:
         array of STE values.   
"""
def calculate_STE(frames):
    # tính STE theo công thức: tổng bình phương các phần tử của từng frame
    energy = []
    for i in range(len(frames)):
        ste = np.sum(abs(np.square(frames[i])))
        energy.append(ste)
    return np.asarray(energy)


""" 3. Hàm get_index_voice:
    Lấy vị trí bắt đầu và kết thúc của tín hiệu tiếng nói (lúc chưa lọc nhiễu)
    Args:
         vad         (array) : các khung giá trị 0 & 1 khi lọc bằng threshold
    Returns:
         big_arr     (array) : mảng chứa chỉ số bắt đầu & kết thúc của speech
"""
def get_index_voice(vad):
    big_arr = []                    # mảng output
    small_arr = []                  # chứa index của đoạn tín hiệu speech
    count = 0                       # đếm số tín hiệu trong 1 small_array
    for i in range(len(vad)):
        if(vad[i] == 1):
            small_arr.append(i)
            count += 1
        else:
            if(count == 0):
                continue
            else:
                # big_arr chỉ llaaysindex bắt đầu - kết thúc của small_arr
                big_arr.append([small_arr[0], small_arr[-1]])
                small_arr = []
                count = 0
    return big_arr


""" 4. Hàm get_index_filter_noise:
    Lấy vị trí bắt đầu và kết thúc của tín hiệu tiếng nói (lọc khoảng lặng > 300ms)
    Args:
         raw            (array) : mảng chứa vị trí bắt đầu - kết thúc của từng voice 
                                  (lúc chưa lọc nhiễu)
         noise          (float) : khoảng cách tối thiểu giữa 2 khoảng lặng
    Returns:
         result         (array) : mảng chứa chỉ số bắt đầu & kết thúc của từng voice
"""
def get_index_filter_noise(raw, noise):
    x = raw[0][0]
    y = raw[0][1]

    filtered = []
    result = []
    # tạo mảng chứa các phần tử dạng [start, end] của voice khi đã lọc nhiễu
    for i in range(1,len(raw)):
        if (raw[i][0] - raw[i-1][1] > noise):
            filtered.append([x,y])
            x = raw[i][0]
            y = raw[i][1]
        else:
            y = raw[i][1]
            
    if len(filtered)==0:
        filtered.append([x,y])
    elif x != filtered[-1][0]:
       filtered.append([x,y])
       
    # tạo mảng chứa các mốc để vẽ đường phân cách
    for i in range(len(filtered)):
        if (filtered[i][1] - filtered[i][0] > noise):
            result.append(filtered[i][0])
            result.append(filtered[i][1])
    return np.asarray(result)


""" 5. Hàm voice_frame:
    Lấy chỉ số tín hiệu voice theo tín hiệu gốc sau khi lọc nhiễu
    Args:
         sig            (array) : tín hiệu gốc sau khi qua hàm framing
         indices        (array) : mảng chứa chỉ số bắt đầu & kết thúc của từng voice
         frame_len      (float) : độ dài 1 khung
         frame_overlap  (float) : độ chồng giữa 2 khung liên tiếp
    Returns:
         index_real     (array) : mảng chứa chỉ số bắt đầu - kết thúc của tiếng nói
                                  trong tín hiệu gốc.
         filtered_vad   (array) : mảng chứa 0 (khoảng lặng), 1 (tiếng nói).
"""
def voice_frame(sig, indices, frame_len, frame_overlap):
    index_real = []
    for i in range(0, len(indices) - 1, 2):
        start = int(indices[i] - (indices[i] // frame_len) * frame_overlap)
        end = int(indices[i + 1] - (indices[i + 1] // frame_len) * frame_overlap)
        index_real.append([start, end])
    index_real = np.asarray(index_real)
    
    filtered_vad = np.zeros(sig.shape[0])
    for i in range(0, len(indices) - 1, 2):
        for j in range(indices[i], indices[i+1]):
            filtered_vad[j] = 1
    return index_real, filtered_vad


""" 6. Hàm predict_threshold:
    Tính threshold dựa trên thuật toán 2 giá trị STE
    Args:
        ste         (array) : mảng chứa giá trị STE
    Returns:          
        threshold   (float) : giá trị ngưỡng tính được
"""
def predict_threshold(ste):
    weight = 0.5
    histSTE, index_STE = np.histogram(ste, round(len(ste)*5))

    max_histSTE1 = 0
    max_histSTE2 = 0

    max_index1 = 0
    max_index2 = 0
    
    for i in range(1, len(histSTE) - 1):
        pre = i - 1
        nex = i + 1
        while(histSTE[i] == histSTE[nex]):
            nex += 1
        if histSTE[i] > histSTE[pre] and histSTE[i] > histSTE[nex]:
            if(max_index1 == 0):
                max_histSTE1 = histSTE[i]
                max_index1 = i
            else:
                max_histSTE2 = histSTE[i]
                max_index2 = i
                break
        i = nex
            
    max_histSTE1 = index_STE[max_index1]
    max_histSTE2 = index_STE[max_index2]

    threshold = (weight * max_histSTE1 + max_histSTE2) / ((weight + 1))
    return threshold


""" 7. Hàm speech_silence:
     Dựa vào giá trị STE để xác định các mốc phân chia speech/silence
     ==> Hàm chính của bài
     Args:
         sig            (array) : tín hiệu mẫu (Nx1).
         fs               (int) : tần số lấy mẫu.
         threshold      (float) : ngưỡng muốn chọn (không bắt buộc)
         noise          (float) : khoảng nhiễu tối thiếu muốn lọc (s)
         win_len        (float) : chiều dài cửa số (tính bằng giây).
                                  mặc định là 0.02s
         win_hop        (float) : bước chuyển giữa 2 cửa số liên tiếp
                                  mặc định là 0.01s
     Returns:
         ste            (array) : giá trị STE
         filtered_vad   (array) : giá trị vad (sau khi lọc nhiễu)
         index_real     (array) : chỉ số đánh dấu speech/silence
"""
def speech_silence(sig, fs, threshold, noise, win_len, win_hop):
    
    frames, fr_len, fr_overlap = framing(sig=sig, fs=fs, 
                                         win_len=win_len, win_hop=win_hop)
    # tính STE và chuẩn hóa
    ste = calculate_STE(frames)
    ste = (ste - min(ste))/(max(ste) - min(ste))
    
    # dự đoán ngưỡng
    pred_threshold = predict_threshold(ste)
    print("Threshold computed by Function:", pred_threshold)
    
    # sinh giá trị energy cho từng frame
    ste = np.repeat(ste, fr_len)

    # tính vad (voice activity detect): phát hiện tiếng nói
    # threshold / pred_threshold / sta_threshold
    vad = np.array(ste > pred_threshold, dtype=sig.dtype)
    
    # lấy vị trí bắt đầu - kết thúc của từng đoạn vad
    index_voice_raw = get_index_voice(vad)
    
    # lấy vị trí bắt đầu - kết thúc của các đoạn voice cách nhau > noise
    filtered = get_index_filter_noise(index_voice_raw, noise*fs)
    
    # lấy chỉ số tín hiệu của tiếng nói và phân vùng tiếng nói (0,1)
    index_real, filtered_vad = voice_frame(np.asarray(frames.flatten()), 
                                           filtered, fr_len, fr_overlap)
    return ste, filtered_vad, index_real


""" 8. Hàm multi-plots:
    Vẽ đồ thị tín hiệu (input, ste, vad, output)
     Args:
         data            (array) : mảng các tín hiệu cần vẽ.
         lab             (array) : mảng chứa chỉ số đọc từ file lab
         titles          (array) : mảng chứa tiêu đề của từng tín hiệu
         fs               (int)  : tần số lấy mẫu.
         plot_rows        (int)  : số đồ thị trong 1 figure
         step             (int)  : bước nhảy tín hiệu trên trục thời gian
         colors          (array) : mảng chứa kí hiệu màu sắc
     Returns:
         figure with multi-plots
"""
def multi_plots(filename, data, lab, titles, fs, plot_rows, win_len, win_hop, step=1, colors=["b", "r", "m","g"]):

    # chia các phần nhỏ để vẽ đồ thị trên 1 figure
    plt.subplots(plot_rows, 1, figsize=(25, 15))
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.2, top=0.93, 
                        wspace=0.05, hspace=0.99)
    
    plt.suptitle(filename)
    
    # tín hiệu gốc, điểm bắt đầu - kết thúc của tiếng nói
    sig, index_real = data[0], data[-1]
    
    for j in range(plot_rows):
        # vẽ output signal (tín hiệu gốc và các đường phân chia tiếng nói)
        if(j == 3):
            plt.subplot(plot_rows, 1, j+1)
            plt.xlabel('Time(s)')
            plt.ylabel('Signal Amplitude')
            
            # Tín hiệu gốc
            plt.plot([i/fs for i in range(len(sig))], sig, colors[j])
            
            # Mốc thời gian chia bởi chương trình STE
            plt.vlines(index_real.flatten()/fs, ymin=-max(abs(sig)), ymax=max(abs(sig)), 
                       linestyles='dashed', colors='blue', label='By student')
            
            plt.vlines([0, (len(sig)-1)/fs], ymin=-max(abs(sig)), ymax=max(abs(sig)), 
                       linestyles='dashed', colors='blue')
            
            # Mốc thời gian chia bởi file LAB
            plt.vlines(lab, ymin=-max(abs(sig)), ymax=max(abs(sig)), 
                       linestyles='dashed', colors='red', label='By LAB')
            plt.gca().set_title(titles[j])
        else:
            plt.subplot(plot_rows, 1, j+1)
            y = data[j]
            plt.xlabel('Time(s)')
            if (j == 0):
                plt.plot([i/fs for i in range(0, len(y), step)], y)
                plt.ylabel('Signal Amplitude')
            else:
                plt.plot([i/(win_len*fs/win_hop) for i in range(0, len(y), step)], y, colors[j])
            plt.gca().set_title(titles[j])
            
    plt.legend(loc='best')
    plt.show()
    
""" 9. Hàm read_file_lab:
    Đọc dữ liệu từ filename (.lab) trả về mảng chỉ số speech/silence (giây)
"""
def read_file_lab(filename):
    froot = "E:\DUT\Kì_5\Xử lí THS\Tín hiệu mẫu\OutputLab"
    file = froot + filename + '.lab'
    result = []
    with open(file) as rf:
        lines = rf.readlines()
        for idx, line in enumerate(lines):
            list = line.split()
            if len(list) == 3 and list[2] == 'sil':
                result.append(float(list[0]))
                result.append(float(list[1]))
    result = np.asarray(result)
    return result

""" 10. Hàm write_file_lab:
    Ghi chỉ số speech/silence vào file .lab mới để so sánh với file mẫu
"""
def write_file_lab(filename, index, end):
    froot = "E:\DUT\Kì_5\Xử lí THS\Tín hiệu mẫu\OutputLab"
    file = froot + filename + '_student.lab'
    f = open(file, "w")
    s = ""
    s += "0.00" + "\t" + str(index[0]) + "\t" + "sil \n" 
    for i in range(0, len(index) - 1 , 2):
        s += str(index[i]) + "\t" + str(index[i+1]) + "\t" + "voice\n"
        if (i + 1) == len(index) - 1:
            s += str(index[i+1]) + "\t" + str(end) + "\t" + "sil \n" 
        else:
            s += str(index[i+1]) + "\t" + str(index[i+2]) + "\t" + "sil \n"    
    f.write(s);
    f.close()


""" 11. Hàm std_mean:
    Tính độ lệch chuẩn (std) và giá trị trung bình (mean) của mảng giá trị STE.
    Args:
         ste            (array) : giá trị STE của các khung tín hiệu.
    Returns:
         std, mean   
"""
def std_mean(ste):
    n = len(ste)
    mean_ste = np.sum(ste) / n
    deviations = [(x - mean_ste)**2 for x in ste]
    variance = np.sum(deviations) / n
    std = math.sqrt(variance)
    return std, mean_ste


""" 12. Hàm predict_std_mean:
    Tính độ lệch chuẩn (std) và giá trị trung bình (mean) của silence từ file Lab.
    ==> std và mean này sẽ được dùng để dự đoán threshold.
    Args:
         lab            (array) : mảng chứa các mốc đánh dấu silence.
    Returns:
         std, mean  
"""
def predict_std_mean(sig, lab, fs, win_len, win_hop):
    start = sig[:int((lab[1]) * fs)]
    end = sig[int((lab[2]) * fs):]  
    signal = np.append(start, end)
    print(signal)
    frames, fr_len, fr_overlap = framing(sig=signal, fs=fs, 
                                         win_len=win_len, win_hop=win_hop)
    ste = calculate_STE(frames)
    std, mean = std_mean(ste)
    print("mean:", mean)
    print("std:", std)
    return std, mean


""" 13. Hàm main:
    Cho chạy từ file 1 đến 4 trong folder tín hiệu mẫu và thực hiện xử lí 
    chương trình phân loại tiếng nói - khoảng lặng bằng thuật toán STE
    
    Các phần tiếng nói sau khi đã được xử lí sẽ được ghi vào file với 
    tên file có dạng filename_only_voice.wav
"""
if __name__ == "__main__":
    # tạo tên file
    froot = "E:\DUT\Kì_5\Xử lí THS\Tín hiệu mẫu"
    filename = ["\studio_M2", "\studio_F2", "\phone_M2", "\phone_F2"]
    threshold = []
    for i in range(len(filename)):
        print("Figure", i + 1, ": STE", filename[i])
        fname = froot + filename[i] + '.wav'
        fs, sig = scipy.io.wavfile.read(fname)
        normalized_sig = sig/max(abs(sig))
        
        # đọc tín hiệu tiếng nói từ file lab
        lab = read_file_lab(filename[i])
        
        # lấy std, mean của file lab huấn luyện (chỉ cần chạy 1 lần để lấy ngưỡng chung)
        std, mean = predict_std_mean(sig=normalized_sig, lab=lab, fs=fs, 
                                     win_len=0.025, win_hop=0.02)
        threshold.append([std, mean])

        # lấy mảng STE, vad, mốc chỉ số đánh dấu tiếng nói trong tín hiệu gốc
        energy, vad, index_real = speech_silence(normalized_sig, fs, threshold=0.03, 
                                                 noise=0.3, win_len=0.025, win_hop=0.02)
        
        # xuất đồ thị minh họa
        fig_name = "Figure: " + str(i + 1) + " " + filename[i]
        multi_plots(fig_name, data=[sig, energy, vad, index_real], lab=lab,
                titles=["Input signal (voiced + silence)", "Short time energy",
                        "Voice activity detection", "Ouput signal (voice marked)"],
                fs=fs, plot_rows=4, win_len=0.025, win_hop=0.02, step=1)
        
        #mảng voice chỉ chứa tín hiệu tiếng nói
        voiced = []
        for t in range(len(index_real)):
            for j in range(index_real[t][0], index_real[t][1]):
                voiced.append(sig[j])
        voiced = np.asarray(voiced)
        
        # ghi kết quả vào file .lab
        index = (index_real.flatten()/fs).round(2)
        end = round(len(sig)/fs, 2)
        write_file_lab(filename[i], index, end)
        
        # ghi vào file wav mới chỉ có tín hiệu tiếng nói
        scipy.io.wavfile.write(froot + "\OutputSTE" + filename[i] + "_only_voice.wav",
                           fs,  np.array(voiced, dtype=sig.dtype))
    # Tính ngưỡng chung: tổng(mean+std)/số file.
    std = 0
    mean = 0
    temp = 0
    for i in range(len(threshold)):
        temp += abs(threshold[i][1] + threshold[i][0])
    print("==> Predict threshold based on Lab: ",round(temp/len(threshold),5))
    
    