import scipy
import math
import numpy as np
import scipy.signal
from scipy.fft import fft
import scipy.io.wavfile
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.style.use('bmh')

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
     signal_length = len(sig)                      
     frame_length = win_len * fs                    
     frame_step = win_hop * fs                      
     frames_overlap = frame_length - frame_step     

     num_frames = np.abs(signal_length - frames_overlap) // np.abs(frame_length - frames_overlap)
     rest_samples = np.abs(signal_length - frames_overlap) % np.abs(frame_length - frames_overlap)
    
     if rest_samples != 0: 
         pad_signal_length = int(frame_step - rest_samples)
         z = np.zeros((pad_signal_length))
         pad_signal = np.append(sig, z)
         num_frames += 1
     else:
         pad_signal = sig

     frame_length = int(frame_length)
     frame_step = int(frame_step)
     num_frames = int(num_frames)

     idx1 = np.tile(np.arange(0, frame_length), (num_frames, 1))
     idx2 = np.tile(np.arange(0, num_frames * frame_step, frame_step),
                    (frame_length, 1)).T
     
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
def speech_silence(sig, fs, win_len, win_hop, noise):
    global frames
    frames, fr_len, fr_overlap = framing(sig=sig, fs=fs, 
                                         win_len=win_len, win_hop=win_hop)

    ste = calculate_STE(frames)
    ste = ste / max(ste)
    pred_threshold = predict_threshold(ste)
    print("Threshold computed by Function:", pred_threshold)
    
    ste = np.repeat(ste, fr_len)
    
    vad = np.array(ste > pred_threshold, dtype=sig.dtype)        
    index_voice_raw = get_index_voice(vad)
    filtered = get_index_filter_noise(index_voice_raw, noise*fs)
    
    global index_real
    index_real, filtered_vad = voice_frame(np.asarray(frames.flatten()), 
                                           filtered, fr_len, fr_overlap)
    return ste, filtered_vad, index_real

""" 8. Hàm hamming:
    Tạo cửa sổ hamming với chiều dài M
    Output: w (array) - hệ số giúp chuyển đổi tín hiệu sang cửa sổ hamming
"""
def hamming(M):
    n = np.arange(M)
    w = 0.54 - 0.46 * np.cos(2 * np.pi * n / (M - 1))
    return w

""" 9. Hàm window:
    Chuyển khung tín hiệu từ cửa sổ chữ nhật sang cửa sổ hamming
    fs (int)                : tần số lấy mẫu
    frame (array)           : 1 khung tín hiệu
    win_len (int)           : độ dài cửa sổ (đơn vị giây)
    Output:
    windowed_frame (array)  : 1 khung tín hiệu mới ở cửa số hamming 
"""
def window(fs, frame, win_len):
    ## windowing 
    win = hamming(fs*win_len)
    if(len(win) != len(frame)):
        win = hamming(fs*win_len - 1)
    windowed_frame = win * frame

    return windowed_frame

""" 10. Hàm find_FFT_point:
    Tìm số điểm fft N theo tần số lấy mẫu fs
"""
def find_FFT_point(fs):
    for i in range(10, 20):
        if(2**i < fs and 2**(i+1) > fs):
            print("FFT Points: ", 2**i)
            return 2**i

""" 11. Hàm rfft:
    Hàm tính fft 1 phía từ 1 vector fft cho trước
"""
def rfft(a):
     ffta = np.fft.fft(a)
     rfft = ffta[1: (math.floor(len(ffta)/2)+2)]
     return rfft
 
""" 12. Hàm log_fft:
    Tính fft(khung tín hiệu) => log(abs(fft)) và sinh trục x cho vector fft
"""
def log_fft(fs, windowed_frame, win_len):
    
    X = rfft(windowed_frame)
    log_X = np.log(np.abs(X))
    
    freq_vector = np.linspace(0, fs/2, len(X))
    
    return log_X, freq_vector

""" 13. Hàm liftering:
    Lọc đoạn giá trị ảo ở đầu vector c
"""
def liftering(c):
    N = 30
    for i in range(N):
        c[i] = 0
    return c

""" 14. Hàm cepstrum
    Tìm giá trị vector cepstrum và sinh trục x cho vector đó. 
        frame (array): 1 khung tín hiệu
        fs (int): tần số lấy mẫu
        win_len (int): độ dài khung (đơn vị giây)
    Output:
        cepstrum_pitch (array): mảng giá trị vector cepstrum
        quefrency_vector (array): trục x (index từng phần tử trong cepstrum)
"""
def cepstrum(frame, fs, win_len):
    global log_X, freq_vector
    
    windowed_frame = window(fs, frame, win_len)
    log_X, freq_vector = log_fft(fs, windowed_frame, win_len)
    
    cepstrum = rfft(log_X)
    quefrency_vector = np.linspace(0, win_hop, len(cepstrum))
    
    cepstrum_pitch = liftering(np.copy(cepstrum))
    return cepstrum_pitch, quefrency_vector

""" 15. Hàm calculate_pitch:
    Dựa vào mảng cepstrum đã tính để tìm F0
        cesptrum_pitch (array): mảng giá trị cepstrum (ifft(log(fft)))
        quefrency_vector (array): mảng index của cepstrum
    Ouput:
        max_quefrency_index (float): index tại điểm giá trị cepstrum đạt max
        pitch_frequency (float): F0 tìm được
"""
def calculate_pitch(cepstrum_pitch, quefrency_vector):

    max_quefrency_index = np.argmax(np.abs(cepstrum_pitch))
    pitch_frequency = 1/quefrency_vector[max_quefrency_index]
  
    return max_quefrency_index, pitch_frequency

""" 16. Hàm collect_pitch:
    Thu giá trị F0 từ thuật toán Cepstrum thành 1 mảng giá trị và vẽ đồ thị trung gian
    Output:
        pitches (array): mảng chứa các giá trị F0 của từng khung tín hiệu
"""
def collect_pitch(fs, win_len, win_hop):
    global pitches
    pitches = []
    
    count = 0

    n_cepstrum_pitch, n_quefrency_vector = cepstrum(normalized_sig[:int(fs*win_len)], fs, win_len)
    n_max_quefrency_index, n_pitch_freq = calculate_pitch(n_cepstrum_pitch, n_quefrency_vector)
    f_ax5.plot(n_quefrency_vector, np.abs(n_cepstrum_pitch), '#ff7f0e')
    f_ax5.set_xlabel('Quefrency (s)')
    f_ax5.set_ylabel('Cepstrum Amplitude')
    f_ax5.set_title('Pitch Cepstrum')
    f_ax5.plot(n_quefrency_vector[n_max_quefrency_index], np.abs(n_cepstrum_pitch)[n_max_quefrency_index], "ro")
    f_ax5.text(0.8, 0.9, "Pitch: {0:.2f} Hz".format(n_pitch_freq), horizontalalignment='center', 
                verticalalignment='center', transform=f_ax5.transAxes, color='red')
    
    for signal in index_real:
        speech_frames, fr_len, fr_overlap = framing(normalized_sig[signal[0]:signal[1]], fs=fs, 
                                         win_len=win_len, win_hop=win_hop)
        pitch_per_voice = []
        for speech_frame in speech_frames:
            cepstrum_pitch, quefrency_vector = cepstrum(speech_frame, fs, win_len)
            max_quefrency_index, pitch_freq = calculate_pitch(cepstrum_pitch, quefrency_vector)
            
            if(count == 10):
                
                f_ax4.plot(quefrency_vector, np.abs(cepstrum_pitch), '#ff7f0e')
                f_ax4.set_xlabel('Quefrency (s)')
                f_ax4.set_ylabel('Cepstrum Amplitude')
                f_ax4.set_title('Pitch Cepstrum')
                f_ax4.plot(quefrency_vector[max_quefrency_index], np.abs(cepstrum_pitch)[max_quefrency_index], "ro")
                f_ax4.text(0.8, 0.9, "Pitch: {0:.2f} Hz".format(pitch_freq), horizontalalignment='center', 
                verticalalignment='center', transform=f_ax4.transAxes, color='red')
            count += 1
            
            if(pitch_freq >= 70 and pitch_freq <= 400):
                pitch_per_voice.append(pitch_freq)
            else:
                pitch_per_voice.append(0)
        pitches.append(pitch_per_voice)
    return pitches

""" 17. Hàm pitch_plot:
    Sinh ra mảng F0 để vẽ đồ thị
"""
def pitch_plot(fs, win_len, win_hop, ref):
    y_values = np.zeros(len(frames))
    for index, value in enumerate(index_real): 
        start = int(value[0] / (fs * (win_hop)))
        end = int(start + len(ref[index]))
        y_values[start:end] = ref[index]
    y_values = medfilt(y_values, 3)
    return y_values

""" 18. Hàm findpeaks_fft:
    Phương pháp fft -> findpeaks để tìm ra các pitch cho 1 frame
        sig (array): mảng tín hiệu gốc (voice+silence)
        fs (int): tần số lấy mẫu
        win_len(float): độ dài khung (đơn vị giây)
        N (int): số điểm FFT
    Output:
        peaks (array): mảng chứa index của các đỉnh tìm được từ mảng fft
        freq (array): mảng chứa index của vector fft
        rfft_speech (array): mảng chứa các giá trị của vector fft
"""
def findpeaks_fft(sig, fs, win_len, N):
    space = int(np.round(2000 / (0.5 * fs) * N, 0))
    
    speech_window = window(fs, sig, win_len)
    fft_speech = np.abs(fft(speech_window, N))
    rfft_speech = fft_speech[:len(fft_speech)//2]
    rfft_speech[1:] = 2 * rfft_speech[1:]
    rfft_speech = rfft_speech/max(rfft_speech)
    
    freq = np.linspace(0, fs/2, len(rfft_speech))
    peaks,_ = scipy.signal.find_peaks(rfft_speech[:space], prominence=0.1)
    
    return peaks, freq, rfft_speech

""" 19. Hàm implement_findpeakFFT:
    Triển khai phương pháp fft -> findpeaks để tìm ra các pitch và vẽ đồ thị trung gian.
    Output:
        arr (array): mảng chứa giá trị các F0 của từng khung
"""
def implement_findpeaksFFT(sig, fs, win_len, win_hop):
    N = find_FFT_point(fs)
    error = np.round(fs / N * 2 + 3, 1)   
    space = space = int(np.round(6000 / (0.5 * fs) * N, 0))
    global arr
    arr = []   
    
    count = 0
    
    non_peaks, non_freq, non_rfft_speech = findpeaks_fft(normalized_sig[10000:10000+int(fs*win_len)], fs, win_len, N)
    f_ax2.set_xlabel('Frequency(Hz)')
    f_ax2.set_ylabel('Amplitude')
    f_ax2.plot(non_freq[:space], non_rfft_speech[:space], '#9467bd')
    f_ax2.plot(non_freq[non_peaks], non_rfft_speech[non_peaks], 'xr')
    f_ax2.set_title('Middle result Findpeaks FFT')
                
    for signal in index_real:
        speech_frames, fr_len, fr_overlap = framing(normalized_sig[signal[0]:signal[1]], fs=fs, 
                                         win_len=win_len, win_hop=win_hop)
        pitch_per_voice = []
        for speech_frame in speech_frames:
            
            peaks, freq, rfft_speech = findpeaks_fft(speech_frame, fs, win_len, N)
            
            if(count == 10):
                f_ax1.set_xlabel('Frequency(Hz)')
                f_ax1.set_ylabel('Amplitude')
                f_ax1.plot(freq[:space], rfft_speech[:space], '#9467bd')
                f_ax1.plot(freq[peaks], rfft_speech[peaks], 'xr')
                f_ax1.set_title('Middle result Findpeaks FFT')
            count += 1
            
            temp = False
            for i in range(1, len(peaks)-1):
                pa = freq[peaks[i]] - freq[peaks[i-1]]
                pb = freq[peaks[i+1]] - freq[peaks[i]]
                if (pa > 70 and pa < 400 and pb > 70 and pb < 400 and np.abs(pb-pa) <= error):
                    pitch_per_voice.append((pa + pb)/2)
                    temp = True
                    break
            if(temp != True):
                pitch_per_voice.append(0)
                
        arr.append(pitch_per_voice)    
    return arr

""" 20. Hàm medfilt:
    Lọc trung vị mảng giá trị x với số phần tử khung lọc là k
"""
def medfilt(x, k): 
    assert k % 2 == 1
    assert x.ndim == 1 
    k2 = (k - 1) // 2
    y = np.zeros ((len (x), k), dtype=x.dtype)
    y[:,k2] = x
    for i in range (k2):
        j = k2 - i
        y[j:,i] = x[:-j]
        y[:j,i] = x[0]
        y[:-j,-(i+1)] = x[j:]
        y[-j:,-(i+1)] = x[-1]
    return np.median (y, axis=1)

""" 21. Hàm F0_mean_std:
    Tính độ lệch chuẩn (std) và giá trị trung bình (mean) của mảng giá trị F0.
    Args:
         y_values            (array) : giá trị F0 của các khung tín hiệu.
    Returns:
         std, mean          (float) : độ lệch chuẩn và trung bình của mảng F0
"""
def F0_mean_std(y_values):
    count = 0
    s = 0
    std = 0
    for i in y_values:
        if(i != 0):
            s += i
            count += 1
    mean = s/count
    
    for i in y_values:
        if(i != 0):
            std += (i - mean)**2        
    std = np.sqrt(std/count)
    print("F0_mean: ", mean)
    print("F0_std: ", std)

""" 22. Hàm multi-plots:
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
        
        if(j == 5):
            y_another = pitch_plot(fs, win_len, win_hop, arr)
            plt.subplot(plot_rows, 1, j+1)
            plt.xlabel('Time(s)')
            plt.ylabel('Frequency Amplitude')
            #plt.scatter([i/(fs*win_len/win_hop) for i in range(0, len(y_another), step)], y_another, 10)
            plt.scatter([(i*win_hop) for i in range(0, len(y_another), step)], y_another, 3, c=['#9467bd'])
            plt.gca().set_title(titles[j])
            
            
            f_ax3.set_xlabel('Time(s)')
            f_ax3.set_ylabel('Frequency Amplitude')
            f_ax3.scatter([(i*win_hop) for i in range(0, len(y_another), step)], y_another, 5, c=['#9467bd'])
            f_ax3.set_title(titles[j])
            
        elif(j == 4):
            y_values = pitch_plot(fs, win_len, win_hop, pitches)
            plt.subplot(plot_rows, 1, j+1)
            plt.xlabel('Time(s)')
            plt.ylabel('Frequency Amplitude')
            plt.scatter([(i*win_hop) for i in range(0, len(y_values), step)], y_values, 3, c=['#ff7f0e'])
            plt.gca().set_title(titles[j])
            
            f_ax6.set_xlabel('Time(s)')
            f_ax6.set_ylabel('Frequency Amplitude')
            f_ax6.scatter([(i*win_hop) for i in range(0, len(y_values), step)], y_values, 5, c=['#ff7f0e'])
            f_ax6.set_title(titles[j])
            
        # vẽ output signal (tín hiệu gốc và các đường phân chia tiếng nói)
        elif(j == 3):
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
                       linestyles='dashed', colors='red', label='By teacher')
            plt.legend(loc='best')
            plt.gca().set_title(titles[j])
        else:
            plt.subplot(plot_rows, 1, j+1)
            y = data[j]
            plt.xlabel('Time(s)')
            if (j == 0):
                plt.plot([i/fs for i in range(0, len(y), step)], y)
                plt.ylabel('Signal Amplitude')
                
                f_ax7.set_xlabel('Time(s)')
                f_ax7.set_ylabel('Signal Amplitude')
                f_ax7.plot([i/fs for i in range(0, len(y), step)], y, '#cc5062')
                f_ax7.set_title(titles[j])
            else:
                plt.plot([i/(win_len*fs/win_hop) for i in range(0, len(y), step)], y, colors[j])
            plt.gca().set_title(titles[j])     
    plt.legend(loc='best')
    plt.show()

""" 23. Hàm read_file_lab:
    Đọc dữ liệu từ filename (.lab) trả về mảng chỉ số speech/silence (giây)
"""
def read_file_lab(filename):
    froot = "E:\DUT\Kì_5\Xử lí THS\Tín hiệu mẫu"
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


if __name__ == "__main__":
    # tạo tên file
    froot = "E:\DUT\Kì_5\Xử lí THS\Tín hiệu mẫu"
    filename = ["\FTN30", "\FQT42", "\MTT44", "\MDV45"]
    threshold = []
    win_len = 0.03
    win_hop = 0.015
    for i in range(len(filename)):
        
        # initialize plots
        fig = plt.figure(constrained_layout=True, figsize=(25, 14))
        spec = gridspec.GridSpec(ncols=1, nrows=7, figure=fig)
        f_ax1 = fig.add_subplot(spec[0, 0])
        f_ax2 = fig.add_subplot(spec[1, 0])
        f_ax3 = fig.add_subplot(spec[2, 0])
        f_ax4 = fig.add_subplot(spec[3, 0])
        f_ax5 = fig.add_subplot(spec[4, 0])
        f_ax6 = fig.add_subplot(spec[5, 0])
        f_ax7 = fig.add_subplot(spec[6, 0])
        
        print("Figure", i + 1, ": STE", filename[i])
        fname = froot + filename[i] + '.wav'
        global sig, normalized_sig
        fs, sig = scipy.io.wavfile.read(fname)
        normalized_sig = sig/max(abs(sig))

        lab = read_file_lab(filename[i])
        
        # lấy mảng STE, vad, mốc chỉ số đánh dấu tiếng nói trong tín hiệu gốc
        energy, vad, index_real = speech_silence(normalized_sig, fs, 
                                   win_len, win_hop, noise=0.3)
        
        pitches = collect_pitch(fs, win_len, win_hop)
        y_values = pitch_plot(fs, win_len, win_hop, pitches)
        
        arr = implement_findpeaksFFT(sig, fs, win_len, win_hop)
        y_another = pitch_plot(fs, win_len, win_hop, arr)
        
        print("Cepstrum:")
        F0_mean_std(y_values)
        print("FFT findpeaks:")
        F0_mean_std(y_another)
        
        # xuất đồ thị minh họa
        fig_name = "Figure: " + str(i + 1) + " " + filename[i]
        multi_plots(fig_name, data=[sig, energy, vad, index_real], lab=lab,
                titles=["Input signal (voiced + silence)", "Short time energy",
                        "Voice activity detection", "Ouput signal (voice marked)",
                        "Cepstrum algorithm", "Findpeaks FFT algorithm"],
                fs=fs, plot_rows=6, win_len=win_len, win_hop=win_hop, step=1)

       #File huấn luyện: "\MDA01", "\FVA02", "\MAB03", "\FTB06" 
    
    