# mnist-Handwritten-digit-recognition
# 一、說明
深度學習就是透過各種類神經網路，本專題會使用多層感知器(MLP)、卷積神經網路(CNN)、循環神經網路(RNN)透過mnist資料集產出訓練出來的值，將一大堆的數據輸入神經網路中，讓電腦透過大量數據的訓練找出規律自動學習，最後讓電腦能依據自動學習累積的經驗作出預測。

# 二、相關文章
多層感知器(MLP)：多層感知器是由多層人工神經元組成的類神經網路, 在 MINST 資料集的手寫數字辨識中要用到的 MLP 為如下具有輸入層, 一個隱藏層, 以及輸出層的類神經網路。
![image](https://github.com/LonelyCaesar/mnist-Handwritten-digit-recognition/assets/101235367/e6df9c0a-662d-4f9b-8b25-0c586c6fdbf9)

為了提高學習的準確率，神經網路更發展到有一個輸入層、一個或多個隱藏層及一個輸出層的多層感知器(MLP)。

	輸入層：數字圖片是一張28*28的圖片、共有784個神經元所組成了神經網路第一層，數值的範圍介於0~1之間。灰階0代表灰色、1代表白色，又稱為激勵值，數值越大則該神經元就越亮。

	輸出層：完成輸入層後先不管其他層的內容，我們來看看他最右方的輸入層，也就是最後判斷的結果，其中有10個神經元，各代表數字0~9，其中也有代表的激勵值。

	隱藏層：為了方便說明，在這裡我們設計了兩個隱藏層，每層有16個神經元。在真實的案例中可依據需求設置調整隱藏層與神經元的數量。

卷積神經網路(CNN)：它是目前深度神經網路(Deep Neural Network)領域發展的主力，在圖片辨別上甚至可以做到比人類還精準之程度。

在真實的案例中可依據需求設置調整隱藏層與神經元的數量。
卷積神經網路(CNN)：它是目前深度神經網路(Deep Neural Network)領域發展的主力，在圖片辨別上甚至可以做到比人類還精準之程度。

	結構圖，和多層感知器相比較，卷積神經網路增加卷積層1、池化層1、卷積層2、池化層2，提取特徵後再以平坦層將特徵輸入神經網路中。

以下使用MNIST資料及進行說明：
![image](https://github.com/LonelyCaesar/mnist-Handwritten-digit-recognition/assets/101235367/9151b959-8b5f-4529-8556-82c3a626c468)

圖片中最上方有卷積層1、池化層1、卷積層2、池化層2，將原始的圖片以卷積、池化處理後產生更多的特徵小圖片，作為輸入的神經元。

	卷積層：是將原始圖片與特定的濾鏡(Feature Detector)進行卷積運算，你也可以將卷積運算看成是原始圖片濾鏡特效的處理，filters可以設定濾鏡數目，kernel_size可以設定濾鏡(filter)大小，每一個濾鏡都會以亂數處理的方式產生不同的卷積運算，因此可以得到不同的濾鏡特效效果，增加圖片數量。

	池化層：是採用Max Pooling，指挑出矩陣當中的最大值，相當於只挑出圖片局部最明顯的特徵，這樣就可以縮減卷積層產生的卷積運算圖片數量。

循環神經網路(RNN)：它是「自然語言處理」領域最常使用的神經網路模型，LSTM因為RNN前面的輸入和後面的輸入具有關連性，即可以建立回饋迴路。也可以用於語言翻譯、情緒分析、氣象預測及股票交易等。

	結構圖：循環神經網路中主要有三種模型，分別是SimpleRNN、LSTM和GRU。因為SimpleRNN超簡單，效果不夠好，記不住長期的事情，所以又發展出長短記憶網路(LSTM)，然後LSTM又被簡化為閘式循環網路GRU。

![image](https://github.com/LonelyCaesar/mnist-Handwritten-digit-recognition/assets/101235367/759762d1-f3d3-4a9e-8fb0-e9b7f484aad1)

如圖共有三個時間點依序是t-1、t、t+1，在t的時間點：

	X1是神經網路t時間點的輸入，Ot是神經網路t時間點的輸出。

	(U,V,W)都是神經網路共用的參數，W參數是神經網路t-1時間點輸出，並且也作為神經網路t時間點的輸入。

	S1是隱藏狀態，代表神經網路上的記憶，是神經網路目前時間點的輸入X1加上上個時間點的狀態St-1，再加上U與W的參數，共同評估之結果：St = f(U * Xt + W * St-1)簡單來說就是前面的狀態會影響現在的狀態，現在的狀態也會影響以後的狀態。

# 三、實作
MNIST資料集是由紐約大學 Yann Le Cun 教授蒐集整理很多0~9的手寫數字圖片所形成的資料集，這是一個大型手寫數字資料庫，對於機器學習學者來說是初學者，圖片每張大小為28*28、皆為灰階影像，每個像素為0~255之數值、資料庫當中包含了60000筆的訓練資料、10000筆的測試資料。在MNIST資料集中，每一筆資料都是由下載好的mnist的資料實作出成果。
![image](https://github.com/LonelyCaesar/mnist-Handwritten-digit-recognition/assets/101235367/6eaefabf-f3b5-4d60-b6e7-e4c6ea6ea2c3)

1.	使用多層感知(MLP)進行辨識訓練：
(1) 建立模型與資料結構：
#### 程式碼
```python
#導入相關套件
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.datasets import mnist # keras.datasets：載入MNIST資料集
from keras.models import Sequential # Keras：建立訓練模型
from keras.layers import Dense, Activation, Dropout
from keras.utils import np_utils
import numpy as np # Numpy：矩陣運算
%matplotlib inline # matplotlib.pyplot 將資料視覺化，可以圖表呈現結果
import matplotlib.pyplot as plt
(x_train_image, y_train_label), (x_test_image, y_test_label) = mnist.load_data() # 呼叫 load_data() 載入 MNIST 資料集
nb_classes = 10 # 類別的數目
x_train_image = x_train_image.reshape(60000, 784).astype('float32')
x_test_image = x_test_image.reshape(10000, 784).astype('float32')
# 壓縮圖片顏色至0 ~ 1
x_train_image /= 255
x_test_image /= 255
#依分類數量將圖片標籤轉換格式的陣列
y_train_cat = np_utils.to_categorical(y_train_label, nb_classes)
y_test_cat = np_utils.to_categorical(y_test_label, nb_classes)
model = Sequential()
model.add(Dense(50, input_shape=(784,)))
model.add(Dense(units=nb_classes))
model.add(Activation('softmax'))
# 定義定義損失函數、優化函數及成效衡量指標
model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
model.summary()
```
#### 執行結果
![image](https://github.com/LonelyCaesar/mnist-Handwritten-digit-recognition/assets/101235367/991c9063-9246-4ce1-9c2d-d6e7745b8e47)

