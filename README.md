# MNIST 딥러닝 예제

|순서|
|---|
|[1. 개요](#1-개요)|
|[2. 이미지 전처리](#2-이미지-전처리)|
|&nbsp;&nbsp;&nbsp;&nbsp;[I. 실행 환경](#i-실행-환경)|
|&nbsp;&nbsp;&nbsp;&nbsp;[II. 이미지 확인](#ii-이미지-확인)|
|&nbsp;&nbsp;&nbsp;&nbsp;[III. 이미지 전처리](#iii-이미지-전처리)|
|[3. 모델 설계 및 실행](#3-모델-설계-및-실행)|
|&nbsp;&nbsp;&nbsp;&nbsp;[I. 모델 설계](#i-모델-설계)|
|&nbsp;&nbsp;&nbsp;&nbsp;[II. 모델 컴파일](#ii-모델-컴파일)|
|&nbsp;&nbsp;&nbsp;&nbsp;[III. 학습 진행](#iii-학습-진행)|
|&nbsp;&nbsp;&nbsp;&nbsp;[IV. 결과 확인](#iv-결과-확인)|
|[4. 결론](#4-결론)|

## 1. 개요
- 목표: 합성곱 신경망(CNN)의 기초적인 사용법 익히기

- MNIST란?
    - 'Modified National Institute of Standards and Technology'의 약자로, 미국 국립표준기술연구소에서 인구조사국 직원 및 중학교 학생들로부터 숫자 손글씨를 취합하여 만들어진 데이터셋
    - 기계 학습 분야의 트레이닝 및 테스트에 널리 사용되어 대중적, 케라스에서 손쉽게 불러올 수 있어서 이용이 편리함.
    - 60,000개의 트레이닝 이미지와 10,000개의 테스트 이미지로 구성되어 있음

- CNN(Convolution Neural Network)
    - 합성곱 신경망은 딥러닝에서 주로 이미지나 영상 데이터를 처리할 때 쓰이며, 이름에 보이듯 Convolution이라는 전처리 작업이 들어가는 인공 신경망 모델
    - 이전에 쓰이던 일반 DNN(Deep Neural Network)은 1차원 형태의 데이터를 입력하기 때문에 2차원 이미지 정보를 평탄화시킴. 이 과정에서 이미지의 공간적/지역적 정보가 손실되되며, 추상화과정 없이 바로 연산과정으로 넘어가기 때문에 학습시간과 능률의 효율성이 저하

## 2. 이미지 전처리

### I. 실행 환경
- 딥러닝을 실행하기 위한 라이브러리로 tensorflow와 keras를 사용
- 결과값을 보기 위해 matplotlib의 pylot을 사용
- google colaboratory 환경에서 실행
- keras에 들어있는 mnist 데이터셋을 에셋으로 사용

### II. 이미지 확인
```
plt.imshow(train_images[0], cmap='Greys')
plt.xticks()
plt.yticks()
plt.tick_params(labelcolor='white', color='white') #다크모드용 색 지정
plt.show()
print("class: %d" %(train_labels[0]))
```
![다운로드](https://user-images.githubusercontent.com/101073973/208300886-20931a7c-2608-4174-817c-c49071df2140.png)

그레이스케일 형태의 이미지셋이 지정되어 나왔음을 알 수 있음
다크 모드 브라우저를 사용하는데 plt로 뽑아낸 글씨는 색 변환이 되지 않아 따로 흰색으로 색 지정

### III. 이미지 전처리
- 이미 28x28픽셀 사이즈로 개별 이미지의 크기가 작기 때문에 그레이스케일로 불러와진 이미지에서 색상 값만을 처리함
- 그레이스케일 이미지의 색상값은 0~255 사이의 값이므로 이를 255로 나누어 0~1 사이의 값으로 크기를 줄임
- 전처리한 이미지는 심링크로 코랩 내의 폴더에 저장
```
env_path = '/content/notebooks'
os.symlink('/content/drive/My Drive/Colab Notebooks/env1', env_path)
sys.path.insert(0, env_path)
```

## 3. 모델 설계 및 실행
### I. 모델 설계
- 시퀀시얼 모델을 적용하여 순차적으로 모델이 진행되게끔 설계
```
model=models.Sequential()

model.add(layers.Conv2D(32, (4, 4), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64, (4,4), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64, (4,4), activation='relu'))
model.add(layers.Dropout(0.2))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='sigmoid'))
model.add(layers.Dense(10, activation='softmax'))     
```
- 형성된 모델 형태 확인
```
keras.utils.plot_model(model, "test_model.png", show_shapes=True)
```
![다운로드 (1)](https://user-images.githubusercontent.com/101073973/208301802-5c187be7-8978-465a-9a78-59fabdf356d2.png)

- callback 함수를 이용하여 5번째 에포크마다 가중치를 저장
- 에포크 횟수는 50회이므로 EarlyStopping을 이용한 조기 종료 기능은 설정하지 않음

### II. 모델 컴파일
- 옵티마이저는 아담(Adam; Adaptive Moment Esimation) 사용
    - Momentum 와 RMSProp 두가지를 섞어 쓴 알고리즘으로, 진행하던 속도에 관성을 주고, 최근 경로의 곡면의 변화량에 따른 적응적 학습률을 갖는 알고리즘. 광범위한 아키덱처를 가진 서로 다른 신경망에서 유효하게 작동하기 때문에 자주 사용되는 방식. 스텝사이즈를 적절하게 가지며 Momentum 방식처럼 지금까지 계산해온 기울기의 지수 평균을 저장하며, RMSProp처럼 기울기의 제곱값에 지수평균을 저장
- 손실 함수는 sparse categorical crossentropy로 적용하여 정수로 지정된 라벨에 대응
```
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

### III. 학습 진행
- 모델 fit 진행

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;model.fit(train_images, train_labels, epochs=50)
```
Epoch 1/50
1875/1875 [==============================] - 7s 3ms/step - loss: 0.2706 - accuracy: 0.9280
Epoch 2/50
1875/1875 [==============================] - 7s 4ms/step - loss: 0.0869 - accuracy: 0.9752
Epoch 3/50
1875/1875 [==============================] - 7s 4ms/step - loss: 0.0601 - accuracy: 0.9828
                                        .
                                        .
                                        .
Epoch 47/50
1875/1875 [==============================] - 6s 3ms/step - loss: 0.0067 - accuracy: 0.9979
Epoch 48/50
1875/1875 [==============================] - 6s 3ms/step - loss: 0.0054 - accuracy: 0.9985
Epoch 49/50
1875/1875 [==============================] - 6s 3ms/step - loss: 0.0037 - accuracy: 0.9990
Epoch 50/50
1875/1875 [==============================] - 6s 3ms/step - loss: 0.0057 - accuracy: 0.9981
```
- 모델 평가

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;model.evaluate(test_images, test_labels)
```
313/313 [==============================] - 1s 3ms/step - loss: 0.0442 - accuracy: 0.9928
```
- history 기능 적용

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;history = model.fit(train_images, train_labels, epochs=50, validation_split=0.2, callbacks=[cp_callback])
```
Epoch 1/50
1500/1500 [==============================] - 6s 4ms/step - loss: 0.0050 - accuracy: 0.9985 - val_loss: 0.0023 - val_accuracy: 0.9995
Epoch 2/50
1500/1500 [==============================] - 7s 4ms/step - loss: 0.0045 - accuracy: 0.9985 - val_loss: 0.0028 - val_accuracy: 0.9992
Epoch 3/50
1500/1500 [==============================] - 6s 4ms/step - loss: 0.0066 - accuracy: 0.9981 - val_loss: 0.0031 - val_accuracy: 0.9992
                                        .
                                        .
                                        .
Epoch 48/50
1500/1500 [==============================] - 6s 4ms/step - loss: 0.0049 - accuracy: 0.9987 - val_loss: 0.0136 - val_accuracy: 0.9970
Epoch 49/50
1500/1500 [==============================] - 6s 4ms/step - loss: 0.0039 - accuracy: 0.9989 - val_loss: 0.0139 - val_accuracy: 0.9970
Epoch 50/50
1500/1500 [==============================] - 6s 4ms/step - loss: 0.0026 - accuracy: 0.9992 - val_loss: 0.0116 - val_accuracy: 0.9973
```

### IV. 결과 확인
- 학습한 내용을 h5 파일로 저장하고 임의의 데이터를 불러와 학습결과가 맞는지 확인
```
model.save('mnist/mnist_model.h5')
```
```
test_images[0].shape
x = test_images[0].reshape(1, 28, 28, 1)
y = loaded_model.predict(x)
np.argmax(y)
```
> 7
- 이미지 확인 결과</br>
![다운로드 (2)](https://user-images.githubusercontent.com/101073973/208304692-0c0c0783-0aaf-4b5e-8a1f-8b7f5141953d.png)</br>
정상적으로 글자를 인식한 것을 확인함
- 그래프를 작성하여 학습용 데이터와 테스트용 데이터의 손실 및 정확도가 얼마나 일치하는지 확인
- pyplot 기능을 사용하였으며 아래는 손실 그래프
```
# bo: 파란색 점
plt.plot(epochs, loss, 'bo', label='Training loss')
# b : 파란 실선
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.tick_params(labelcolor='white', color='white')
plt.legend()
plt.show()
```
![다운로드 (3)](https://user-images.githubusercontent.com/101073973/208304731-93b73a32-6aac-4260-9378-4fa46c18dcbd.png)
- 같은 방식으로 정확도 그래프도 만들어 확인

![다운로드 (4)](https://user-images.githubusercontent.com/101073973/208304733-d3d8daca-a6f4-4229-95bb-e258d39af342.png)

- 에포크를 반복하면서 20회 전후로 차이가 벌어지는 것이 보이는 것으로 보아 20회 전후의 모델이 가장 정확하지 않을까 생각됨

## 4. 결론
- tensorflow 및 keras 라이브러리를 이용해 CNN 모델 학습을 진행해보았음
- 여러 회차 학습을 진행시켜 예상보다 높은 정확도를 얻을 수 있었지만, EarlyStopping 등의 기능을 사용하지 않고 진행하여 어느 지점이 가장 정확도가 높을지 파악하는 방식은 추후 확인해야 함
- 동일한 방식을 통하여 보다 클래스도 많고 복잡한 문자에 적용해 볼 계