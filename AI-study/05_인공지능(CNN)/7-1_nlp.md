# 7. NLP / RNN / LSTM

## NLP (Natural Language Processing)

### 정의

| 단어 | 의미 |
|---|---|
| Natural | 자연의, 천연의 |
| Language | 언어 |
| Processing | 과정, 절차 |
| NLP | 인간의 언어 현상을 컴퓨터와 같은 기계를 이용해서 묘사할 수 있도록 연구하고 이를 구현하는 인공지능의 분야 중 하나 |

### NLP 처리 흐름

| 단계 | 설명 | 예시 |
|---|---|---|
| 사용자 입력 | 사용자로부터 음성 또는 텍스트 입력을 받는 단계 | "What's the weather like today?" |
| 전처리 | 소문자 변환, 약어 표준화 등 입력 데이터 정제 | "what is the weather like today?" |
| 형태소 분석 | 단어를 형태소 단위로 분해하고 품사 태깅 | "what/PRON is/VERB the/DET weather/NOUN..." |
| 벡터화 | 형태소 분석된 텍스트를 숫자 벡터로 변환 | [0, 1, 0, 2, ...] (단어 빈도 벡터) |
| 구문 분석 | 형태소 분석 결과를 기반으로 문장 구조를 분석 | 문법 구조 및 의미 해석 |
| 후처리 | 구문 분석 결과를 기반으로 최종 결과 생성 | "Today's weather is sunny in Seoul" |
| 결과 출력 | 후처리된 결과를 사용자에게 전달 | 음성 또는 텍스트로 응답 |

### 사용 이유

| 이유 | 설명 |
|---|---|
| 자동화된 정보 처리 | 문서 분류, 키워드 추출, 요약 작성 등의 작업을 자동화 |
| 고급 검색 기능 | 자연어 질의를 이해하고 적절한 답변을 제공하는 고급 검색 기능 제공 |
| 언어 번역 | 서로 다른 언어 간의 번역을 자동으로 수행 |
| 감성 분석 | 텍스트 데이터에서 긍정/부정/중립 감성을 파악 |
| 대화형 AI | 챗봇, 가상 비서 등 대화형 AI 시스템의 핵심 기술 |
| 음성 인식 및 변환 | 음성 데이터를 텍스트로 변환하거나 텍스트를 음성으로 변환 |

### TensorFlow 코드 예시

```python
import nltk
import os
import shutil
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.utils import to_categorical

# NLTK 데이터 다운로드
shutil.rmtree('/root/nltk_data', ignore_errors=True)
nltk.data.path.append("/root/nltk_data")
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# 데이터셋 로드 (야구 vs 우주 뉴스 분류)
categories = ['rec.sport.baseball', 'sci.space']
newsgroups = fetch_20newsgroups(subset='train', categories=categories)
texts = newsgroups.data
labels = newsgroups.target

# 텍스트 전처리 함수
def preprocess_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

preprocessed_texts = [preprocess_text(text) for text in texts]

# 토크나이저 초기화 및 시퀀스 변환
tokenizer = Tokenizer()
tokenizer.fit_on_texts(preprocessed_texts)
sequences = tokenizer.texts_to_sequences(preprocessed_texts)

# 시퀀스 패딩 및 레이블 원-핫 인코딩
max_length = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen=max_length)
y = to_categorical(labels, num_classes=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 정의
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128),
    GlobalAveragePooling1D(),
    Dense(2, activation='softmax')
])

# 모델 컴파일, 학습, 평가
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")
print(f"Loss: {loss:.4f}")
```

### PyTorch 코드 예시

```python
import nltk
import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

# NLTK 데이터 다운로드
shutil.rmtree('/root/nltk_data', ignore_errors=True)
nltk.data.path.append("/root/nltk_data")
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# 데이터셋 로드
categories = ['rec.sport.baseball', 'sci.space']
newsgroups = fetch_20newsgroups(subset='train', categories=categories)
texts = newsgroups.data
labels = newsgroups.target

# 텍스트 전처리 함수
def preprocess_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

preprocessed_texts = [preprocess_text(text) for text in texts]

# 단어 사전 생성 및 시퀀스 변환
vocab = {"<pad>": 0}
for text in preprocessed_texts:
    for token in text.split():
        if token not in vocab:
            vocab[token] = len(vocab)

sequences = [[vocab.get(token, 0) for token in text.split()] for text in preprocessed_texts]

# 시퀀스 패딩
max_length = max(len(seq) for seq in sequences)
def pad_sequence(seq, max_len):
    if len(seq) < max_len:
        seq = seq + [vocab["<pad>"]] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
    return seq

X = np.array([pad_sequence(seq, max_length) for seq in sequences])
y = np.array(labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 텐서 변환 및 DataLoader
batch_size = 32
train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.long)),
    batch_size=batch_size, shuffle=True
)
test_loader = DataLoader(
    TensorDataset(torch.tensor(X_test, dtype=torch.long), torch.tensor(y_test, dtype=torch.long)),
    batch_size=batch_size
)

# 모델 정의
class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.mean(x, dim=1)
        x = self.fc(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TextClassificationModel(len(vocab), 128, 2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 모델 학습
for epoch in range(10):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(batch_x), batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
```

---

## RNN (Recurrent Neural Network)

### 정의

| 단어 | 의미 |
|---|---|
| Recurrent | 되풀이되는, 반복되는 |
| Neural | 신경의 |
| Network | 망 |
| RNN | 순차적인 데이터(텍스트, 시간 시계열 데이터 등) 처리를 위해 설계된 인공 신경망 |

### RNN 주요 요소

| 요소 | 설명 |
|---|---|
| `x` | 시점마다 입력되는 데이터 ($x_{t-1}$, $x_t$, $x_{t+1}$) |
| `y` | 각 시점에서의 출력 값 |
| `t` | 시퀀스 데이터의 현재 시점 인덱스 |
| `hidden state` | 이전 시점의 정보를 현재 시점으로 전달하는 중간 상태 |
| `$w_x$` | 입력에 적용되는 가중치 |
| `$w_h$` | 이전 hidden state에서 현재로 정보를 전달하는 가중치 |

### 사용 이유

| 항목 | 설명 |
|---|---|
| 기억 능력 | 이전 시점의 정보를 기억하여 현재 시점의 출력에 영향을 미침 |
| 시퀀스 데이터 처리 | 텍스트, 음성, 시계열 데이터 등 순차적 패턴과 종속성을 학습 |
| 시간적 패턴 학습 | 데이터 간의 시간적 관계와 패턴을 학습하여 예측 성능 향상 |
| 연속적인 데이터 처리 | 텍스트 생성, 음성 인식, 기계 번역 등에 활용 |
| 종속성 모델링 | 앞뒤 문맥을 고려한 자연어 처리 작업에 효과적 |
| 다양한 응용 분야 | 텍스트 분류, 감정 분석, 시계열 예측, 음악 생성 등 |

### RNN 수식

$$
h_t = f_W(h_{t-1}, x_t)
$$

| 기호 | 설명 |
|---|---|
| $h_t$ | 새로운 상태 (현재 시점의 hidden state) |
| $f_W$ | 활성화 함수 + 가중치 (퍼셉트론) |
| $h_{t-1}$ | 이전 상태 (이전 시점의 hidden state) |
| $x_t$ | 인풋 벡터 (현재 시점의 입력 값) |

### 기울기 소실 (Vanishing Gradient) 문제

시퀀스가 길어질수록 기울기가 지수적으로 감소하여 초기 입력 데이터의 영향이 사라지는 문제가 발생한다.

| 영향 | 내용 |
|---|---|
| 텍스트 생성 | 장문의 텍스트를 생성할 때 문맥을 유지하기 어려움 |
| 음성 인식 | 긴 음성 데이터를 처리할 때 앞부분의 음성 정보가 손실됨 |

---

## LSTM (Long Short-Term Memory)

### 정의

| 단어 | 의미 |
|---|---|
| Long | 긴 |
| Short | 짧은 |
| Term | 기간 |
| Memory | 기억 |
| LSTM | 기존의 RNN에서 출력과 멀리 있는 정보를 기억할 수 없다는 단점을 보완해 장/단기 기억을 가능하게 설계한 신경망 구조 |

### LSTM 게이트 구조

| 구분 | 설명 |
|---|---|
| Forget Gate | 과거 기억 중 남길 것 결정 |
| Input Gate + 후보 정보 | 새로운 정보를 얼마나 받아들일지 결정 |
| 셀 상태 업데이트 | 이전 기억(Forget) + 새로운 정보(Input) 결합 |
| 하이퍼볼릭 탄젠트 변환 + Output Gate | 최종 hidden state 출력 및 다음 시점 전달 |

### 사용 이유

| 이유 | 설명 |
|---|---|
| 기울기 소실 문제 해결 | RNN의 vanishing gradient 문제를 해결하기 위해 설계 |
| 장기 의존성 학습 | 셀 상태와 게이트 구조를 통해 장기적인 의존성을 효과적으로 학습 |
| 망각과 기억 조절 | 입력/출력/망각 게이트를 사용하여 중요하지 않은 정보는 망각하고 중요한 정보는 기억 |

### TensorFlow 코드 예시

```python
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.data import Dataset

# 텍스트 데이터를 시퀀스로 변환하는 데이터셋 클래스
class TextDataset:
    def __init__(self, text, sequence_length):
        self.text = text
        self.sequence_length = sequence_length
        self.char_to_idx = {ch: i for i, ch in enumerate(sorted(set(text)))}
        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}
        self.data = [self.char_to_idx[ch] for ch in text]

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        return (
            tf.convert_to_tensor(self.data[idx:idx+self.sequence_length], dtype=tf.int32),
            tf.convert_to_tensor(self.data[idx+1:idx+self.sequence_length+1], dtype=tf.int32)
        )

text = "hello world. this is a simple text dataset for LSTM."
sequence_length = 10
dataset = TextDataset(text, sequence_length)

def generator():
    for i in range(len(dataset)):
        yield dataset[i]

dataloader = tf.data.Dataset.from_generator(
    generator,
    output_signature=(
        tf.TensorSpec(shape=(sequence_length,), dtype=tf.int32),
        tf.TensorSpec(shape=(sequence_length,), dtype=tf.int32)
    )
).batch(2).shuffle(buffer_size=len(dataset))

# LSTM 모델
class SimpleLSTM(models.Model):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(SimpleLSTM, self).__init__()
        self.embedding = layers.Embedding(vocab_size, embed_dim)
        self.lstm = layers.LSTM(hidden_dim, return_sequences=True)
        self.fc = layers.Dense(output_dim)

    def call(self, x):
        x = self.embedding(x)
        x = self.lstm(x)
        x = self.fc(x)
        return x

vocab_size = len(dataset.char_to_idx)
embed_dim = 10
hidden_dim = 20
output_dim = vocab_size
model = SimpleLSTM(vocab_size, embed_dim, hidden_dim, output_dim)

# 손실 함수 및 옵티마이저
loss_fn = losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=loss_fn)

# 모델 훈련
num_epochs = 50
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        with tf.GradientTape() as tape:
            outputs = model(inputs)
            loss = loss_fn(targets, outputs)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}')

# 모델 평가
def evaluate(model, dataloader):
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for inputs, targets in dataloader:
        outputs = model(inputs)
        outputs = tf.reshape(outputs, [-1, outputs.shape[-1]])
        targets = tf.reshape(targets, [-1])
        loss = loss_fn(targets, outputs)
        total_loss += loss.numpy() * inputs.shape[0]
        predicted = tf.argmax(outputs, axis=-1, output_type=tf.int32)
        total_correct += tf.reduce_sum(tf.cast(predicted == targets, tf.float32)).numpy()
        total_samples += targets.shape[0]

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples * 100
    print(f'Validation Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%')

evaluate(model, dataloader)

# 텍스트 생성
def generate_text(model, start_text, length):
    chars = [dataset.char_to_idx[ch] for ch in start_text]
    input_seq = tf.convert_to_tensor(chars, dtype=tf.int32)[tf.newaxis, ...]
    generated_text = start_text

    for _ in range(length):
        output = model(input_seq)
        predicted = tf.argmax(output[:, -1, :], axis=-1, output_type=tf.int32)
        next_char = dataset.idx_to_char[predicted.numpy()[0]]
        generated_text += next_char
        input_seq = tf.concat([input_seq[:, 1:], tf.expand_dims(predicted, axis=0)], axis=1)

    return generated_text

start_text = "hello"
generated_text = generate_text(model, start_text, 20)
print(f'Generated Text: {generated_text}')
```

### PyTorch 코드 예시

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# 데이터셋 클래스
class TextDataset(Dataset):
    def __init__(self, text, sequence_length):
        self.text = text
        self.sequence_length = sequence_length
        self.char_to_idx = {ch: i for i, ch in enumerate(sorted(set(text)))}
        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}
        self.data = [self.char_to_idx[ch] for ch in text]

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx:idx+self.sequence_length], dtype=torch.long),
            torch.tensor(self.data[idx+1:idx+self.sequence_length+1], dtype=torch.long)
        )

text = "hello world. this is a simple text dataset for LSTM."
sequence_length = 10
dataset = TextDataset(text, sequence_length)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# LSTM 모델
class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(SimpleLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

vocab_size = len(dataset.char_to_idx)
embed_dim = 10
hidden_dim = 20
output_dim = vocab_size
model = SimpleLSTM(vocab_size, embed_dim, hidden_dim, output_dim)

# 손실 함수 및 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 모델 훈련
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        outputs = outputs.view(-1, outputs.size(-1))
        targets = targets.view(-1)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

# 모델 평가
def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples * 100
    print(f'Validation Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%')

evaluate(model, dataloader)

# 텍스트 생성
def generate_text(model, start_text, length):
    model.eval()
    chars = [dataset.char_to_idx[ch] for ch in start_text]
    input_seq = torch.tensor(chars, dtype=torch.long).unsqueeze(0).to(device)
    generated_text = start_text
    with torch.no_grad():
        for _ in range(length):
            output = model(input_seq)
            _, predicted = torch.max(output[:, -1, :], 1)
            next_char = dataset.idx_to_char[predicted.item()]
            generated_text += next_char
            input_seq = torch.cat([input_seq[:, 1:], predicted.unsqueeze(0)], dim=1)
    return generated_text

start_text = "hello"
generated_text = generate_text(model, start_text, 20)
print(f'Generated Text: {generated_text}')
```
